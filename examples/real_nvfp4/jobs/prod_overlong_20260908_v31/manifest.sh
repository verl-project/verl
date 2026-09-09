#!/usr/bin/env bash
# shellcheck disable=SC2034

# Two matched long runs, both R3 on / TIS on / dynamic sampling on, strict
# Minerva verifier:
#   ARM=w4a4  real NVFP4 on the routed experts only, first 2 and last 4 layers
#             left in BF16 -- the same carve-out NeMo-RL's R3-on arm uses.
#   ARM=bf16  the matched BF16 control.
# See README.md for why this bundle reuses v23's image instead of building one.

readonly RN4PT_VERSION=${RN4PT_VERSION_OVERRIDE:-verl_real_nvfp4_prod_overlong_20260908_v31}
readonly RN4PT_JOB_LABEL=${RN4PT_JOB_LABEL_OVERRIDE:-v31}
readonly RN4PT_WORKSPACE=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031
readonly RN4PT_VERL=$RN4PT_WORKSPACE/verl_nvfp4_e2e_r3_20260831_v3
readonly RN4PT_ROOT=$RN4PT_VERL/examples/real_nvfp4
readonly RN4PT_BUNDLE=${RN4PT_BUNDLE_OVERRIDE:-$RN4PT_ROOT/jobs/prod_overlong_20260908_v31}
readonly RN4PT_JOB_IMPL=$RN4PT_ROOT/jobs/prod_overlong_20260908_v31
readonly RN4PT_BUILD_JOB_IMPL=$RN4PT_ROOT/jobs/prod_overlong_20260908_v31
readonly RN4PT_STATE=$RN4PT_WORKSPACE/run_state/$RN4PT_VERSION
readonly RN4PT_LOGS=$RN4PT_WORKSPACE/ray_log/$RN4PT_VERSION
readonly RN4PT_CHECKPOINTS=$RN4PT_WORKSPACE/checkpoints/DAPO-NVFP4-QAT/$RN4PT_VERSION

readonly RN4PT_BASE_IMAGE=${RN4PT_BASE_IMAGE_OVERRIDE:-/lustre/fsw/general_sa/shuazhang/images/verl.vllm0202.mcore0161.te215.realnvfp4.cgruntime.20260829.v14.sqsh}
# This bundle carries a runtime payload change (the R3 capture hook is now
# installed for any rollout that returns routed experts, not just the NVFP4
# one), so it builds its own image rather than inheriting v23's evidence.
readonly RN4PT_RUNTIME_IMAGE=${RN4PT_RUNTIME_IMAGE_OVERRIDE:-/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.fi0615p1.prod.20260908.v31.sqsh}
readonly RN4PT_IMAGE_ENV=${RN4PT_IMAGE_ENV_OVERRIDE:-/opt/verl-rn4pt-20260908-v31}
readonly RN4PT_IMAGE_PYTHON=$RN4PT_IMAGE_ENV/.venv/bin/python
readonly RN4PT_NETRC=/home/shuazhang/.netrc
readonly RN4PT_MOUNTS=/lustre/fsw/general_sa/shuazhang:/lustre/fsw/general_sa/shuazhang,/home/shuazhang/.netrc:/root/.netrc

readonly RN4PT_ACCOUNT=general_sa
readonly RN4PT_PARTITIONS=36x2-a01r,tcpo,batch

readonly RN4PT_PROJECT=DAPO-NVFP4-QAT
readonly RN4PT_VERL_BASE_COMMIT=b356f4301e67c9896c9f1690785e096242c08705
readonly RN4PT_RECIPE_COMMIT=e7f889574b8301cc0f0fc1d57c6d67f31ffeb689
readonly RN4PT_TE_COMMIT=e7c550c5f80636cf841a8204b1d6f85a5f3f28b7
readonly RN4PT_TE_VERSION=2.18.0+e7c550c5
readonly RN4PT_VLLM_DISABLE_TRTLLM_MOE_PDL=1
RN4PT_SOURCE_COMMIT=$(git -C "$RN4PT_VERL" rev-parse HEAD)
readonly RN4PT_SOURCE_COMMIT
RN4PT_LOCK_SHA256=$(sha256sum "$RN4PT_VERL/uv.lock" | awk '{print $1}')
readonly RN4PT_LOCK_SHA256
readonly RN4PT_PROBE_PASS=$RN4PT_STATE/probe.pass
readonly RN4PT_BUILD_PASS=$RN4PT_STATE/build.pass
readonly RN4PT_PREFLIGHT_PASS=$RN4PT_STATE/preflight.pass
readonly RN4PT_SMOKE_PASS=$RN4PT_STATE/smoke.pass

# ---------------------------------------------------------------- arm settings
# Both arms: R3 on, token-level TIS on, dynamic sampling on, strict verifier.
readonly LONG_FILTER_GROUPS=True
readonly LONG_MAX_GEN_BATCHES=20
readonly LONG_GEN_PROMPT_BSZ_MULT=3
readonly LONG_ROLLOUT_IS=token
readonly LONG_STRICT_MINERVA=1
readonly RN4PT_EXP_TAG=${RN4PT_EXP_TAG_OVERRIDE:-20260908_v31}
# DAPO's soft overlong punishment, matching NeMo-RL's arm. Leaving it off is what
# let 12.8% of W4A4 responses run to the 20480 cap against BF16's 0.7%, and since
# a batch's wall clock is set by its slowest sequence that turned a per-token
# rollout win into a per-batch loss.
readonly LONG_OVERLONG_PENALTY=True
readonly LONG_OVERLONG_BUFFER_LEN=512
readonly LONG_OVERLONG_PENALTY_FACTOR=1.0
# NeMo-RL's R3-on arm quantizes the routed experts only, leaving the first 2 and
# last 4 decoder layers in BF16, so 42 of 48 MoE layers carry NVFP4 MLP GEMMs.
readonly LONG_BF16_LAYERS_AT_START=2
readonly LONG_BF16_LAYERS_AT_END=4
readonly LONG_NVFP4_MLP_LAYERS=42
readonly LONG_NVFP4_BF16_MLP_LAYERS=6
readonly LONG_NVFP4_SCOPE=routed_expert_mlp_first${LONG_BF16_LAYERS_AT_START}_last${LONG_BF16_LAYERS_AT_END}

# Chunk boundaries are per-arm and shrink as response length grows, because step
# time is roughly `0.42 + 0.00102*len` min/step for BF16 and `1.25 + 0.00058*len`
# for W4A4 (fitted from the measured 40-step chunk durations of steps 1-260),
# and the partition cap is a hard 5h. Every target is a save_freq (10) multiple.
# The first seven entries are the original 260-step run; the rest extend it to
# 1000. Entries past the wave actually submitted are a plan, not a commitment --
# re-fit from real timings before releasing the next wave.
# Uniform 40-step chunks. Fitted step time is 1.50 + 0.00064*len min for W4A4,
# so 40 steps stays inside the 5h partition cap up to about 7k tokens; past that
# these entries are a plan to re-fit from real timings, not a commitment. The
# overlong penalty should also keep the length tail much shorter than v25's,
# where 12.8%% of responses ran to the 20480 cap and set the batch wall clock.
# Only the first wave is submitted at a time.
readonly -a LONG_TARGET_STEPS_BF16=(40 80 120 160 200 240 280 320 360 400 440 480 520 560 600 640 680 720 760 800 840 880 920 960 1000)
readonly -a LONG_TARGET_STEPS_W4A4=(40 80 120 160 200 240 280 320 360 400 440 480 520 560 600 640 680 720 760 800 840 880 920 960 1000)

rn4pt_die() { echo "REAL_NVFP4_PERTOKEN_REFUSED: $*" >&2; return 2; }

rn4pt_arm_settings() {
  # Sets the per-arm variables for "$1" (w4a4|bf16). Callers use `readonly`
  # copies afterwards; keep this the single place that knows the difference.
  case "$1" in
    w4a4)
      LONG_PRECISION_MODE=real_nvfp4
      LONG_FIRST_LAST_BF16=True
      LONG_EXP=verl_30b_w4a4_carveout_overlong_8n_${RN4PT_EXP_TAG}
      LONG_SMOKE_EXP=verl_30b_w4a4_overlong_smoke_${RN4PT_EXP_TAG}
      LONG_RUNTIME_ENV=$RN4PT_BUNDLE/runtime_env_w4a4.yaml
      LONG_JOB_TAG=w4a4
      LONG_TARGET_STEPS=("${LONG_TARGET_STEPS_W4A4[@]}")
      ;;
    bf16)
      LONG_PRECISION_MODE=bf16
      # The carve-out is meaningless without quantization, and leaving it on
      # would silently change the BF16 control's transformer config.
      LONG_FIRST_LAST_BF16=False
      LONG_EXP=verl_30b_bf16_overlong_8n_${RN4PT_EXP_TAG}
      LONG_SMOKE_EXP=verl_30b_bf16_overlong_smoke_${RN4PT_EXP_TAG}
      LONG_RUNTIME_ENV=$RN4PT_BUNDLE/runtime_env_bf16.yaml
      LONG_JOB_TAG=bf16
      LONG_TARGET_STEPS=("${LONG_TARGET_STEPS_BF16[@]}")
      ;;
    *) rn4pt_die "unknown arm: $1" || return ;;
  esac
}

rn4pt_validate_static() {
  local path scan_status
  [[ -d "$RN4PT_VERL/.git" || -f "$RN4PT_VERL/.git" ]] || rn4pt_die "worktree missing" || return
  [[ -z "$(git -C "$RN4PT_VERL" status --porcelain --untracked-files=normal)" ]] || \
    rn4pt_die "worktree must be clean so the runtime image and source commit cannot diverge" || return
  # The runtime build uses `git archive $RN4PT_SOURCE_COMMIT`, not the live
  # directory as a container context. An ignored host .venv therefore cannot
  # enter the image; only refuse one that somebody accidentally tracked.
  if git -C "$RN4PT_VERL" ls-files --error-unmatch .venv >/dev/null 2>&1; then
    rn4pt_die "tracked worktree .venv would enter the runtime image" || return
  fi
  git -C "$RN4PT_VERL" merge-base --is-ancestor "$RN4PT_VERL_BASE_COMMIT" HEAD || \
    rn4pt_die "worktree is not based on the audited latest Verl commit" || return
  [[ -s "$RN4PT_BASE_IMAGE" ]] || rn4pt_die "base image missing" || return
  [[ -f "$RN4PT_NETRC" && ! -L "$RN4PT_NETRC" ]] || rn4pt_die "updated W&B netrc missing" || return
  [[ "$(git -C "$RN4PT_VERL/recipe" rev-parse HEAD)" = "$RN4PT_RECIPE_COMMIT" ]] || \
    rn4pt_die "recipe submodule is missing or drifted" || return
  for path in \
    "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" \
    "$RN4PT_ROOT/config/attn_bf16_mlp_nvfp4.yaml" \
    "$RN4PT_ROOT/config/attn_bf16_mlp_nvfp4_first${LONG_BF16_LAYERS_AT_START}_last${LONG_BF16_LAYERS_AT_END}.yaml" \
    "$RN4PT_ROOT/runtime_backports/disable_vllm_trtllm_nvfp4_moe_pdl.py" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/bf16_transport.py" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/r3_monolithic_capture.py" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/vllm_runtime.py" \
    "$RN4PT_BUNDLE/runtime_env_w4a4.yaml" \
    "$RN4PT_BUNDLE/runtime_env_bf16.yaml"; do
    [[ -f "$path" && ! -L "$path" && -s "$path" ]] || rn4pt_die "missing input: $path" || return
  done
  bash -n "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" "$RN4PT_BUNDLE"/*.sh "$RN4PT_JOB_IMPL"/*.job || return
  grep -q 'PHASE" = smoke8' "$RN4PT_JOB_IMPL/train.job" || \
    rn4pt_die "the 8-node production-shape validation phase is missing" || return
  grep -q 'SMOKE8_PASS' "$RN4PT_BUNDLE/submit.sh" || \
    rn4pt_die "a chain may not be released without the 8-node validation" || return
  # Unlike v6..v27, this bundle reinstalls the environment, so the build job has
  # to reapply every site-packages patch the v5..v8 images carried in place --
  # dropping the FA4 guard alone makes importing megatron.core fail outright.
  for marker in VERL_MCORE_FA4_DIST_GUARD 'repeat(num_experts)' \
    'torch.empty(0, dtype=torch.uint8)] \* self.num_gemms'; do
    grep -q "$marker" "$RN4PT_JOB_IMPL/build_runtime.job" || \
      rn4pt_die "build job lost a site-packages backport: $marker" || return
  done
  # v29c retired the enable_pdl=False workaround because FlashInfer 0.6.16.post3
  # fixes flashinfer#3279 (the startup hang it was covering for). On 8 nodes the
  # W4A4 rollout then returned NaN log-probs one step after the first refit --
  # trainer log-probs stayed finite, so this was the rollout, not the trainer --
  # and token-level TIS turned that into a NaN gradient. v29c's site-packages
  # are byte-identical to the v25 image that ran 630 healthy steps except for
  # this one file, so the workaround comes back and the cubin fix stays.
  [[ "$RN4PT_VLLM_DISABLE_TRTLLM_MOE_PDL" = 1 ]] || rn4pt_die "PDL workaround must be applied" || return
  grep -q 'count("enable_pdl=False") == 2' "$RN4PT_JOB_IMPL/build_runtime.job" || \
    rn4pt_die "build job no longer asserts the PDL workaround is present" || return
  grep -q 'disable_vllm_trtllm_nvfp4_moe_pdl.py' "$RN4PT_JOB_IMPL/build_runtime.job" || \
    rn4pt_die "build job no longer applies the PDL workaround" || return
  # FlashInfer must stay at the last release whose TRTLLM_GEN batched-GEMM
  # artifact is good on SM100; see build_runtime.job for the full reason.
  grep -q 'flashinfer-python", marker = "extra == .vllm.", specifier = "==0.6.15.post1"' \
    "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock no longer pins FlashInfer 0.6.15.post1" || return
  grep -q 'batched_gemm-da58956-b4ac80e' "$RN4PT_JOB_IMPL/build_runtime.job" || \
    rn4pt_die "build job no longer pins the audited TRTLLM_GEN batched-GEMM artifact" || return
  grep -q "platform_machine == 'aarch64'" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock lacks aarch64" || return
  grep -q 'vllm-0.26.0-cp38-abi3-manylinux_2_28_aarch64.whl' "$RN4PT_VERL/uv.lock" || \
    rn4pt_die "uv.lock lacks the aarch64 vLLM 0.26 wheel" || return
  grep -q "rev=$RN4PT_TE_COMMIT" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock lacks the audited TE commit" || return
  grep -q "version = \"$RN4PT_TE_VERSION\"" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock has the wrong TE version" || return
  grep -q 'router_replay.mode=R3' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "R3 is not enabled" || return
  grep -q 'enable_rollout_routing_replay=True' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "rollout routing replay missing" || return
  grep -q 'NVTE_NVFP4_ROW_SCALED_ACTIVATION=1' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "row-scaled activation missing" || return
  grep -q 'NVTE_NVFP4_4OVER6=none' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "4-over-6 must stay off" || return
  grep -q 'readonly MAX_NUM_SEQS=128' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "max_num_seqs must be 128" || return
  grep -q 'actor_rollout_ref.rollout.enforce_eager=False' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "CUDA graph must remain enabled" || return
  # The knobs this bundle drives must still default to the pre-existing
  # behaviour, so an unset variable can never silently change another arm.
  grep -q 'readonly FIRST_LAST_BF16=${FIRST_LAST_BF16:-False}' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "first/last carve-out no longer defaults to off" || return
  grep -q 'readonly FILTER_GROUPS=${FILTER_GROUPS:-False}' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "dynamic sampling no longer defaults to off" || return
  grep -q 'readonly GEN_PROMPT_BSZ_MULT=${GEN_PROMPT_BSZ_MULT:-1}' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "gen batch multiplier no longer defaults to 1" || return
  grep -q 'readonly ROLLOUT_IS=${ROLLOUT_IS:-token}' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "token-level TIS is no longer the default" || return
  grep -q 'VERL_MATH_DAPO_STRICT_MINERVA: "1"' "$RN4PT_BUNDLE/runtime_env_w4a4.yaml" || \
    rn4pt_die "W4A4 runtime does not enforce the strict Minerva verifier" || return
  grep -q 'VERL_MATH_DAPO_STRICT_MINERVA: "1"' "$RN4PT_BUNDLE/runtime_env_bf16.yaml" || \
    rn4pt_die "BF16 runtime does not enforce the strict Minerva verifier" || return
  grep -q 'export VERL_MATH_DAPO_STRICT_MINERVA="$STRICT_MINERVA"' \
    "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "driver does not export the verifier contract" || return
  grep -q 'NVFP4_PER_TOKEN_METHOD = "nvfp4_per_token"' "$RN4PT_VERL/verl/utils/real_nvfp4/vllm_runtime.py" || \
    rn4pt_die "native vLLM online method missing" || return
  grep -q 'reload_weights(' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || \
    rn4pt_die "native reload missing" || return
  grep -q 'defer_last_ack=True' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || \
    rn4pt_die "post-finalize ACK missing" || return
  # The BF16 arm is only meaningful if the rollout actually captures the routing
  # the trainer replays; gating that hook on NVFP4 is what broke it before.
  grep -q 'enable_return_routed_experts", False)' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || \
    rn4pt_die "R3 capture hook is still gated on the NVFP4 rollout" || return
  if git -C "$RN4PT_VERL" grep -Eq \
    'three_stability|adv_length_norm_enable|seg_gate_enable|alignment_loss_enable|GP95_|custom[-_]loss' -- \
    verl examples/real_nvfp4/run_qwen3_30b_megatron.sh examples/real_nvfp4/main_dapo_compat.py; then
    rn4pt_die "three-loss implementation leaked into branch" || return
  else
    scan_status=$?
    [[ $scan_status -eq 1 ]] || rn4pt_die "three-loss source scan failed" || return
  fi
  git -C "$RN4PT_VERL" diff --check || return
  git -C "$RN4PT_VERL" diff --cached --check || return
}

rn4pt_require_runtime_image() {
  local built_source
  [[ -s "$RN4PT_BUILD_PASS" ]] || rn4pt_die "runtime build state missing" || return
  built_source=$(sed -n 's/^source_commit=//p' "$RN4PT_BUILD_PASS")
  [[ -n "$built_source" ]] || rn4pt_die "runtime image source commit is missing" || return
  if [[ "$built_source" != "$RN4PT_SOURCE_COMMIT" ]]; then
    git -C "$RN4PT_VERL" cat-file -e "$built_source^{commit}" || \
      rn4pt_die "runtime image source commit is unavailable" || return
    git -C "$RN4PT_VERL" diff --quiet "$built_source" "$RN4PT_SOURCE_COMMIT" -- \
      . ":(exclude,glob)examples/real_nvfp4/jobs/**" || \
      rn4pt_die "runtime payload changed after the image build" || return
    echo "REAL_NVFP4_PERTOKEN_HARNESS_ONLY source_commit=$RN4PT_SOURCE_COMMIT image_source_commit=$built_source"
  fi
  grep -Fxq "lock_sha256=$RN4PT_LOCK_SHA256" "$RN4PT_BUILD_PASS" || \
    rn4pt_die "runtime image was built from a different uv.lock" || return
  [[ -s "$RN4PT_RUNTIME_IMAGE" && -s "$RN4PT_RUNTIME_IMAGE.sha256" ]] || \
    rn4pt_die "runtime image or checksum missing" || return
  sha256sum -c "$RN4PT_RUNTIME_IMAGE.sha256" || rn4pt_die "runtime image checksum mismatch" || return
}

if [[ "${BASH_SOURCE[0]}" = "$0" ]]; then
  rn4pt_validate_static || exit $?
  echo "REAL_NVFP4_PERTOKEN_STATIC_PASS version=$RN4PT_VERSION"
fi

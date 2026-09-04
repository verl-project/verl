#!/usr/bin/env bash
# shellcheck disable=SC2034

readonly RN4PT_VERSION=${RN4PT_VERSION_OVERRIDE:-verl_real_nvfp4_r3_nativeonline_pdl_off_v8base_20260901_v12}
readonly RN4PT_WORKSPACE=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031
readonly RN4PT_VERL=$RN4PT_WORKSPACE/verl_nvfp4_e2e_r3_20260831_v3
readonly RN4PT_ROOT=$RN4PT_VERL/examples/real_nvfp4
readonly RN4PT_BUNDLE=${RN4PT_BUNDLE_OVERRIDE:-$RN4PT_ROOT/jobs/r3_nativeonline_pdl_off_v8base_20260901_v12}
readonly RN4PT_JOB_IMPL=$RN4PT_ROOT/jobs/r3_nativeonline_20260901_v8
readonly RN4PT_STATE=$RN4PT_WORKSPACE/run_state/$RN4PT_VERSION
readonly RN4PT_LOGS=$RN4PT_WORKSPACE/ray_log/$RN4PT_VERSION
readonly RN4PT_CHECKPOINTS=$RN4PT_WORKSPACE/checkpoints/DAPO-NVFP4-QAT/$RN4PT_VERSION

# v8 is the validated vLLM 0.26 / MCore 0.18 / TE 2.18 image: it completed a
# 1-node smoke and an 8-node 20-step run with healthy rollout numerics
# (rollout_corr/kl 0.008, ESS 0.99, entropy 0.87, response length ~1.0k).
# v12 changes exactly one runtime variable relative to that image: vLLM's two
# FlashInfer TRTLLM NVFP4 MoE call sites pass enable_pdl=False.  The v10
# post-0.26 online-NVFP4 backports are deliberately NOT applied here; see
# ../20260901_SUBMISSION_INCIDENTS_CN.md.
readonly RN4PT_BASE_IMAGE=${RN4PT_BASE_IMAGE_OVERRIDE:-/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.nativeonline.20260901.v8.sqsh}
readonly RN4PT_RUNTIME_IMAGE=${RN4PT_RUNTIME_IMAGE_OVERRIDE:-/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.nativeonline.pdl-off.v8base.20260901.v12.sqsh}
readonly RN4PT_IMAGE_ENV=/opt/verl-rn4pt-20260831-v5
readonly RN4PT_IMAGE_PYTHON=$RN4PT_IMAGE_ENV/.venv/bin/python
readonly RN4PT_NETRC=/home/shuazhang/.netrc
readonly RN4PT_MOUNTS=/lustre/fsw/general_sa/shuazhang:/lustre/fsw/general_sa/shuazhang,/home/shuazhang/.netrc:/root/.netrc

readonly RN4PT_ACCOUNT=general_sa
readonly RN4PT_PARTITIONS=36x2-a01r,tcpo,batch

readonly RN4PT_PROJECT=DAPO-NVFP4-QAT
readonly RN4PT_SMOKE_EXP=${RN4PT_SMOKE_EXP_OVERRIDE:-verl_30b_realw4a4_r3_nativeonline_pdl_off_v8base_smoke_20260901_v12}
readonly RN4PT_SHORT_EXP=${RN4PT_SHORT_EXP_OVERRIDE:-verl_30b_realw4a4_r3_nativeonline_pdl_off_v8base_8n_20260901_v12}
readonly RN4PT_VERL_BASE_COMMIT=b356f4301e67c9896c9f1690785e096242c08705
readonly RN4PT_RECIPE_COMMIT=e7f889574b8301cc0f0fc1d57c6d67f31ffeb689
readonly RN4PT_TE_COMMIT=e7c550c5f80636cf841a8204b1d6f85a5f3f28b7
readonly RN4PT_TE_VERSION=2.18.0+e7c550c5
readonly RN4PT_VLLM_WRITABLE_SCALES_COMMIT=b07ec92faa2d534ed97fe97f44ca71c816404849
readonly RN4PT_MCORE_STATELESS_EXTRA_STATE_COMMIT=7e6a045a2a99659ddd2919fc19013a164c508825
# Intentionally empty: the post-0.26 expert-packing (#50029) and reload-kernel
# (#50074) backports are excluded from this image so PDL is the only variable.
readonly RN4PT_VLLM_DISABLE_TRTLLM_MOE_PDL=1
RN4PT_SOURCE_COMMIT=$(git -C "$RN4PT_VERL" rev-parse HEAD)
readonly RN4PT_SOURCE_COMMIT
RN4PT_LOCK_SHA256=$(sha256sum "$RN4PT_VERL/uv.lock" | awk '{print $1}')
readonly RN4PT_LOCK_SHA256
readonly RN4PT_PROBE_PASS=$RN4PT_STATE/probe.pass
readonly RN4PT_BUILD_PASS=$RN4PT_STATE/build.pass
readonly RN4PT_PREFLIGHT_PASS=$RN4PT_STATE/preflight.pass
readonly RN4PT_SMOKE_PASS=$RN4PT_STATE/smoke.pass

rn4pt_die() { echo "REAL_NVFP4_PERTOKEN_REFUSED: $*" >&2; return 2; }

rn4pt_validate_static() {
  local path scan_status
  [[ -d "$RN4PT_VERL/.git" || -f "$RN4PT_VERL/.git" ]] || rn4pt_die "worktree missing" || return
  [[ -z "$(git -C "$RN4PT_VERL" status --porcelain --untracked-files=normal)" ]] || \
    rn4pt_die "worktree must be clean so the runtime image and source commit cannot diverge" || return
  [[ ! -e "$RN4PT_VERL/.venv" && ! -L "$RN4PT_VERL/.venv" ]] || \
    rn4pt_die "worktree .venv would leak a host environment into the runtime image" || return
  git -C "$RN4PT_VERL" merge-base --is-ancestor "$RN4PT_VERL_BASE_COMMIT" HEAD || \
    rn4pt_die "worktree is not based on the audited latest Verl commit" || return
  [[ -s "$RN4PT_BASE_IMAGE" ]] || rn4pt_die "base image missing" || return
  [[ -f "$RN4PT_NETRC" && ! -L "$RN4PT_NETRC" ]] || rn4pt_die "updated W&B netrc missing" || return
  [[ "$(git -C "$RN4PT_VERL/recipe" rev-parse HEAD)" = "$RN4PT_RECIPE_COMMIT" ]] || \
    rn4pt_die "recipe submodule is missing or drifted" || return
  for path in \
    "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" \
    "$RN4PT_ROOT/config/attn_bf16_mlp_nvfp4.yaml" \
    "$RN4PT_ROOT/runtime_backports/disable_vllm_trtllm_nvfp4_moe_pdl.py" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/bf16_transport.py" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/r3_monolithic_capture.py" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/vllm_runtime.py"; do
    [[ -f "$path" && ! -L "$path" && -s "$path" ]] || rn4pt_die "missing input: $path" || return
  done
  bash -n "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" "$RN4PT_BUNDLE"/*.sh "$RN4PT_JOB_IMPL"/*.job || return
  if [ "${RN4PT_SKIP_UV_LOCK_CHECK:-0}" != 1 ]; then
    if [ "${VERL_USE_UV:-1}" = 1 ] && [ "${DEVICE:-gpu}" = gpu ]; then
      uv -q lock --check --directory "$RN4PT_VERL" || rn4pt_die "uv.lock is stale" || return
    else
      rn4pt_die "real NVFP4 bundle requires uv on the GPU branch" || return
    fi
  fi
  grep -q "platform_machine == 'aarch64'" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock lacks aarch64" || return
  grep -q 'vllm-0.26.0-cp38-abi3-manylinux_2_28_aarch64.whl' "$RN4PT_VERL/uv.lock" || \
    rn4pt_die "uv.lock lacks the aarch64 vLLM 0.26 wheel" || return
  [[ -z "${RN4PT_VLLM_EXPERT_PACKING_COMMIT:-}" && -z "${RN4PT_VLLM_RELOAD_KERNEL_COMMIT:-}" ]] || \
    rn4pt_die "v12 must not carry the post-v0.26 online-NVFP4 backports" || return
  if [[ "${RN4PT_VLLM_DISABLE_TRTLLM_MOE_PDL:-0}" = 1 ]]; then
    grep -q 'enable_pdl=False' "$RN4PT_ROOT/runtime_backports/disable_vllm_trtllm_nvfp4_moe_pdl.py" || \
      rn4pt_die "TRTLLM NVFP4 MoE PDL-off patch missing" || return
    grep -q 'REAL_NVFP4_VLLM_TRTLLM_MOE_PDL_OFF_PASS' "$RN4PT_JOB_IMPL/build_runtime.job" || \
      rn4pt_die "TRTLLM NVFP4 MoE PDL-off image gate missing" || return
  fi
  grep -q "rev=$RN4PT_TE_COMMIT" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock lacks the audited TE commit" || return
  grep -q "version = \"$RN4PT_TE_VERSION\"" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock has the wrong TE version" || return
  grep -q 'router_replay.mode=R3' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "R3 is not enabled" || return
  grep -q 'enable_rollout_routing_replay=True' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "rollout routing replay missing" || return
  grep -q 'NVTE_NVFP4_ROW_SCALED_ACTIVATION=1' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "row-scaled activation missing" || return
  grep -q 'NVTE_NVFP4_4OVER6=none' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "4-over-6 must stay off" || return
  grep -q 'readonly MAX_NUM_SEQS=128' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "max_num_seqs must be 128" || return
  grep -q 'readonly FIRST_LAST_BF16=${FIRST_LAST_BF16:-False}' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "all-MLP first/last carve-out is not disabled" || return
  grep -q 'actor_rollout_ref.rollout.enforce_eager=False' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    rn4pt_die "CUDA graph must remain enabled" || return
  grep -q 'NVFP4_PER_TOKEN_METHOD = "nvfp4_per_token"' "$RN4PT_VERL/verl/utils/real_nvfp4/vllm_runtime.py" || \
    rn4pt_die "native vLLM online method missing" || return
  grep -q 'quantization = NVFP4_PER_TOKEN_METHOD' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/vllm_async_server.py" || \
    rn4pt_die "native vLLM online method is not selected" || return
  grep -q 'reload_weights(' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || rn4pt_die "native reload missing" || return
  grep -q 'defer_last_ack=True' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || \
    rn4pt_die "post-finalize ACK missing" || return
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

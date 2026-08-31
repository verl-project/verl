#!/usr/bin/env bash
# shellcheck disable=SC2034

readonly RN4PT_VERSION=verl_real_nvfp4_r3_nativeonline_20260831_v3
readonly RN4PT_WORKSPACE=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031
readonly RN4PT_VERL=$RN4PT_WORKSPACE/verl_nvfp4_e2e_r3_20260831_v3
readonly RN4PT_ROOT=$RN4PT_VERL/examples/real_nvfp4
readonly RN4PT_BUNDLE=$RN4PT_ROOT/jobs/r3_nativeonline_20260831_v3
readonly RN4PT_STATE=$RN4PT_WORKSPACE/run_state/$RN4PT_VERSION
readonly RN4PT_LOGS=$RN4PT_WORKSPACE/ray_log/$RN4PT_VERSION
readonly RN4PT_CHECKPOINTS=$RN4PT_WORKSPACE/checkpoints/DAPO-NVFP4-QAT/$RN4PT_VERSION

# Known-working aarch64 CUDA/TE/MCore base from the previous experiment. The
# versioned build job upgrades vLLM and the current Verl dependencies in a new
# image; the old image is never modified.
readonly RN4PT_BASE_IMAGE=/lustre/fsw/general_sa/shuazhang/images/verl.vllm0202.mcore0161.te215.realnvfp4.cgruntime.20260829.v14.sqsh
readonly RN4PT_RUNTIME_IMAGE=/lustre/fsw/general_sa/shuazhang/images/verl.vllm026.mcore018.te218e7.realnvfp4.nativeonline.20260831.v3.sqsh
readonly RN4PT_IMAGE_ENV=/opt/verl-rn4pt-20260831-v3
readonly RN4PT_IMAGE_PYTHON=$RN4PT_IMAGE_ENV/.venv/bin/python
readonly RN4PT_NETRC=/home/shuazhang/.netrc
readonly RN4PT_MOUNTS=/lustre/fsw/general_sa/shuazhang:/lustre/fsw/general_sa/shuazhang,/home/shuazhang/.netrc:/root/.netrc

readonly RN4PT_ACCOUNT=general_sa
readonly RN4PT_PARTITIONS=36x2-a01r,tcpo,batch

readonly RN4PT_PROJECT=DAPO-NVFP4-QAT
readonly RN4PT_SMOKE_EXP=verl_30b_realw4a4_r3_nativeonline_smoke_20260831_v3
readonly RN4PT_SHORT_EXP=verl_30b_realw4a4_r3_nativeonline_8n_short_20260831_v3
readonly RN4PT_VERL_BASE_COMMIT=b356f4301e67c9896c9f1690785e096242c08705
readonly RN4PT_RECIPE_COMMIT=e7f889574b8301cc0f0fc1d57c6d67f31ffeb689
readonly RN4PT_TE_COMMIT=e7c550c5f80636cf841a8204b1d6f85a5f3f28b7
readonly RN4PT_TE_VERSION=2.18.0+e7c550c5
readonly RN4PT_PROBE_PASS=$RN4PT_STATE/probe.pass
readonly RN4PT_BUILD_PASS=$RN4PT_STATE/build.pass
readonly RN4PT_PREFLIGHT_PASS=$RN4PT_STATE/preflight.pass
readonly RN4PT_SMOKE_PASS=$RN4PT_STATE/smoke.pass

rn4pt_die() { echo "REAL_NVFP4_PERTOKEN_REFUSED: $*" >&2; return 2; }

rn4pt_validate_static() {
  local path
  [[ -d "$RN4PT_VERL/.git" || -f "$RN4PT_VERL/.git" ]] || rn4pt_die "worktree missing" || return
  [[ ! -e "$RN4PT_VERL/.venv" && ! -L "$RN4PT_VERL/.venv" ]] || \
    rn4pt_die "worktree .venv would leak a host environment into the runtime image" || return
  git -C "$RN4PT_VERL" merge-base --is-ancestor "$RN4PT_VERL_BASE_COMMIT" HEAD || \
    rn4pt_die "worktree is not based on the audited latest Verl commit" || return
  [[ -s "$RN4PT_BASE_IMAGE" ]] || rn4pt_die "base image missing" || return
  [[ -f "$RN4PT_NETRC" && ! -L "$RN4PT_NETRC" ]] || rn4pt_die "updated W&B netrc missing" || return
  [[ "$(git -C "$RN4PT_VERL/recipe" rev-parse HEAD)" = "$RN4PT_RECIPE_COMMIT" ]] || rn4pt_die "recipe submodule is missing or drifted" || return
  for path in \
    "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" \
    "$RN4PT_ROOT/config/attn_bf16_mlp_nvfp4.yaml" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/bf16_transport.py" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/r3_monolithic_capture.py" \
    "$RN4PT_VERL/verl/utils/real_nvfp4/vllm_runtime.py"; do
    [[ -f "$path" && ! -L "$path" && -s "$path" ]] || rn4pt_die "missing input: $path" || return
  done
  bash -n "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" "$RN4PT_BUNDLE"/*.sh "$RN4PT_BUNDLE"/*.job || return
  if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    uv -q lock --check --directory "$RN4PT_VERL" || rn4pt_die "uv.lock is stale" || return
  else
    rn4pt_die "real NVFP4 bundle requires uv on the GPU branch" || return
  fi
  grep -q "platform_machine == 'aarch64'" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock lacks aarch64" || return
  grep -q 'vllm-0.26.0-cp38-abi3-manylinux_2_28_aarch64.whl' "$RN4PT_VERL/uv.lock" || \
    rn4pt_die "uv.lock lacks the aarch64 vLLM 0.26 wheel" || return
  grep -q "rev=$RN4PT_TE_COMMIT" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock lacks the audited TE commit" || return
  grep -q "version = \"$RN4PT_TE_VERSION\"" "$RN4PT_VERL/uv.lock" || rn4pt_die "uv.lock has the wrong TE version" || return
  grep -q 'router_replay.mode=R3' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "R3 is not enabled" || return
  grep -q 'enable_rollout_routing_replay=True' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "rollout routing replay missing" || return
  grep -q 'NVTE_NVFP4_ROW_SCALED_ACTIVATION=1' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "row-scaled activation missing" || return
  grep -q 'NVTE_NVFP4_DISABLE_RHT=1' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "RHT contract mismatch" || return
  grep -q 'NVTE_NVFP4_4OVER6=none' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "4-over-6 must stay off for vLLM alignment" || return
  grep -q 'NVTE_NVFP4_4OVER6: "none"' "$RN4PT_ROOT/runtime_env.yaml" || rn4pt_die "Ray workers lack the 4-over-6 scope contract" || return
  grep -q 'NVTE_NVFP4_4OVER6_E4M3_USE_256: "all"' "$RN4PT_ROOT/runtime_env.yaml" || rn4pt_die "Ray workers lack the 4-over-6 E4M3 contract" || return
  grep -q 'NVTE_NVFP4_4OVER6_ERR_MODE: "MAE"' "$RN4PT_ROOT/runtime_env.yaml" || rn4pt_die "Ray workers lack the 4-over-6 error contract" || return
  grep -q 'FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "FlashInfer exact FP4 math contract missing" || return
  grep -q 'TRTLLM_DISABLE_FP4_QUANT_FAST_MATH=1' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "TRT-LLM exact FP4 math contract missing" || return
  grep -q 'readonly MAX_NUM_SEQS=128' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "formal max_num_seqs must be 128" || return
  grep -q 'override_transformer_config.first_last_layers_bf16=False' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || rn4pt_die "all-MLP first/last carve-out is not disabled" || return
  grep -q 'NVFP4_PER_TOKEN_METHOD = "nvfp4_per_token"' "$RN4PT_VERL/verl/utils/real_nvfp4/vllm_runtime.py" || rn4pt_die "native vLLM online method missing" || return
  grep -q 'quantization = NVFP4_PER_TOKEN_METHOD' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/vllm_async_server.py" || rn4pt_die "native vLLM online method is not selected" || return
  grep -q 'engine_kwargs\["moe_backend"\] = REAL_NVFP4_MOE_BACKEND' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/vllm_async_server.py" || rn4pt_die "FlashInfer TRT-LLM backend is not selected explicitly" || return
  grep -q 'Nvfp4OnlineMoEMethod' "$RN4PT_VERL/verl/utils/real_nvfp4/vllm_runtime.py" || rn4pt_die "native vLLM online runtime attestation missing" || return
  grep -q 'VERL_R3_MONOLITHIC_CAPTURE_PATCH PASS' "$RN4PT_VERL/verl/utils/real_nvfp4/r3_monolithic_capture.py" || rn4pt_die "monolithic R3 capture patch missing" || return
  grep -q 'patch_vllm_monolithic_moe_r3_capture()' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || rn4pt_die "R3 patch is not installed in model workers" || return
  grep -q 'VERL_R3_ROLLOUT_ROUTES PASS' "$RN4PT_VERL/verl/utils/real_nvfp4/r3_monolithic_capture.py" || rn4pt_die "R3 route runtime attestation missing" || return
  grep -q 'reload_weights(' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || rn4pt_die "native reload missing" || return
  grep -q 'location="vllm_receive"' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || rn4pt_die "receiver-side BF16 attestation missing" || return
  grep -q 'defer_last_ack=True' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || rn4pt_die "post-finalize ACK missing" || return
  grep -q 'require_vllm_native_reload_contract' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/utils.py" || rn4pt_die "native reload API gate missing" || return
  grep -q 'VERL_REAL_NVFP4_ENGINE_CONTRACT PASS' "$RN4PT_VERL/verl/workers/rollout/vllm_rollout/vllm_async_server.py" || rn4pt_die "engine runtime contract marker missing" || return
  ! rg -q 'three_stability|adv_length_norm_enable|seg_gate_enable|alignment_loss_enable|GP95_|custom[-_]loss' \
    "$RN4PT_VERL/verl" \
    "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" \
    "$RN4PT_ROOT/main_dapo_compat.py" || rn4pt_die "three-loss implementation leaked into branch" || return
  git -C "$RN4PT_VERL" diff --check || return
  git -C "$RN4PT_VERL" diff --cached --check || return
}

rn4pt_require_runtime_image() {
  [[ -s "$RN4PT_RUNTIME_IMAGE" ]] || rn4pt_die "runtime image missing: $RN4PT_RUNTIME_IMAGE" || return
  [[ -s "$RN4PT_RUNTIME_IMAGE.sha256" ]] || rn4pt_die "runtime image checksum missing" || return
  sha256sum -c "$RN4PT_RUNTIME_IMAGE.sha256" || rn4pt_die "runtime image checksum mismatch" || return
}

if [[ "${BASH_SOURCE[0]}" = "$0" ]]; then
  rn4pt_validate_static || exit $?
  echo "REAL_NVFP4_PERTOKEN_STATIC_PASS version=$RN4PT_VERSION"
fi

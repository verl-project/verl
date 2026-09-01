#!/usr/bin/env bash
# shellcheck disable=SC2034,SC1091

# This is an orchestration-only long-run bundle. The executable training
# payload and immutable image are exactly the v8 artifacts that passed the
# 1-node smoke and 8-node 20-step run.
source /lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/verl_nvfp4_e2e_r3_20260831_v3/examples/real_nvfp4/jobs/r3_nativeonline_20260901_v8/manifest.sh

readonly LONG_VERSION=verl_real_nvfp4_r3_nativeonline_long_20260901_v9
readonly LONG_BUNDLE=$RN4PT_ROOT/jobs/r3_nativeonline_long_20260901_v9
readonly LONG_STATE=$RN4PT_WORKSPACE/run_state/$LONG_VERSION
readonly LONG_LOGS=$RN4PT_WORKSPACE/ray_log/$LONG_VERSION
readonly LONG_CHECKPOINTS=$RN4PT_WORKSPACE/checkpoints/$RN4PT_PROJECT/$LONG_VERSION
readonly LONG_EXP=verl_30b_realw4a4_r3_nativeonline_8n_long_20260901_v9
readonly LONG_CHAIN_STATE=$LONG_STATE/chain.tsv
readonly LONG_VALIDATED_IMPLEMENTATION_COMMIT=18bed46024fcd34bfd1252b70078a05085fef087
readonly LONG_VALIDATED_SHORT_JOB=2694027
readonly LONG_VALIDATED_SHORT_EXP=${RN4PT_SHORT_EXP}_j${LONG_VALIDATED_SHORT_JOB}
readonly LONG_VALIDATED_SHORT_OUT=$RN4PT_LOGS/verl-rn4pt-train_${LONG_VALIDATED_SHORT_JOB}.out
readonly LONG_VALIDATED_SHORT_CHECKPOINT=$RN4PT_CHECKPOINTS/$LONG_VALIDATED_SHORT_EXP/global_step_20/actor

long_die() { echo "REAL_NVFP4_LONG_REFUSED: $*" >&2; return 2; }

long_validate_static() {
  rn4pt_validate_static || return
  git -C "$RN4PT_VERL" merge-base --is-ancestor "$LONG_VALIDATED_IMPLEMENTATION_COMMIT" HEAD || \
    long_die "validated v8 implementation commit is not an ancestor" || return
  bash -n "$LONG_BUNDLE"/*.sh "$LONG_BUNDLE"/*.job || return
  [[ -s "$RN4PT_PREFLIGHT_PASS" ]] || long_die "v8 preflight state is missing" || return
  [[ -s "$RN4PT_SMOKE_PASS" ]] || long_die "v8 smoke state is missing" || return
  [[ -s "$LONG_VALIDATED_SHORT_OUT" ]] || long_die "validated 8-node short log is missing" || return
  grep -q 'REAL_NVFP4_PERTOKEN_TRAIN_PASS arm=short steps=20' "$LONG_VALIDATED_SHORT_OUT" || \
    long_die "validated 8-node short PASS marker is missing" || return
  grep -q 'REAL_NVFP4_FULL_ADAM_CHECKPOINT_PASS' "$LONG_VALIDATED_SHORT_OUT" || \
    long_die "validated 8-node short full-Adam marker is missing" || return
  [[ -s "$LONG_VALIDATED_SHORT_CHECKPOINT/ckpt_contents.json" ]] || \
    long_die "validated 8-node short checkpoint is missing" || return
  grep -q 'losses=0of3' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    long_die "0/3-loss contract marker is missing" || return
  grep -q 'router_replay.mode=R3' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    long_die "R3 must remain enabled" || return
  grep -q 'readonly MAX_NUM_SEQS=128' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    long_die "max_num_seqs must remain 128" || return
  grep -q 'actor_rollout_ref.rollout.enforce_eager=False' "$RN4PT_ROOT/run_qwen3_30b_megatron.sh" || \
    long_die "CUDA graph must remain enabled" || return
}

if [[ "${BASH_SOURCE[0]}" = "$0" ]]; then
  long_validate_static || exit $?
  rn4pt_require_runtime_image || exit $?
  echo "REAL_NVFP4_LONG_STATIC_PASS version=$LONG_VERSION exp=$LONG_EXP targets=80,160,240,320"
fi

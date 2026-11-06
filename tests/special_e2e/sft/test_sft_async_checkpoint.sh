#!/usr/bin/env bash
set -euo pipefail

test_root=$(mktemp -d "${TMPDIR:-/tmp}/verl-sft-async.XXXXXX")
trap 'rm -rf "$test_root"' EXIT

for mode in spmd ray; do
    ray stop --force
    case_root="${test_root}/${mode}"
    checkpoint_root="${case_root}/checkpoints"
    mkdir -p "$case_root"

    common_env=(
        BACKEND=megatron
        NUM_GPUS=2
        TP_SIZE=2
        PP_SIZE=1
        VPP_SIZE=null
        CP_SIZE=1
        TRAIN_BATCH_SIZE=8
        TEST_FREQ=-1
        VAL_FILES=null
        ASYNC_SAVE=True
        KEEP_CHECKPOINTS=True
        "mode=${mode}"
        "ckpts_home=${checkpoint_root}"
    )

    env "${common_env[@]}" RESUME_MODE=disable TOTAL_TRAIN_STEP=2 \
        bash tests/special_e2e/sft/run_sft_engine.sh 2>&1 | tee "${case_root}/save.log"

    test "$(cat "${checkpoint_root}/latest_checkpointed_iteration.txt")" = 2
    test -s "${checkpoint_root}/global_step_2/ckpt_contents.json"
    test -s "${checkpoint_root}/global_step_2/data_0.pt"

    ray stop --force
    env "${common_env[@]}" RESUME_MODE=auto TOTAL_TRAIN_STEP=3 \
        bash tests/special_e2e/sft/run_sft_engine.sh 2>&1 | tee "${case_root}/resume.log"

    grep -Fq "Found latest checkpoint: ${checkpoint_root}/global_step_2 (step 2)" \
        "${case_root}/resume.log"
    test "$(cat "${checkpoint_root}/latest_checkpointed_iteration.txt")" = 3
    test -s "${checkpoint_root}/global_step_3/ckpt_contents.json"
done

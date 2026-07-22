#!/usr/bin/env bash
# SFT smoke that exercises verl's Megatron-Core Muon wiring end-to-end.
#
# Runs verl.trainer.sft_trainer with the classic Megatron engine (DistributedOptimizer
# path, i.e. get_megatron_optimizer -> TensorParallelMuon via emerging_optimizers) so a
# real training loop actually constructs and steps the Muon optimizer instead of a stub.
#
# It is deliberately parametrized by ${OPTIMIZER} so the exact same config can be run as a
# Muon arm and an Adam(W) baseline arm for a fair loss comparison:
#
#   OPTIMIZER=muon  bash examples/sft/gsm8k/run_muon_megatron_sft.sh   # verl+Megatron Muon
#   OPTIMIZER=adam  bash examples/sft/gsm8k/run_muon_megatron_sft.sh   # baseline (Megatron Adam == AdamW)
#
# Everything else (model, data, batch size, lr schedule, steps) is identical between arms.
# Loss is logged to console AND to a per-step JSONL file (trainer.logger includes "file"),
# so the loss curve can be extracted without wandb. The JSONL lands at:
#   ${SAVE_PATH}/gsm8k-muon-sft/${EXP_NAME}.jsonl   (one {"step":N,"data":{...}} per line)
# Extract the curve with, e.g.:
#   python -c 'import json,sys;[print(json.loads(l)["step"], json.loads(l)["data"].get("loss")) for l in open(sys.argv[1])]' \
#     ${SAVE_PATH}/gsm8k-muon-sft/${EXP_NAME}.jsonl
#
# Requires a GPU + a Megatron-Core build that ships emerging_optimizers (Muon). On CW this
# is the verl.complete.sqsh container used by sibling Megatron-Muon jobs. If Muon is
# requested against a Megatron build with no Muon fields, verl raises instead of silently
# falling back to Adam (that is the whole point of this smoke).

set -xeuo pipefail

########################### Quick Config ###########################

OPTIMIZER=${OPTIMIZER:-muon}                       # muon | adaptive_muon | adam
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}   # small so the loss-drop smoke is cheap
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k_sft}         # produced by gsm8k_multiturn_sft.py (messages-keyed)
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}

NPROC=${NPROC:-1}                                  # <= 8 GPUs; 0.5B fits on 1
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}

MAX_STEPS=${MAX_STEPS:-40}                          # a few dozen steps is enough to show loss drop
LR=${LR:-1e-4}
MINLR=${MINLR:-1e-5}

SAVE_ROOT=${SAVE_ROOT:-$HOME/verl/muon_sft}
EXP_NAME=${EXP_NAME:-gsm8k-sft-megatron-${OPTIMIZER}}
SAVE_PATH=${SAVE_PATH:-${SAVE_ROOT}/${EXP_NAME}}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VERL_SFT_LOGGING_LEVEL=INFO
export VERL_FILE_LOGGER_ROOT=${SAVE_PATH}   # FileLogger writes ${SAVE_PATH}/<project>/<exp>.jsonl
unset ROCR_VISIBLE_DEVICES || true

mkdir -p "${SAVE_PATH}"

########################### Parameter Arrays ###########################

DATA=(
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    data.messages_key=messages
    data.train_batch_size=32
    data.micro_batch_size_per_gpu=2
    data.use_dynamic_bsz=True
    data.max_token_len_per_gpu=2048
    data.pad_mode=no_padding
    data.truncation=error
    data.num_workers=0
)

MODEL=(
    model=hf_model
    "model.path=${MODEL_PATH}"
    model.trust_remote_code=True
    model.use_remove_padding=true
)

# Optimizer arm. optim=megatron routes to McoreOptimizerConfig; optim.optimizer selects the
# Megatron algorithm. "muon"/"adaptive_muon" trigger verl's Muon passthrough; "adam" is the
# classic Megatron optimizer (AdamW-equivalent) used as the baseline arm.
OPTIM=(
    optim=megatron
    "optim.optimizer=${OPTIMIZER}"
    "optim.lr=${LR}"
    "optim.min_lr=${MINLR}"
    optim.lr_warmup_steps=5
    optim.weight_decay=0.1
    "optim.betas=[0.9,0.95]"
    optim.clip_grad=1.0
    optim.lr_warmup_init=0
    optim.lr_decay_style=cosine
)

# Muon-only knobs (ignored by Megatron when optimizer=adam). Defaults already mirror
# Megatron-Core; expose the common ones so A/B sweeps can override them.
if [[ "${OPTIMIZER}" == "muon" || "${OPTIMIZER}" == "adaptive_muon" ]]; then
    OPTIM+=(
        "optim.muon_tp_mode=${MUON_TP_MODE:-blockwise}"
        "optim.muon_scalar_optimizer=${MUON_SCALAR_OPT:-adam}"
        "optim.muon_num_ns_steps=${MUON_NS_STEPS:-5}"
    )
fi

ENGINE=(
    engine=megatron
    engine.tensor_model_parallel_size=${TP}
    engine.pipeline_model_parallel_size=${PP}
    engine.context_parallel_size=${CP}
    engine.use_mbridge=True
)

TRAINER=(
    "trainer.default_local_dir=${SAVE_PATH}"
    trainer.project_name=gsm8k-muon-sft
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.logger=[console,file]"
    trainer.total_epochs=1
    "trainer.total_training_steps=${MAX_STEPS}"
    trainer.test_freq=-1
    trainer.save_freq=-1
)

########################### Launch ###########################

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    -m verl.trainer.sft_trainer \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${OPTIM[@]}" \
    "${ENGINE[@]}" \
    "${TRAINER[@]}" \
    "$@"

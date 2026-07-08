set -x

NUM_GPUS=${NUM_GPUS:-4}

mode=${mode:-spmd}

if [ "$mode" = "spmd" ]; then
  ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}
  COMMAND="torchrun --standalone --nnodes=${NNODES:-1} --nproc-per-node=${NUM_GPUS:-1} ${ENTRYPOINT}"
else
  ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer_ray"}
  COMMAND="python ${ENTRYPOINT} trainer.nnodes=${NNODES:-1} trainer.n_gpus_per_node=${NUM_GPUS:-1}"
fi

RESUME_MODE=disable

ckpts_home=${ckpts_home:-~/verl/test/gsm8k-sft-fsdp}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}

DATASET_DIR=${DATASET_DIR:-$HOME/data/gsm8k_sft}
TRAIN_FILES=${DATASET_DIR}/train.parquet
VAL_FILES=${DATASET_DIR}/test.parquet

exp_name=gsm8k-sft-qwen-2.5-0.5b-instruct-mode-${mode}

mkdir -p "${ckpts_home}"

$COMMAND \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.messages_key=messages \
    model.path=$MODEL_PATH \
    model.use_remove_padding=True \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    engine=fsdp \
    optim=fsdp \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.lr_scheduler_type=cosine \
    engine.ulysses_sequence_parallel_size=2 \
    engine.strategy=fsdp2 \
    engine.fsdp_size=2 \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger=['console','file'] \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.total_epochs=2 \
    trainer.total_training_steps=2 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \

rm -rf "${ckpts_home:?}/*"

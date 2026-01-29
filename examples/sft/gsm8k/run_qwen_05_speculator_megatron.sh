export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE=0
export PYTORCH_JIT=0


torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  -m verl.trainer.sft_trainer \
  data.train_files=$HOME/data/multiturn/train.parquet \
  data.val_files=$HOME/data/multiturn/test.parquet \
  ++data.pad_mode=no_padding \
  data.micro_batch_size_per_gpu=1 \
  data.ignore_input_ids_mismatch=true \
  model.path=/root/autodl-tmp/qwen3_moe_small  \
  model.use_remove_padding=true \
  model.lora.rank=32 \
  model.lora.alpha=16 \
  model.lora.target_modules=all-linear \
  engine=megatron \
  optim=megatron \
  engine.tensor_model_parallel_size=1 \
  engine.pipeline_model_parallel_size=1 \
  engine.virtual_pipeline_model_parallel_size=null \
  engine.context_parallel_size=1 \
  engine.use_mbridge=true \
  trainer.total_training_steps=2 \
  trainer.total_epochs=1 \
  trainer.logger=console \
  trainer.default_local_dir=/model/ljl/output/lstm

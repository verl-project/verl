#!/usr/bin/env bash
set -xeuo pipefail

ulimit -u 65535
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="9.0"
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p "$RAY_TMPDIR"

rm -f run*
rm -rf outputs
rm -rf checkpoints
rm -rf wandb

ray stop -f || true

LOG="run_gkd_only_fsdp2_qwen2p5_14b_to_0p5b_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="run.pid"

# ====== Sequence length knobs ======
max_prompt_length=512
max_response_length=512
max_total_length=$((max_prompt_length + max_response_length))

# vLLM settings
vllm_max_batched_tokens=$((max_total_length))
train_max_token_len_per_gpu=$((max_total_length))

# ====== Teacher server settings ======
# Make sure teacher server is running before starting training!
# cd recipe/gkd/teacher && bash start_server.sh
teacher_ip="127.0.0.1"
teacher_port=15555
teacher_workers=1

args=(
  # ====== GKD Configuration ======
  +actor_rollout_ref.actor.use_teacher_kl_loss=True
  +actor_rollout_ref.actor.gkd_only_mode=True
  +actor_rollout_ref.actor.teacher_kl_coef=1.0
  +actor_rollout_ref.actor.teacher_kl_temperature=1.0

  # Teacher server
  +actor_rollout_ref.teacher.server_ip=${teacher_ip}
  +actor_rollout_ref.teacher.server_port=${teacher_port}
  +actor_rollout_ref.teacher.n_server_workers=${teacher_workers}
  +actor_rollout_ref.teacher.temperature=1.0

  # ====== Data ======
  data.train_files=/workspace/mlf2/verl/reproduce/data/gsm8k/local_parquet_dir/train.parquet
  data.val_files=/workspace/mlf2/verl/reproduce/data/gsm8k/local_parquet_dir/test.parquet
  data.train_batch_size=6
  data.max_prompt_length=${max_prompt_length}
  data.max_response_length=${max_response_length}
  data.filter_overlong_prompts=True
  data.truncation=left
  data.train_max_samples=4800
  data.val_max_samples=200
  data.val_batch_size=12

  # ====== Model ======
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
  actor_rollout_ref.model.use_remove_padding=False  # GKD needs logits, disable remove_padding
  ++actor_rollout_ref.model.enable_gradient_checkpointing=true

  # ====== FSDP2 Backend ======
  actor_rollout_ref.actor.strategy=fsdp2

  # FSDP2 memory optimization
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

  # Dynamic batch size for variable sequence lengths
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1

  # PPO batch sizes (still used for actor update in GKD)
  actor_rollout_ref.actor.ppo_mini_batch_size=8
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${train_max_token_len_per_gpu}

  # ====== Optimizer ======
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.optim.lr_warmup_steps=20

  # ====== Rollout (vLLM) ======
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_num_batched_tokens=${vllm_max_batched_tokens}
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75
  actor_rollout_ref.rollout.n=1  # GKD only: single response per prompt
  actor_rollout_ref.rollout.temperature=0.6
  actor_rollout_ref.rollout.top_p=0.95
  # ++actor_rollout_ref.rollout.stop_token_ids='[151645,151643]' # this is for qwen3, turn it off for qwen2
  ++actor_rollout_ref.rollout.stop_token_ids='[151643]'
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct

  # ====== Trainer ======
  trainer.critic_warmup=0
  trainer.logger='["console","wandb"]'
  trainer.project_name=mw_verl_recipe_reasoning
  trainer.experiment_name=gkd_only_qwen2p5_14b_to_0p5b_gsm8k_fsdp2
  trainer.n_gpus_per_node=3
  trainer.nnodes=1
  trainer.save_freq=200
  trainer.test_freq=200
  trainer.total_epochs=1
  trainer.val_before_train=False
  +trainer.gkd_lambda=1.0
  +trainer.enable_off_policy=False
)

nohup \
env \
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=4,5,6 \
python3 -u -m verl.trainer.main_ppo "${args[@]}" \
> "$LOG" 2>&1 < /dev/null &

echo $! > "$PIDFILE"
echo "Started GKD training. PID=$(cat $PIDFILE)"
echo "View logs: tail -f $LOG"
echo ""
echo "NOTE: Make sure teacher server is running!"
echo "      cd /workspace/mlf2/verl/recipe/gkd/teacher && bash start_server.sh"


# Before you run this, start the teacher server
# bash /workspace/mlf2/verl/recipe/gkd/teacher/start_server.sh


# To stop the background run
# kill $(cat run.pid) || true
# sleep 5
# kill -9 $(cat run.pid) || true
# ray stop -f || true
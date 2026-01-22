ulimit -u 65535
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="9.0"
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p $RAY_TMPDIR
ray stop -f || true
rm run*
rm -r outputs

args=(
  algorithm.adv_estimator=grpo
  data.train_files=/workspace/mlf2/verl/reproduce/data/gsm8k/train.parquet
  data.val_files=/workspace/mlf2/verl/reproduce/data/gsm8k/test.parquet
  data.train_batch_size=1024
  data.max_prompt_length=512
  data.max_response_length=1024
  data.filter_overlong_prompts=True
  data.truncation=error

  actor_rollout_ref.model.path=Qwen/Qwen3-4B # Qwen/Qwen2-0.5B-Instruct
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size=256
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 # 40

  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.model.enable_gradient_checkpointing=False

  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 # 40
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 # 2
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.n=1 # 5

  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 # 40
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.strategy=fsdp2

  algorithm.use_kl_in_reward=False

  trainer.critic_warmup=0
  trainer.logger='["console","wandb"]'
  trainer.project_name=mw_verl_reproduce
  trainer.experiment_name=gsm8k-grpo-qwen3_4b
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.save_freq=20
  trainer.test_freq=5
  trainer.total_epochs=1
)

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1,2,5,6 \
# python3 -m verl.trainer.main_ppo "${args[@]}" \
# 2>&1 | tee "run_grpo_gsm8k_$(date +%Y%m%d_%H%M%S).log"


LOG="run_grpo_gsm8k_qwen3_4b_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="run.pid"

nohup \
env \
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=1,2,5,6 \
  python3 -u -m verl.trainer.main_ppo "${args[@]}" \
  > "$LOG" 2>&1 < /dev/null &


PYTHON_PID=$!
echo $PYTHON_PID > "$PIDFILE"
echo "Started training. PID=$PYTHON_PID"
echo "View logs: tail -f $LOG"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="9.0"
ray stop -f || true
rm run*
rm -r outputs

args=(
  --config-path=/workspace/mlf2/verl/recipe/gkd/config
  --config-name=on_policy_distill_trainer

  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync # default is sync and not in _ROLLOUT_REGISTRY in verl/verl/workers/rollout/base.py
  ++actor_rollout_ref.rollout.n=1
  ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  ++actor_rollout_ref.actor.ppo_mini_batch_size=2
  ++actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.teacher.server_ip=127.0.0.1
  actor_rollout_ref.teacher.server_port=15555

  data.train_files=local_parquet_dir/train.parquet
  data.val_files=local_parquet_dir/test.parquet
  data.off_policy_files=local_parquet_dir/train.parquet
  ++data.max_prompt_length=512
  ++data.max_response_length=512
  ++data.train_batch_size=4
  ++data.dataloader_num_workers=1 # keep this value small, otherwise sometimes it will make terminal unresponsive
  # ++data.val_batch_size=16

  trainer.total_epochs=1
  trainer.n_gpus_per_node=1
  trainer.scheduler=one_step_off
  trainer.test_freq=5
  trainer.logger='["console","wandb"]'
  trainer.project_name=mw_verl_reproduce
  trainer.experiment_name=gsm8k-gkd-qwen2p5_1p5b_to_0p5b_lambda0p1_test
  trainer.total_training_steps=null # null
  trainer.gkd_lambda=0.1
  trainer.enable_off_policy=True


  rollout.n_gpus_per_node=1
  # ++rollout.gpu_memory_utilization=0.7

)

# Disable MoE related
args+=(
  ++actor_rollout_ref.actor.router_replay.mode=disabled
  ++actor_rollout_ref.model.override_config.moe_config.freeze_moe_router=true
)

LOG="run_gkd_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="run.pid"

# RAY_DEBUG=legacy \
# HYDRA_FULL_ERROR=1 \
# CUDA_VISIBLE_DEVICES=1,2,4,5,6 \
# python3 -u -m recipe.gkd.main_gkd "${args[@]}" 2>&1 | tee "$LOG" &

nohup \
env \
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=1,2 \
  python3 -u -m recipe.gkd.main_gkd "${args[@]}" \
  > "$LOG" 2>&1 < /dev/null &


PYTHON_PID=$!
echo $PYTHON_PID > "$PIDFILE"
echo "Started training. PID=$PYTHON_PID"
echo "View logs: tail -f $LOG"


# Before you run this, start the teacher server
# cd /workspace/mlf2/verl/recipe/gkd/teacher/ && bash start_server.sh



# To stop the background run
# kill $(cat run.pid) || true
# sleep 5
# kill -9 $(cat run.pid) || true
# ray stop -f || true
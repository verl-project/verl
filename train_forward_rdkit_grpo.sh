
set -e

# 1. 设置Ray和其他环境变量
ulimit -n 65535
export RAY_DEDUP_LOGS=0

# 多节点 Ray：根据 NODE_RANK / HEAD_ADDRESS 在 head 起 Ray 并跑训练，在 worker 起 Ray 并挂到 head
# 环境变量：NODE_RANK(0=head,>=1=worker), HEAD_ADDRESS 或 RAY_HEAD_ADDRESS(worker 必填, ip:port),
#          MASTER_ADDR(集群提供时自动用作 Ray head 地址), SOCKET_IP(集群提供的本机 IP),
#          RAY_PORT(默认6379), RAY_MIN_GPUS(可选), RAY_WAIT_TIMEOUT_SEC(默认120),
#          HEAD_IP/POD_IP(可选), NUM_GPUS_PER_NODE/NUM_CPUS_PER_NODE(默认8/96)
RAY_PORT=${RAY_PORT:-6379}
NODE_RANK=${NODE_RANK:-0}
HEAD_ADDRESS=${HEAD_ADDRESS:-${RAY_HEAD_ADDRESS}}
[[ -z "$HEAD_ADDRESS" && -n "$MASTER_ADDR" ]] && HEAD_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
NUM_CPUS_PER_NODE=${NUM_CPUS_PER_NODE:-96}
RAY_MIN_GPUS=${RAY_MIN_GPUS:-}
RAY_WAIT_TIMEOUT_SEC=${RAY_WAIT_TIMEOUT_SEC:-120}
# 若集群提供 NODE_COUNT，可自动设置等待 GPU 数（未设置 NUM_GPUS_PER_NODE 时按每节点 8 卡算）
[[ -z "$RAY_MIN_GPUS" && -n "${NODE_COUNT:-}" ]] && RAY_MIN_GPUS=$((NODE_COUNT * ${NUM_GPUS_PER_NODE:-8}))
get_self_ip() {
  [[ -n "$HEAD_IP" ]] && { echo "$HEAD_IP"; return; }
  [[ -n "$POD_IP" ]] && { echo "$POD_IP"; return; }
  [[ -n "$SOCKET_IP" ]] && { echo "$SOCKET_IP"; return; }
  [[ -n "$MASTER_ADDR" ]] && { echo "$MASTER_ADDR"; return; }
  local ip; ip=$(hostname -i 2>/dev/null | awk '{print $1}')
  [[ -n "$ip" ]] && { echo "$ip"; return; }
  ip=$(hostname -I 2>/dev/null | awk '{print $1}')
  [[ -n "$ip" ]] && { echo "$ip"; return; }
  echo "127.0.0.1"
}
if [[ "${NODE_RANK}" -ge 1 ]]; then
  [[ -z "$HEAD_ADDRESS" ]] && { echo "ERROR: NODE_RANK=$NODE_RANK but HEAD_ADDRESS / RAY_HEAD_ADDRESS / MASTER_ADDR is not set." >&2; exit 1; }
  echo "[NODE_RANK=$NODE_RANK] Starting Ray worker, connecting to $HEAD_ADDRESS"
  ray start --address "$HEAD_ADDRESS" --num-cpus "${NUM_CPUS_PER_NODE}" --num-gpus "${NUM_GPUS_PER_NODE:-8}" --block
  exit 0
fi
if [[ "${NODE_RANK}" -eq 0 ]]; then
  self_ip=$(get_self_ip)
  echo "[NODE_RANK=0] Starting Ray HEAD at ${self_ip}:${RAY_PORT}"
  ray start --head --node-ip-address="$self_ip" --port="$RAY_PORT" \
    --num-cpus "${NUM_CPUS_PER_NODE}" --num-gpus "${NUM_GPUS_PER_NODE:-8}" --block &
  RAY_HEAD_PID=$!
  sleep 10
  if [[ -n "$RAY_MIN_GPUS" && "$RAY_MIN_GPUS" -gt 0 ]]; then
    echo "[NODE_RANK=0] Waiting for at least $RAY_MIN_GPUS GPUs (timeout ${RAY_WAIT_TIMEOUT_SEC}s)..."
    waited=0
    while [[ $waited -lt $RAY_WAIT_TIMEOUT_SEC ]]; do
      n=$(python3 -c "
import os, sys
os.environ.setdefault('RAY_ADDRESS', 'auto')
try:
  import ray
  ray.init(address='auto', ignore_reinit_error=True)
  r = ray.cluster_resources()
  print(int(r.get('GPU', 0)))
except Exception:
  print(0, file=sys.stderr)
" 2>/dev/null || echo "0")
      if [[ -n "$n" && "$n" -ge "$RAY_MIN_GPUS" ]]; then
        echo "[NODE_RANK=0] Cluster has $n GPUs, proceeding."
        break
      fi
      sleep 5; waited=$((waited + 5))
    done
    [[ $waited -ge $RAY_WAIT_TIMEOUT_SEC ]] && echo "[NODE_RANK=0] WARNING: Timeout waiting for $RAY_MIN_GPUS GPUs. Proceeding anyway."
  fi
  export RAY_ADDRESS="auto"
fi

# 2. 定义模型和数据路径
ACTOR_MODEL_PATH="/mnt/shared-storage-user/mineru4s/dingruiyi/WanJRxn_Downstream/output/0305_cot_v1_10e/v0-20260305-225317/checkpoint-0305_cot_v1_10e_10e"
FORWARD_MODEL_PATH="/mnt/shared-storage-user/mineru4s/shenxuli/qwen_smiles/output/0305_smiles2smiles_forward/v1-20260306-110334/checkpoint-99090"
VAL_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/USPTO-50k/raw_train_SL_COT_V1_50k_rl.parquet"
TRAIN_DATA="/mnt/shared-storage-user/mineru4s/dingruiyi/USPTO-50k/raw_train_SL_COT_V1_50k_rl.parquet"

# 3. 输出目录
OUTPUT_DIR="/mnt/shared-storage-gpfs2/mineru4s-gpfs2/dingruiyi/wanjuan-0305/checkpoints_rl/forward_rdkit_grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/"

# 4. 训练参数
NUM_GPUS_PER_NODE=8
NNODES=2
TRAIN_BATCH_SIZE=1024
LEARNING_RATE=1e-5
TOTAL_EPOCHS=10

export TIKTOKEN_ENCODINGS_BASE=/root/encoder
export TIKTOKEN_RS_CACHE_DIR=/root/encoder

cd /mnt/shared-storage-user/mineru4s/dingruiyi/WanjuanTraining
export VERL_FILE_LOGGER_PATH="($OUTPUT_DIR/log.jsonl)"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.top_p=0.1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.name=forward_rdkit \
    reward.reward_manager.source=register \
    reward.num_workers=8 \
    reward.reward_model.enable=False \
    +reward.forward_model.model_path="$FORWARD_MODEL_PATH" \
    +reward.forward_model.num_gpus=2 \
    +reward.beam_search_config.beam_width=3 \
    +reward.beam_search_config.max_tokens=512\
    +reward.beam_search_config.max_model_len=3176 \
    +reward.beam_search_config.gpu_memory_utilization=0.3 \
    +reward.beam_search_config.dtype=bfloat16 \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "file"]' \
    trainer.project_name='forward_rdkit_grpo' \
    trainer.experiment_name='qwen3.5_4b_forward_rdkit_grpo' \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=39 \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.val_before_train=False $@

train_exit=$?
[[ -n "${RAY_HEAD_PID:-}" ]] && kill "$RAY_HEAD_PID" 2>/dev/null || true
echo "训练完成! 输出目录: $OUTPUT_DIR"
exit ${train_exit}


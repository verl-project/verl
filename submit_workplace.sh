#!/bin/bash
#SBATCH --job-name=verl-nemogym-dapo-4b-workplace
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

GPUS_PER_NODE=8

source "$(dirname "$0")/config.env"

MODEL_PATH="${HF_HOME}/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
TRAIN_FILE="${DATA_ROOT}/workplace_assistant/train.jsonl"
TEST_FILE="${DATA_ROOT}/workplace_assistant/validation.jsonl"
CKPTS_DIR="${RESULTS_ROOT}/dapo-qwen3-4b-workplace"

CONTAINER="verlai/verl:vllm017.latest"
MOUNTS="/lustre:/lustre"

mkdir -p "${CKPTS_DIR}"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

RAY_PORT=6379
ip_head="${head_node_ip}:${RAY_PORT}"
echo "Head node: ${head_node} (${head_node_ip})"

SRUN_ARGS="--no-container-mount-home --container-image=${CONTAINER} --container-mounts=${MOUNTS} --container-workdir=${VERL_ROOT}"

echo "Starting Ray head on ${head_node}..."
srun --nodes=1 --ntasks=1 -w "${head_node}" ${SRUN_ARGS} --container-name=ray-head \
    env -u ROCR_VISIBLE_DEVICES WANDB_API_KEY="${WANDB_API_KEY}" ray start --head \
        --node-ip-address="${head_node_ip}" \
        --port=${RAY_PORT} \
        --num-gpus="${GPUS_PER_NODE}" \
        --block &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray worker ${i} on ${node_i}..."
    srun --nodes=1 --ntasks=1 -w "${node_i}" ${SRUN_ARGS} \
        env -u ROCR_VISIBLE_DEVICES WANDB_API_KEY="${WANDB_API_KEY}" ray start \
            --address="${ip_head}" \
            --num-gpus="${GPUS_PER_NODE}" \
            --block &
    sleep 5
done

CONTAINER_DIR="/raid/enroot/data/user-${UID}/pyxis_${SLURM_JOB_ID}_ray-head"
echo "Waiting for ray-head container at ${CONTAINER_DIR}..."
elapsed=0
while [[ ! -d "${CONTAINER_DIR}" && ${elapsed} -lt 300 ]]; do
    sleep 5
    elapsed=$((elapsed + 5))
done
if [[ ! -d "${CONTAINER_DIR}" ]]; then
    echo "ERROR: ray-head container never appeared after 300s"
    exit 1
fi
echo "Container ready. Waiting 90s for all Ray workers to connect..."
sleep 90

echo "Installing nemo-gym..."
srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" \
    --no-container-mount-home --container-mounts=${MOUNTS} \
    --container-name=ray-head \
    bash -c "PYTHONPATH= touch ${NEMO_GYM_ROOT}/scripts/__init__.py && pip install -q uv && echo 'blinker==1.4' > /tmp/constraints.txt && pip install -q -e ${NEMO_GYM_ROOT} -c /tmp/constraints.txt"

echo "Launching training on ${head_node}..."
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" \
    --no-container-mount-home --container-mounts=${MOUNTS} \
    --container-workdir=${VERL_ROOT} --container-name=ray-head \
    env -u ROCR_VISIBLE_DEVICES \
        WANDB_API_KEY="${WANDB_API_KEY}" \
        HF_HOME="${HF_HOME}" \
        HF_HUB_CACHE="${HF_HOME}/hub" \
        RAY_ADDRESS="auto" \
        VLLM_USE_V1=1 \
        TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        RAY_grpc_keepalive_time_ms=60000 \
        RAY_grpc_keepalive_timeout_ms=600000 \
        RAY_grpc_client_keepalive_time_ms=60000 \
        RAY_grpc_client_keepalive_timeout_ms=600000 \
    python3 -m verl.trainer.main_ppo \
            --config-path="${VERL_ROOT}/recipe/dapo/config" \
            --config-name=dapo_megatron_trainer.yaml \
            data.train_files="${TRAIN_FILE}" \
            data.val_files="${TEST_FILE}" \
            +data.custom_cls.path="${VERL_ROOT}/verl/experimental/nemo_gym_dataset.py" \
            +data.custom_cls.name=NemoGymJSONLDataset \
            data.truncation=left \
            data.train_batch_size=512 \
            actor_rollout_ref.rollout.n=16 \
            algorithm.adv_estimator=grpo \
            algorithm.use_kl_in_reward=False \
            algorithm.kl_ctrl.kl_coef=0.0 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.0 \
            actor_rollout_ref.actor.clip_ratio_low=0.2 \
            actor_rollout_ref.actor.clip_ratio_high=0.28 \
            actor_rollout_ref.actor.clip_ratio_c=10.0 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.model.path="${MODEL_PATH}" \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
            actor_rollout_ref.actor.optim.weight_decay=0.1 \
            actor_rollout_ref.actor.ppo_mini_batch_size=32 \
            actor_rollout_ref.actor.megatron.param_offload=True \
            actor_rollout_ref.actor.megatron.optimizer_offload=True \
            actor_rollout_ref.actor.megatron.grad_offload=True \
            actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
            actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.actor.optim.clip_grad=1.0 \
            actor_rollout_ref.actor.loss_agg_mode=token-mean \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
            actor_rollout_ref.rollout.enable_chunked_prefill=True \
            actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.rollout.top_p=1.0 \
            actor_rollout_ref.rollout.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
            actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            actor_rollout_ref.rollout.val_kwargs.n=1 \
            actor_rollout_ref.rollout.name=vllm \
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable-auto-tool-choice=true' \
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.tool-call-parser=hermes_patched' \
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool-parser-plugin=${VERL_ROOT}/verl/experimental/tool_parsers/hermes_tool_parser_patched.py" \
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.max-model-len=32768' \
            actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
            actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
            actor_rollout_ref.ref.megatron.param_offload=True \
            reward_model.reward_manager=dapo \
            +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
            +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
            +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
            +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
            +reward_model.reward_kwargs.max_resp_len=8192 \
            'trainer.logger=["console","wandb"]' \
            trainer.project_name=${WANDB_USERNAME}-verl-nemogym-int \
            trainer.experiment_name=dapo-qwen3-4b-workplace \
            trainer.n_gpus_per_node=${GPUS_PER_NODE} \
            trainer.nnodes=${SLURM_JOB_NUM_NODES} \
            trainer.val_before_train=False \
            trainer.test_freq=10 \
            trainer.save_freq=10 \
            trainer.total_epochs=10 \
            trainer.default_local_dir="${CKPTS_DIR}" \
            trainer.resume_mode=auto \
            trainer.log_val_generations=10 \
            +actor_rollout_ref.rollout.agent.agent_loop_manager_class='verl.experimental.nemo_gym_agent_loop.NemoGymAgentLoopManager' \
            "+actor_rollout_ref.rollout.agent.nemo_gym.initial_global_config_dict.config_paths=[${NEMO_GYM_ROOT}/responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,${NEMO_GYM_ROOT}/resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
            +actor_rollout_ref.rollout.agent.nemo_gym.uses_reasoning_parser=False \
            +actor_rollout_ref.rollout.agent.nemo_gym.nemo_gym_root="${NEMO_GYM_ROOT}" \
    2>&1

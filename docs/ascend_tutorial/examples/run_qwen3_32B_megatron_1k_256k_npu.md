# Long Sequence Qwen3-32B 1k-to-256k Example

Last updated: 6/3/2026.

本章对Qwen3-32B进行了长序列开发。Qwen3-32B的模型能力为最长推到40k

## 全层实验

对Qwen3-32B进行了长序列开发，脚本如下：

```bash
set -x

export USE_OPTIMIZED_MODEL=0
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_VERSION="0.13.0"
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

PROJECT_NAME="GRPO-Qwen3-32B"
EXPERIMENT_NAME="GRPO-Qwen3-32B-megatron-gsm8k"

SAVE_CHECKPOINT_DIR=$HOME/verl_checkpoints
math_train_path=$HOME/datasets/gsm8k/train.parquet
math_test_path=$HOME/datasets/gsm8k/test.parquet
train_files="['$math_train_path']"
test_files="['$math_test_path']"

use_dynamic_bsz=False
enable_chunked_prefill=True
tp_size=8
max_prompt_length=1024
max_response_length=$((1024*256))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / tp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / tp_size))
cp_size=4

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.shuffle=False \
    data.validation_shuffle=False \
    data.train_batch_size=64 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.path=$HOME/hf_weights/Qwen3-32B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${cp_size} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${cp_size} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${SAVE_CHECKPOINT_DIR} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${SAVE_CHECKPOINT_DIR} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=2 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=100 \
    trainer.device=npu \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=30
```

- 相关实验结果
  
![qwen3-32b-perfo](https://github.com/ChibiQuest/verl_data/blob/main/qwen3-32B-1k-256k/performance.png)

## 减层实验

在实际推理中，我们发现其最大在20k左右，因此对其进行减层实验，其response能到达到40k。

在权重的`config.json`文件中，我们将`num_hidden_layers`从64减层到16

```
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 25600,
  "max_position_embeddings": 40960,
  "max_window_layers": 64,
  "model_type": "qwen3",
  "num_attention_heads": 64,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

```

- 其实验结果如下：

![qwen3-32b-function](https://github.com/ChibiQuest/verl_data/blob/main/qwen3-32B-1k-256k/function.png)
#!/usr/bin/env bash
# GRPO training of Qwen3.8-Flash-Next with VeOmni and vLLM on Ascend NPUs.
# Prepare CANN/ATB, the Python environment and compatible model implementations
# on every node, then start a Ray cluster and run this script on its head node.
# Deployment-specific memory workarounds must be configured in that environment.
# Default topology: 4 x 16 NPUs; actor FSDP=64, EP=32, PLE=8, SP=1;
# rollout TP=8, DP=8, EP=64. Additional Hydra overrides may be passed as arguments.
set -euo pipefail

# ===================================== Data / Model =====================================
data_path=${data_path:-$HOME/data/geo3k}
model_path=${model_path:-$HOME/model/Qwen3.8-Flash-Next}
output_path=${output_path:-$HOME/output}
project_name=${project_name:-verl_grpo_qwen3_8_flash_next_geo3k}
exp_name=${exp_name:-qwen3_8_flash_next_veomni_$(date +%Y%m%d_%H%M%S)}
run_dir=$output_path/$project_name/$exp_name
max_prompt_length=${max_prompt_length:-1024}
max_response_length=${max_response_length:-4096}
max_model_len=$((max_prompt_length + max_response_length))
train_batch_size=${train_batch_size:-32}
rollout_n=${rollout_n:-4}
actor_lr=${actor_lr:-1e-6}
total_training_steps=${total_training_steps:-100}

# ===================================== Parallelism =====================================
nnodes=${nnodes:-4}
npu_count=${npu_count:-16}
fsdp_size=${fsdp_size:-64}
expert_parallel_size=${expert_parallel_size:-32}
ple_parallel_size=${ple_parallel_size:-8}
infer_tp=${infer_tp:-8}
infer_dp=${infer_dp:-8}
infer_ep=${infer_ep:-64}
# Use an existing cluster; set RAY_ADDRESS explicitly when multiple clusters exist.
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VERL_FILE_LOGGER_ROOT="$run_dir/logs"

CONFIG=(
  "--config-path=./config"
  "--config-name=ppo_trainer"
  "model_engine=veomni"
)

ALGORITHM=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=false"
)

DATA=(
  "data.train_files=$data_path/train.parquet"
  "data.val_files=$data_path/test.parquet"
  "data.train_batch_size=$train_batch_size"
  "data.max_prompt_length=$max_prompt_length"
  "data.max_response_length=$max_response_length"
  "data.filter_overlong_prompts=true"
  "data.truncation=error"
)

ACTOR=(
  "actor_rollout_ref.model.path=$model_path"
  "actor_rollout_ref.model.trust_remote_code=true"
  "actor_rollout_ref.model.enable_gradient_checkpointing=true"
  "actor_rollout_ref.model.use_remove_padding=true"
  "actor_rollout_ref.actor.optim.lr=$actor_lr"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=true"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_model_len"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.use_kl_loss=false"
  "actor_rollout_ref.actor.entropy_coeff=0"
)

ACTOR_VEOMNI=(
  "actor_rollout_ref.actor.veomni.fsdp_size=$fsdp_size"
  "actor_rollout_ref.actor.veomni.expert_parallel_size=$expert_parallel_size"
  "actor_rollout_ref.actor.veomni.ple_parallel_size=$ple_parallel_size"
  "actor_rollout_ref.actor.veomni.ulysses_parallel_size=1"
  "actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2"
  "actor_rollout_ref.actor.veomni.qsa_attention_implementation=torch_npu"
  "actor_rollout_ref.actor.veomni.causal_conv1d_implementation=triton"
  "actor_rollout_ref.actor.veomni.chunk_gated_delta_rule_implementation=triton"
  "actor_rollout_ref.actor.veomni.broadcast_model_weights_from_rank0=false"
  "actor_rollout_ref.actor.veomni.ep_sharded_stream_load=true"
  "actor_rollout_ref.actor.veomni.mixed_precision=true"
  "actor_rollout_ref.actor.veomni.enable_autocast=false"
  "actor_rollout_ref.actor.veomni.param_offload=true"
  "actor_rollout_ref.actor.veomni.optimizer_offload=true"
  "actor_rollout_ref.actor.veomni.use_torch_compile=false"
  "actor_rollout_ref.actor.veomni.forward_prefetch=false"
  "actor_rollout_ref.actor.veomni.moe_implementation=fused_npu"
  "actor_rollout_ref.actor.veomni.freeze_ple_embeddings=true"
  "actor_rollout_ref.actor.veomni.enable_async_activation_offload=true"
  "actor_rollout_ref.actor.veomni.activation_offload_modules=[model.language_model.layers.*]"
)

ROLLOUT=(
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.load_format=safetensors"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp"
  "actor_rollout_ref.rollout.data_parallel_size=$infer_dp"
  "actor_rollout_ref.rollout.expert_parallel_size=$infer_ep"
  "actor_rollout_ref.rollout.n=$rollout_n"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_model_len"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.70"
  "actor_rollout_ref.rollout.free_cache_engine=true"
  "+actor_rollout_ref.rollout.enable_sleep_mode=true"
  "actor_rollout_ref.rollout.enforce_eager=false"
  "actor_rollout_ref.rollout.enable_prefix_caching=true"
  "actor_rollout_ref.rollout.enable_chunked_prefill=true"
  "actor_rollout_ref.rollout.max_model_len=$max_model_len"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$max_model_len"
  "actor_rollout_ref.rollout.max_num_seqs=32"
  "actor_rollout_ref.rollout.ignore_eos=false"
  # Ascend does not support the UVA lookup used by Engram CPU offload.
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.engram_config.cpu_offload=false"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.engram_config.dp_shared_memory=false"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=false"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.compile_mm_encoder=false"
  "++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=false"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY"
)

ROUTER_REPLAY=(
  "actor_rollout_ref.actor.veomni.router_replay.mode=R3"
  "actor_rollout_ref.rollout.enable_rollout_routing_replay=True"
)

TRAINER=(
  "trainer.device=npu"
  "trainer.n_gpus_per_node=$npu_count"
  "trainer.nnodes=$nnodes"
  "trainer.logger=[console,file]"
  "trainer.project_name=$project_name"
  "trainer.experiment_name=$exp_name"
  "trainer.val_before_train=false"
  "trainer.test_freq=-1"
  "trainer.save_freq=10"
  "trainer.max_actor_ckpt_to_keep=1"
  "trainer.resume_mode=auto"
  "trainer.total_training_steps=$total_training_steps"
  "trainer.total_epochs=100"
  "trainer.default_local_dir=$run_dir/checkpoints"
  "trainer.rollout_data_dir=$run_dir/rollouts"
)

RAY=(
  "++ray_kwargs.ray_init.address=$RAY_ADDRESS"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER='1'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD=spawn"
)

# ===================================== Launch =====================================
python -m verl.trainer.main_ppo \
  "${CONFIG[@]}" \
  "${ALGORITHM[@]}" \
  "${DATA[@]}" \
  "${ACTOR[@]}" \
  "${ACTOR_VEOMNI[@]}" \
  "${ROLLOUT[@]}" \
  "${ROUTER_REPLAY[@]}" \
  "${TRAINER[@]}" \
  "${RAY[@]}" \
  "hydra.run.dir=$run_dir/hydra" \
  "$@"

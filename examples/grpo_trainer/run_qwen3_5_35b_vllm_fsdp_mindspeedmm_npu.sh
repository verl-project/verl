ray stop --force
ps -aux | grep "VLLM" | grep -v grep| awk '{print $2}' | xargs kill -9 | pkill -9 python

export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.85

project_name='GRPO-Qwen3_5'
exp_name='GRPO-Qwen3_5-35b-npu'
gen_tp=8
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-35B-A3B"}
DCP_MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-35B-A3B-DCP"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/geo3k/test.parquet"}

start_time=$(date +%Y%m%d)_$(date +%H%M%S)


DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=16
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.image_key=images
    data.shuffle=False
)

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.logger=['console','tensorboard']
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=16
    trainer.nnodes=1
    trainer.balance_batch=False
    trainer.resume_from_path=checkpoints
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.total_epochs=10
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.lr_warmup_ratio=-1
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=adamw
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_torch_compile=False
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,8,12,16,24,32,48,56,64]"
)

MINDSPEED_CONFIG=(
    actor_rollout_ref.actor.mindspeed.strategy=mindspeed_fsdp
    actor_rollout_ref.actor.mindspeed.model_name=qwen3.5-35b
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.load=${DCP_MODEL_PATH}
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.plugin='[mindspeed_mm/fsdp/models/qwen3_5_moe, mindspeed_mm/fsdp/data/datasets/huggingface]'
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.model_id=qwen3_5_moe
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.use_triton_gdn=True
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.freeze='[model.visual]'
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.use_grouped_expert_matmul=True
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.apply_modules="['model.visual.blocks.{*}', \
        'model.visual', 'model.language_model.layers.{*}.linear_attn', 'model.language_model.layers.{*}.mlp', 'model.language_model.layers.{*}', \
        'model.language_model.embed_tokens', 'model.language_model.norm', 'model.language_model.rotary_emb', 'model.language_model', 'lm_head']"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.recompute=True
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.recompute_plan.apply_modules="['model.language_model.layers.{*}', 'model.visual.blocks.{*}']"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.hook_modules="['model.language_model.layers.{*}']"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.ep_plan.apply_modules="['model.language_model.layers.{*}.mlp.experts']"
    actor_rollout_ref.actor.mindspeed.reshard_after_forward=True
    actor_rollout_ref.actor.mindspeed.ulysses_sequence_parallel_size=1
    actor_rollout_ref.ref.mindspeed.ulysses_sequence_parallel_size=1
    actor_rollout_ref.ref.mindspeed.reshard_after_forward=True
    actor_rollout_ref.actor.mindspeed.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.mindspeed.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.mindspeed.wrap_policy.min_num_params=0
    actor_rollout_ref.actor.mindspeed.offload_policy=True
    actor_rollout_ref.ref.mindspeed.offload_policy=True
    actor_rollout_ref.actor.mindspeed.param_offload=True
    actor_rollout_ref.actor.mindspeed.optimizer_offload=True
    actor_rollout_ref.ref.mindspeed.param_offload=True
)

REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.optim.optimizer=adamw
)


mkdir -p logs
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    model_engine=mindspeed \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${MINDSPEED_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    critic.strategy=fsdp2 \
    $@ 2>&1 | tee logs/qwen3_5-35b-${start_time}.log

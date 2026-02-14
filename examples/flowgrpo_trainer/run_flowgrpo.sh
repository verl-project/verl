# Qwen-Image lora, vllm_omni rollout
set -x
export TOKENIZERS_PARALLELISM="false"

ENGINE=vllm_omni
REWARD_ENGINE=vllm

reward_path=tests/experimental/reward_loop/reward_fn.py
reward_model_name=$HOME/models/Qwen/Qwen2.5-VL-3B-Instruct


python3 -m verl.trainer.main_flowgrpo \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$HOME/dataset/ocr/train.txt \
    data.val_files=$HOME/dataset/ocr/test.txt \
    data.train_batch_size=32 \
    data.val_max_samples=128 \
    data.max_prompt_length=1058 \
    data.filter_overlong_prompts=True \
    data.data_source=ocr \
    data.custom_cls.path=verl/utils/dataset/qwen_dataset.py \
    data.custom_cls.name=QwenDataset \
    +data.apply_chat_template_kwargs.max_length=1058 \
    +data.apply_chat_template_kwargs.padding=True \
    +data.apply_chat_template_kwargs.truncation=True \
    actor_rollout_ref.model.path=$HOME/models/Qwen/Qwen-Image \
    actor_rollout_ref.model.tokenizer_path=$HOME/models/Qwen/Qwen-Image/tokenizer \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.policy_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.guidance_scale=1.0 \
    actor_rollout_ref.rollout.agent.default_agent_loop=diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_model_len=1058 \
    actor_rollout_ref.rollout.sde_window_size=3 \
    actor_rollout_ref.rollout.sde_window_range="[0,5]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.custom_pipeline=verl.workers.utils.vllm_omni_patch.pipelines.pipeline_qwenimage.QwenImagePipelineWithLogProb \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    reward.reward_manager.name=diffusion \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.enable=True \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.custom_reward_function.path=$reward_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_ocr \
    trainer.log_val_generations=8 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

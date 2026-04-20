# Qwen-Image lora RL, vllm_omni rollout
set -x

ocr_train_path=$HOME/data/ocr/train.parquet
ocr_test_path=$HOME/data/ocr/test.parquet

ENGINE=vllm_omni
REWARD_ENGINE=vllm

reward_path=examples/flowgrpo_trainer/reward_fn.py
reward_model_name=$HOME/models/Qwen/Qwen3-VL-8B-Instruct


python3 -m verl.trainer.main_flowgrpo \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$HOME/models/Qwen/Qwen-Image \
    actor_rollout_ref.model.tokenizer_path=$HOME/models/Qwen/Qwen-Image/tokenizer \
    actor_rollout_ref.model.external_lib="examples.flowgrpo_trainer.diffusers.qwen_image" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.policy_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.default_agent_loop=diffusion_single_turn_agent \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.val_kwargs.num_inference_steps=50 \
    +actor_rollout_ref.rollout.extra_configs.true_cfg_scale=4.0 \
    +actor_rollout_ref.rollout.extra_configs.noise_level=1.2 \
    +actor_rollout_ref.rollout.extra_configs.sde_type="sde" \
    +actor_rollout_ref.rollout.extra_configs.sde_window_size=2 \
    +actor_rollout_ref.rollout.extra_configs.sde_window_range="[0,5]" \
    +actor_rollout_ref.rollout.extra_configs.max_sequence_length=256 \
    +actor_rollout_ref.rollout.val_kwargs.extra_configs.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.custom_pipeline=examples.flowgrpo_trainer.vllm_omni.pipeline_qwenimage.QwenImagePipelineWithLogProb \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    reward.num_workers=4 \
    reward.reward_manager.name=visual \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=4 \
    reward.custom_reward_function.path=$reward_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_ocr_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 $@

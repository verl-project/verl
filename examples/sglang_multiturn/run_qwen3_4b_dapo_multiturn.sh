set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

pip install --upgrade "huggingface-hub>=0.34.0"
hf download \
    BytedTsinghua-SIA/DAPO-Math-17k \
    --repo-type dataset \
    --local-dir $HOME/data/BytedTsinghua-SIA/DAPO-Math-17k


hf download \
    Maxwell-Jia/AIME_2024 \
    --repo-type dataset \
    --local-dir $HOME/data/Maxwell-Jia/AIME_2024


# Note:
# 1. 
# a sandbox fusion server is needed to run the code interpreter tool.
# docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609

# 2. 
# The model located at font-info/qwen3-4b-sft-SGLang-RL (https://huggingface.co/font-info/qwen3-4b-sft-SGLang-RL)
# is a fine-tuned version provided by the SGLang RL team. Without supervised fine-tuning (SFT)
# on the Retool dataset, Dapo training will not converge.

# If you still wish to perform SFT from scratch, follow the steps below:

# Step 1: Download the SFT dataset
#hf download JoeYing/ReTool-SFT --repo-type dataset --local-dir ./ReTool-SFT

# Step 2: Preprocess the data for SFT
#python3 recipe/retool/retool_sft_preprocess.py

# Step 3: Run SFT training
#bash recipe/retool/run_qwen2-32b_sft.sh

# having trouble setup? see https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/release_log/latest_sglang.md for more details.


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=$HOME/data/BytedTsinghua-SIA/DAPO-Math-17k \
    data.val_files=$HOME/data/Maxwell-Jia/AIME_2024 \
    data.return_raw_chat=True \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$PROJECT_DIR/recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=$PROJECT_DIR/recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=font-info/qwen3-4b-sft-SGLang-RL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=16 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$PROJECT_DIR/recipe/retool/sandbox_fusion_tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=30 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=sglang-dapo-multiturn \
    trainer.experiment_name=qwen3_4b_sft_dapo_multiturn \
    trainer.n_gpus_per_node=8 \
    trainer.log_val_generations=20 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    $@

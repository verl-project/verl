train_files=data/full_hh_rlhf/rl/train.parquet
test_files=data/full_hh_rlhf/rl/train.parquet # no use

max_prompt_length=4096
max_response_length=2048

gen_tp=4
n_per_prompt=5
adv_estimator="grpo"

project_name=verl_full_hh_rlhf_examples
exp_name="grpo_mistral13B-skyworkLlama8b-hhrlhf"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.prompt_key="prompt" \
    data.return_raw_chat=True \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=mistralai/Mistral-Nemo-Instruct-2407 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$n_per_prompt \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.enable=True \
    reward_model.model.path=Skywork/Skywork-Reward-Llama-3.1-8B \
    reward_model.use_reward_loop=True \
    reward_model.rollout.name=vllm \
    reward_model.rollout.gpu_memory_utilization=0.8 \
    reward_model.rollout.tensor_model_parallel_size=1 \
    reward_model.rollout.prompt_length=8192 \
    reward_model.rollout.response_length=4096 \
    reward_model.num_workers=8 \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=False \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=5 $@
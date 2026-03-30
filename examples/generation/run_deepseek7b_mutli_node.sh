set -x

data_path=$HOME/data/rlhf/gsm8k/test.parquet
save_path=$HOME/data/rlhf/math/deepseek_v2_lite_gen_test.parquet
model_path=deepseek-ai/deepseek-llm-7b-chat

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    data.train_files=$data_path \
    data.prompt_key=prompt \
    +data.output_path=$save_path \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8

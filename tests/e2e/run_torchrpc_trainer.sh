#!/usr/bin/env bash

set -e -x

OUTPUT_FILE="/tmp/output_torchrpc_trainer.txt"

export PATH=$PATH:~/.local/bin

rm -rf $OUTPUT_FILE
MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT TORCHRPC_RANK=$TORCHRPC_RANK TORCHRPC_WORLD_SIZE=$TORCHRPC_WORLD_SIZE \
python3 tests/e2e/arithmetic_sequence/rl/main_trainer_torchrpc.py \
    algorithm.adv_estimator=gae \
    data.train_files=tests/e2e/arithmetic_sequence/data/train.parquet \
    data.val_files=tests/e2e/arithmetic_sequence/data/test.parquet \
    data.train_batch_size=800 \
    data.max_prompt_length=16 \
    data.max_response_length=32 \
    data.return_raw_input_ids=True \
    actor_rollout_ref.model.path=tests/e2e/arithmetic_sequence/model \
    actor_rollout_ref.model.external_lib=tests.e2e.envs.digit_completion \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=200 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.path=tests/e2e/arithmetic_sequence/model \
    critic.optim.lr=1e-3 \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs=200 \
    trainer.experiment_name=arithmetic_sequences \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=8 \
    trainer.test_freq=1 \
    trainer.save_freq=110 | tee $OUTPUT_FILE;

python3 tests/e2e/check_results.py --output_file=$OUTPUT_FILE
rm -rf $OUTPUT_FILE

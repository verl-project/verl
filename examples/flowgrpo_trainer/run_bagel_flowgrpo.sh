# Bagel LoRA RL, vllm_omni rollout (FlowGRPO)
#
# Prerequisites:
#   1. A Bagel model (e.g. BAGEL-7B-MoT) at $BAGEL_MODEL_PATH
#   2. A stage config YAML at $BAGEL_STAGE_CONFIG for vllm-omni
#   3. DiffusionModelBase registered as "OmniBagelForConditionalGeneration"
#      at examples/flowgrpo_trainer/diffusers/bagel.py
#   4. A reward VLM model at $REWARD_MODEL_PATH
#   5. OCR training data at $OCR_TRAIN_PATH / $OCR_TEST_PATH
#      (generate via: python examples/flowgrpo_trainer/prepare_ocr_data.py)
#
# Usage:
#   export BAGEL_MODEL_PATH=/path/to/BAGEL-7B-MoT
#   export REWARD_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
#   bash examples/flowgrpo_trainer/run_bagel_flowgrpo.sh
#
#   # Override any param via CLI:
#   bash examples/flowgrpo_trainer/run_bagel_flowgrpo.sh trainer.n_gpus_per_node=8

set -x

# --------------- Paths (override via environment) ---------------
BAGEL_MODEL_PATH=${BAGEL_MODEL_PATH:-$HOME/models/BAGEL-7B-MoT}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_STAGE_CONFIG=${BAGEL_STAGE_CONFIG:-$SCRIPT_DIR/bagel_stage_config.yaml}

REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-$HOME/models/Qwen3-VL-8B-Instruct}

ocr_train_path=${OCR_TRAIN_PATH:-$HOME/data/ocr/train.parquet}
ocr_test_path=${OCR_TEST_PATH:-$HOME/data/ocr/test.parquet}

ENGINE=vllm_omni
REWARD_ENGINE=vllm

reward_path=examples/flowgrpo_trainer/reward_fn.py

python3 -m verl.trainer.main_flowgrpo \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$BAGEL_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$BAGEL_MODEL_PATH \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.external_lib="examples.flowgrpo_trainer.diffusers.bagel" \
    actor_rollout_ref.model.height=512 \
    actor_rollout_ref.model.width=512 \
    actor_rollout_ref.model.num_inference_steps=15 \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen']" \
    actor_rollout_ref.actor.optim.lr=1e-3 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.policy_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=diffusion_single_turn_agent \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.num_inference_steps=15 \
    actor_rollout_ref.rollout.val_kwargs.num_inference_steps=15 \
    +actor_rollout_ref.rollout.extra_configs.noise_level=1.2 \
    +actor_rollout_ref.rollout.extra_configs.sde_type="sde" \
    +actor_rollout_ref.rollout.extra_configs.sde_window_size=2 \
    +actor_rollout_ref.rollout.extra_configs.sde_window_range="[0,5]" \
    +actor_rollout_ref.rollout.extra_configs.max_sequence_length=256 \
    +actor_rollout_ref.rollout.val_kwargs.extra_configs.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.custom_pipeline=examples.flowgrpo_trainer.vllm_omni.pipeline_bagel.BagelPipelineWithLogProb \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path=$BAGEL_STAGE_CONFIG \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    reward.num_workers=1 \
    reward.reward_manager.name=visual \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$REWARD_MODEL_PATH \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=4 \
    reward.custom_reward_function.path=$reward_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=bagel_ocr_lora \
    trainer.log_val_generations=4 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=50 $@

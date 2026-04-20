# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import tempfile

import numpy as np
import pytest
import ray
from omegaconf import DictConfig, open_dict

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto

pytestmark = pytest.mark.vllm_omni


def _create_tp_compatible_model(parent_dir, src_model_path, num_attention_heads=2):
    """Copy base model and recreate transformer on-the-fly with TP-compatible head count.

    The tiny-random Qwen-Image model has num_attention_heads=1 in its transformer config,
    which is not divisible by tensor_model_parallel_size=2. This helper copies the full
    model directory (vae, text_encoder, tokenizer, scheduler) and overwrites only the
    transformer component with a freshly-initialized one that has the desired head count.
    """
    from diffusers import QwenImageTransformer2DModel

    dst = os.path.join(parent_dir, "Qwen-Image")
    shutil.copytree(src_model_path, dst)

    transformer = QwenImageTransformer2DModel(
        num_attention_heads=num_attention_heads,
        attention_head_dim=32,
        num_layers=2,
        in_channels=64,
        out_channels=16,
        patch_size=2,
        joint_attention_dim=32,
        axes_dims_rope=(8, 12, 12),
        guidance_embeds=False,
    )
    transformer.save_pretrained(os.path.join(dst, "transformer"))

    return dst


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="diffusion_trainer")

    base_model_path = os.path.expanduser("~/models/tiny-random/Qwen-Image")
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = _create_tp_compatible_model(tmp_dir, base_model_path, num_attention_heads=2)
        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.model.tokenizer_path = os.path.join(model_path, "tokenizer")
        config.actor_rollout_ref.rollout.name = "vllm_omni"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.enforce_eager = True
        config.actor_rollout_ref.rollout.n = 4
        config.actor_rollout_ref.rollout.num_inference_steps = 10
        config.actor_rollout_ref.rollout.calculate_log_probs = True
        config.actor_rollout_ref.rollout.agent.num_workers = 2
        config.actor_rollout_ref.rollout.agent.default_agent_loop = "diffusion_single_turn_agent"
        tokenizer_max_length = 1024
        prompt_template_encode_start_idx = 34
        max_length = tokenizer_max_length + prompt_template_encode_start_idx

        with open_dict(config.actor_rollout_ref.rollout.extra_configs):
            config.actor_rollout_ref.rollout.extra_configs.true_cfg_scale = 4.0
            config.actor_rollout_ref.rollout.extra_configs.max_sequence_length = max_length
            config.actor_rollout_ref.rollout.extra_configs.noise_level = 1.0
            config.actor_rollout_ref.rollout.extra_configs.sde_window_size = 2
            config.actor_rollout_ref.rollout.extra_configs.sde_window_range = [0, 5]

        config.actor_rollout_ref.rollout.nnodes = 1

        qwen_pipeline = "examples.flowgrpo_trainer.vllm_omni.pipeline_qwenimage.QwenImagePipelineWithLogProb"
        config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni = {"custom_pipeline": qwen_pipeline}
        config.reward.reward_manager.name = "image"
        config.trainer.n_gpus_per_node = 4

        config.data.apply_chat_template_kwargs = dict(max_length=max_length, padding=True, truncation=True)
        config.data.max_prompt_length = max_length
        config.actor_rollout_ref.rollout.max_model_len = max_length

        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2

        yield config


def test_single_turn(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        }
    )
    try:
        agent_loop_manager = AgentLoopManager.create(init_config)

        system_prompt = (
            "Describe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:"
        )
        user_prompts = ["A photo of cute cat with long fur and big eyes.", "A photo of cute dog with short hair."]

        raw_prompts = []
        for user_prompt in user_prompts:
            raw_prompts.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

        raw_negative_prompts = []
        for user_prompt in user_prompts:
            raw_negative_prompts.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": " "},
                ]
            )

        batch = DataProto(
            non_tensor_batch={
                "raw_prompt": np.array(raw_prompts),
                "raw_negative_prompt": np.array(raw_negative_prompts),
                "data_source": np.array(["jpeg_compressibility"] * len(raw_prompts)),
                "reward_model": np.array([{"style": "rule", "ground_truth": ""}] * len(raw_prompts)),
            },
        )
        n = init_config.actor_rollout_ref.rollout.n
        batch = batch.repeat(n)
        result = agent_loop_manager.generate_sequences(prompts=batch)
        assert len(result) == len(raw_prompts) * n

        expected_batch_keys = [
            "responses",
            "all_latents",
            "all_timesteps",
            "prompt_embeds",
            "prompt_embeds_mask",
            "input_ids",
            "attention_mask",
            "rollout_log_probs",
        ]
        for key in expected_batch_keys:
            assert key in result.batch, f"Key {key} not found in result batch with keys {list(result.batch.keys())}."

        # check turns
        num_turns = result.non_tensor_batch["__num_turns__"]
        assert np.all(num_turns == 2)

        print("Test passed!")
    finally:
        ray.shutdown()

"""JAX-based rollout using OpenPI checkpoint for pi05 inference.

This replaces PI0RolloutRob for the rollout phase, using the official JAX
checkpoint directly instead of the converted PyTorch weights.
"""

import logging

import torch

from verl import DataProto
from verl.experimental.vla.naive_rollout_rob import NaiveRolloutRob

logger = logging.getLogger(__name__)

__all__ = ["PI0JaxRolloutRob"]


class PI0JaxRolloutRob(NaiveRolloutRob):
    """Rollout that uses JAX for VLA inference. Critic is skipped (dummy zeros)."""

    def __init__(
        self,
        model_config: dict,
        module: torch.nn.Module,
        tokenizer,
        jax_policy,
    ):
        self.model_config = model_config
        self.module = module
        self.tokenizer = tokenizer
        self.jax_policy = jax_policy

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate action sequences using JAX inference only (no critic)."""
        from verl.utils.device import get_device_id

        prompts.to(get_device_id())

        result = self.jax_policy.infer_from_dataproto(prompts)
        action = result["action"].to(get_device_id())
        full_action = result["full_action"].to(get_device_id())
        batch_size = action.shape[0]

        from verl.experimental.vla.models.pi0_torch.policy.libero_policy import LiberoPi0Input

        pi0_input = LiberoPi0Input.from_env_obs(prompts)
        state = pi0_input.state.to(get_device_id(), dtype=torch.float32)

        dummy_images = torch.zeros(batch_size, 3, 3, 224, 224, device=get_device_id(), dtype=torch.bfloat16)
        dummy_img_masks = torch.zeros(batch_size, 3, device=get_device_id(), dtype=torch.bool)
        dummy_lang_tokens = torch.zeros(batch_size, 1, device=get_device_id(), dtype=torch.long)
        dummy_lang_masks = torch.zeros(batch_size, 1, device=get_device_id(), dtype=torch.bool)
        dummy_critic = torch.zeros(batch_size, device=get_device_id(), dtype=torch.float32)

        tensor_batch = {
            "action": action,
            "full_action": full_action,
            "images": dummy_images,
            "image_masks": dummy_img_masks,
            "lang_tokens": dummy_lang_tokens,
            "lang_masks": dummy_lang_masks,
            "states": state,
            "critic_value": dummy_critic,
        }

        return DataProto.from_dict(tensor_batch)

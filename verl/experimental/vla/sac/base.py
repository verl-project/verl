# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch

from verl import DataProto


class SupportSACTraining:
    """
    Base class for Soft Actor-Critic (SAC).

    Subclasses implement a Policy that can be plugged directly into SAC training.
    This implementation requires the actor and critic to be integrated within a
    single model instance, e.g., sharing a backbone with an additional MLP head
    that outputs critic values (Q/V) alongside the actor's action distribution.

    Note:
        This class intentionally does NOT inherit from `abc.ABC`.
        The root model may be wrapped or transformed by FSDP (Fully Sharded
        Data Parallel), which performs runtime class substitution; using
        `ABCMeta` can break FSDP's class rewriting mechanism.
    """

    def sac_init(self):
        raise NotImplementedError("Subclasses must implement sac_init method.")

    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: Any,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Compute Q-values for given state-action pairs.
        Args:
            a: Dictionary of tensors representing actions, with key:
                - "full_action": torch.Tensor of shape (B, action_steps, action_dim)
            state_features: Any data structure representing the processed state features.
            use_target_network: Whether to use the target critic network heads.
            method: Method to combine multiple heads' outputs ("cat" or "min").
            requires_grad: Whether to enable gradients for the critic head parameters.

        Returns:
            q_values: torch.Tensor of shape (B, num_heads) if method is "cat",
                      or (B, 1) if method is "min", representing the computed Q-values
        """

        raise NotImplementedError("Subclasses must implement sac_forward_critic method.")

    def sac_forward_actor(
        self,
        state_features: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute actions and their log probabilities from state features.

        Args:
            state_features: Any data structure representing the processed state features.

        Returns:
            actions: torch.Tensor of shape (B, n_action_steps, action_dim), sampled actions.
            log_probs: torch.Tensor of shape (B,), log probabilities of the sampled actions.
        """

        raise NotImplementedError("Subclasses must implement sac_forward_actor method.")

    def sac_forward_state_features(self, s: dict[str, torch.Tensor]) -> Any:
        """Compute state features needed for SAC actor and critic.

        Args:
            s: Dictionary of tensors representing the states, with keys
                - "images": torch.Tensor of shape (B, n_images, C, H, W)
                - "image_masks": torch.Tensor of shape (B, n_images)
                - "lang_tokens": torch.Tensor of shape (B, L)
                - "lang_masks": torch.Tensor of shape (B, L)
                - "states": torch.Tensor of shape (B, state_dim)
        Returns:
            state_features: Any data structure representing the processed state features.
        """

        raise NotImplementedError("Subclasses must implement sac_forward_state_features method.")

    def sac_update_target_network(self, tau: float):
        """Update the target network heads using Polyak averaging.

        Args:
            tau: The interpolation parameter for Polyak averaging.
        """

        raise NotImplementedError("Subclasses must implement sac_update_target_network method.")


class BaseSACActor(ABC):
    @abstractmethod
    def update_policy(self, data: DataProto) -> dict:
        """
        Update the policy using the provided data batch.

        Args:
            data: DataProto containing the following entries in `data.batch`:
                - "a0.full_action": Tensor of shape (B, action_steps, action_dim),
                    representing the current action chunk for each sample.
                - "a1.full_action": Tensor of shape (B, action_steps, action_dim),
                    representing the next action chunk for each sample.
                - "s0.states": Tensor of shape (B, state_dim),
                    representing the current environment or agent state.
                - "s1.states": Tensor of shape (B, state_dim),
                    representing the next environment or agent state.
                - "s0.images": Tensor of shape (B, n_images, C, H, W),
                    containing current visual observations.
                - "s1.images": Tensor of shape (B, n_images, C, H, W),
                    containing next-step visual observations.
                - "s0.image_masks": Tensor of shape (B, n_images),
                    indicating valid images per sample.
                - "s1.image_masks": Tensor of shape (B, n_images),
                    indicating valid images per sample.
                - "s0.lang_tokens": Tensor of shape (B, max_seq_len),
                    tokenized language instructions.
                - "s1.lang_tokens": Tensor of shape (B, max_seq_len),
                    tokenized language instructions for the next step.
                - "s0.lang_masks": Tensor of shape (B, max_seq_len),
                    attention masks for language tokens.
                - "s1.lang_masks": Tensor of shape (B, max_seq_len),
                    attention masks for language tokens for the next step.
                - "rewards": Tensor of shape (B,),
                    chunk-level scalar rewards aligned to the next step.
                - "response_mask": Tensor of shape (B, action_steps),
                    mask indicating whether each sample has a valid response.
        """

        raise NotImplementedError("Subclasses must implement update_policy method.")

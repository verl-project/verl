# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from typing_extensions import override

from verl.experimental.vla.sac.replay_pool import SACReplayPool
from verl.protocol import DataProto
from verl.utils.device import get_device_id, get_device_name

from .base import BaseSACActor, SupportSACTraining

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_dict_from_prefix(tensordict: TensorDict, prefix: str) -> dict:
    """Extract a sub-dictionary from a TensorDict based on a given prefix.

    Args:
        tensordict: The input TensorDict containing various keys.
        prefix: The prefix string to filter keys.
    Returns:
        A dictionary containing key-value pairs from the TensorDict
        where the keys start with the specified prefix. The prefix is removed
        from the keys in the resulting dictionary.
    """

    result = {}
    prefix_length = len(prefix)
    for key in tensordict.keys():
        if key.startswith(prefix):
            new_key = key[prefix_length:]
            result[new_key] = tensordict[key]
    return result


def merge_nested_dicts_or_tuples(a: dict | tuple, b: dict | tuple) -> dict | tuple:
    """Merge two nested structures (dictionaries or tuples) by concatenating tensors
    along the first dimension.
    """

    if isinstance(a, dict) and isinstance(b, dict):
        merged = {}
        for key in a.keys():
            merged[key] = merge_nested_dicts_or_tuples(a[key], b[key])
        return merged
    elif isinstance(a, tuple) and isinstance(b, tuple):
        merged = []
        for item_a, item_b in zip(a, b, strict=False):
            merged.append(merge_nested_dicts_or_tuples(item_a, item_b))
        return tuple(merged)
    else:
        return torch.cat([a, b], dim=0)


def split_nested_dicts_or_tuples(data: dict | tuple, split_num: int) -> list[dict | tuple]:
    """Split a nested structure (dictionary or tuple) into smaller chunks along the first dimension."""

    if isinstance(data, torch.Tensor):
        split_tensors = torch.chunk(data, split_num, dim=0)
        return list(split_tensors)
    elif isinstance(data, dict):
        split_dicts = [dict() for _ in range(split_num)]
        for key, value in data.items():
            split_values = split_nested_dicts_or_tuples(value, split_num)
            for i in range(split_num):
                split_dicts[i][key] = split_values[i]
        return split_dicts
    elif isinstance(data, tuple):
        split_tuples = [list() for _ in range(split_num)]
        for item in data:
            split_items = split_nested_dicts_or_tuples(item, split_num)
            for i in range(split_num):
                split_tuples[i].append(split_items[i])
        return [tuple(split_tuple) for split_tuple in split_tuples]
    else:
        raise TypeError("Input data must be a torch.Tensor, dict, or tuple.")


class RobDataParallelSACActor(BaseSACActor):
    def __init__(
        self,
        config,
        actor_module: SupportSACTraining,
        actor_optimizer: torch.optim.Optimizer,
        tokenizer=None,
    ):
        super().__init__()
        self.config = config
        self.sac_config = config.sac
        self.device = get_device_name()

        self.actor_optimizer = actor_optimizer
        self.actor_module = actor_module
        self.actor_module.sac_init()
        self.tokenizer = tokenizer

        self.replay_pool = SACReplayPool(capacity=self.config.replay_pool_capacity, sample_device=self.device)
        self.replay_pool.load(self.config.replay_pool_save_dir)

        self._init_alpha()

    def _init_alpha(self):
        """Initialize the alpha optimizer for automatic entropy tuning."""

        self.auto_entropy = self.sac_config.get("auto_entropy", False)

        if self.auto_entropy:
            self.target_entropy = torch.tensor(float(self.sac_config.get("target_entropy", -32.0)), device=self.device)

            # Initialize raw_alpha parameter
            self.alpha_type = self.sac_config.get("alpha_type", "softplus")
            if self.alpha_type == "exp":
                self.raw_alpha = torch.nn.Parameter(
                    np.log(np.exp(self.sac_config.get("initial_alpha", 1))) * torch.ones(1, device=self.device),
                    requires_grad=True,
                )
            elif self.alpha_type == "softplus":
                self.raw_alpha = torch.nn.Parameter(
                    np.log(np.exp(self.sac_config.get("initial_alpha", 0.01)) - 1) * torch.ones(1, device=self.device),
                    requires_grad=True,
                )
            else:
                return NotImplementedError(f"Unsupported alpha_type: {self.alpha_type}")

            # build alpha optimizer and scheduler
            self.alpha_optimizer = torch.optim.Adam([self.raw_alpha], lr=self.sac_config.get("alpha_lr", 3e-4))
            self.alpha_scheduler = torch.optim.lr_scheduler.ConstantLR(self.alpha_optimizer, factor=1.0)

    def _get_alpha(self) -> torch.Tensor:
        if self.auto_entropy:
            if self.alpha_type == "exp":
                return self.raw_alpha.exp()
            elif self.alpha_type == "softplus":
                return torch.nn.functional.softplus(self.raw_alpha)
            else:
                return NotImplementedError(f"Unsupported alpha_type: {self.alpha_type}")
        else:
            return torch.tensor(float(self.sac_config.get("initial_alpha", 0.2)), device=self.device)

    def _calculate_actor_loss(
        self,
        log_probs: torch.Tensor,
        q_values: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate actor loss using the SAC loss function.

        Args:
            log_probs: Tensor of shape (B,) representing the log probabilities of actions.
            q_values: Tensor of shape (B,) representing the Q-values for the actions.
            valid: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).

        Returns:
            Tensor of shape (1,) representing the actor loss.
        """

        alpha = self._get_alpha()
        loss = alpha * log_probs - q_values
        actor_loss = (loss * valid).sum() / (valid.sum().clamp_min(1.0))

        return actor_loss

    def _calculate_alpha_loss(self, log_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Calculate alpha loss for automatic entropy tuning.

        Args:
            log_probs: Tensor of shape (B,) representing the log probabilities of actions.
            valid: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).

        Returns:
            Tensor of shape (1,) representing the alpha loss.
        """

        alpha_loss = -self._get_alpha() * (log_probs.detach() + self.target_entropy)
        alpha_loss = (alpha_loss * valid).sum() / (valid.sum().clamp_min(1.0))
        return alpha_loss

    def _calculate_critic_loss(
        self,
        q_predict: torch.Tensor,
        q_target: torch.Tensor,
        rewards: torch.Tensor,
        valid: torch.Tensor,
        next_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate critic loss using the SAC loss function.

        Args:
            q_predict: Tensor of shape (B, critic_num) representing predicted Q-values.
            q_target: Tensor of shape (B,) representing target Q-values.
            rewards: Tensor of shape (B,) representing rewards.
            valid: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).
            next_log_prob: Tensor of shape (B,) representing log probabilities of next actions.

        Returns:
            Tensor of shape (1,) representing the critic loss.
        """

        gamma = self.sac_config.gamma
        alpha = self._get_alpha()

        with torch.no_grad():
            y = rewards + valid * gamma * (q_target - alpha * next_log_prob)

        y = y.unsqueeze(1).expand_as(q_predict)  # (B, critic_num)
        valid_mask = valid.unsqueeze(1)
        mse = F.mse_loss(q_predict, y, reduction="none")
        per_critic = (mse * valid_mask).sum(dim=0) / valid_mask.sum().clamp_min(1.0)
        critic_loss = per_critic.sum()
        return critic_loss

    def _forward_critic(self, micro_batch: TensorDict) -> torch.Tensor:
        s0 = get_dict_from_prefix(micro_batch, "s0.")
        s1 = get_dict_from_prefix(micro_batch, "s1.")
        a0 = get_dict_from_prefix(micro_batch, "a0.")

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            with torch.no_grad():
                s = merge_nested_dicts_or_tuples(s0, s1)
                state_features = self.actor_module.sac_forward_state_features(s)
                s0_state_features, s1_state_features = split_nested_dicts_or_tuples(state_features, 2)
                a1_actions, log_probs_1 = self.actor_module.sac_forward_actor(s1_state_features)

            q_values_0 = self.actor_module.sac_forward_critic(
                a0,
                s0_state_features,
                use_target_network=False,
                method="cat",
                requires_grad=True,
            )
            q_values_1 = self.actor_module.sac_forward_critic(
                {"full_action": a1_actions},
                s1_state_features,
                use_target_network=True,
                method="min",
                requires_grad=False,
            )

            critic_loss = self._calculate_critic_loss(
                q_predict=q_values_0,
                q_target=q_values_1,
                rewards=micro_batch["rewards"].max(dim=-1).values,
                valid=micro_batch["valid"],
                next_log_prob=log_probs_1,
            )
        return critic_loss

    def _forward_actor(self, micro_batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        micro_batch = micro_batch.to(get_device_id())
        s0 = get_dict_from_prefix(micro_batch, "s0.")

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            s0_state_features = self.actor_module.sac_forward_state_features(s0)
            a0_actions, log_probs_0 = self.actor_module.sac_forward_actor(s0_state_features)
            q_values_0 = self.actor_module.sac_forward_critic(
                {"full_action": a0_actions},
                s0_state_features,
                use_target_network=False,
                method="min",
                requires_grad=False,
            )

            actor_loss = self._calculate_actor_loss(
                log_probs=log_probs_0,
                q_values=q_values_0,
                valid=micro_batch["valid"],
            )
        return actor_loss, log_probs_0

    @override
    def update_policy(self, data: DataProto):
        batch: TensorDict = data.select(
            [
                "a0.full_action",
                "a1.full_action",
                "s0.states",
                "s1.states",
                "s0.images",
                "s1.images",
                "s0.image_masks",
                "s1.image_masks",
                "s0.lang_tokens",
                "s1.lang_tokens",
                "s0.lang_masks",
                "s1.lang_masks",
                "rewards",
                "response_mask",
            ]
        ).batch

        batch = self.replay_pool.insert_and_resample(batch)
        batch["valid"] = batch["response_mask"].any(dim=-1).float()  # (B,)
        micro_batches = batch.split(self.config.ppo_micro_batch_size_per_gpu)
        global_steps = data.meta_info["global_steps"]
        grad_accum_steps = len(micro_batches) * torch.distributed.get_world_size()

        actor_logprobs_list = []
        actor_loss_list, critic_loss_list, alpha_loss_list = [], [], []

        # Training critic
        self.actor_optimizer.zero_grad()
        for batch_idx, micro_batch in enumerate(micro_batches):
            logger.info(f"[{batch_idx + 1}/{len(micro_batches)}] critic micro batch ")

            micro_batch = micro_batch.to(get_device_id())
            raw_critic_loss = self._forward_critic(micro_batch)
            (raw_critic_loss / grad_accum_steps).backward()
            critic_loss_list.append(raw_critic_loss.detach().item())
        critic_grad_norm = self._optimizer_step()

        if global_steps >= self.config.critic_warmup_steps:
            # Training actor
            self.actor_optimizer.zero_grad()
            for batch_idx, micro_batch in enumerate(micro_batches):
                logger.info(f"[{batch_idx + 1}/{len(micro_batches)}] actor micro batch ")

                micro_batch = micro_batch.to(get_device_id())
                raw_actor_loss, log_probs = self._forward_actor(micro_batch)
                (raw_actor_loss / grad_accum_steps).backward()
                actor_loss_list.append(raw_actor_loss.detach().item())
                actor_logprobs_list.append(log_probs.detach())
            actor_grad_norm = self._optimizer_step()

            # Training alpha
            # NOTE: We reuse the log-probabilities computed during the actor forward pass
            # to update the entropy temperature (alpha), instead of re-forwarding
            # the actor after the policy update (saving compute).
            if self.auto_entropy:
                self.alpha_optimizer.zero_grad()
                for micro_batch, log_probs in zip(micro_batches, actor_logprobs_list, strict=False):
                    micro_batch = micro_batch.to(get_device_id())
                    raw_alpha_loss = self._calculate_alpha_loss(log_probs, micro_batch["valid"])
                    (raw_alpha_loss / grad_accum_steps).backward()
                    alpha_loss_list.append(raw_alpha_loss.detach().item())
                torch.distributed.all_reduce(self.raw_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_alpha, max_norm=self.config.grad_clip)
                self.alpha_optimizer.step()
                self.alpha_scheduler.step()

        # Update target networks
        self.actor_module.sac_update_target_network(self.sac_config.tau)

        # Save replay pool
        if global_steps % self.config.replay_pool_save_interval == 0:
            self.replay_pool.save(self.config.replay_pool_save_dir)

        # Log metrics
        metrics = {
            "data/reward_mean": (batch["rewards"].max(dim=-1).values * batch["valid"]).sum().item()
            / batch["valid"].sum().clamp_min(1.0).item(),
            "data/valid_ratio": batch["valid"].float().mean().item(),
            "sac/alpha": self._get_alpha().detach().item(),
            "sac/alpha_lr": self.alpha_optimizer.param_groups[0]["lr"] if self.auto_entropy else 0.0,
            "sac/alpha_loss": sum(alpha_loss_list) / len(alpha_loss_list) if alpha_loss_list else 0.0,
            "sac/alpha_grad_norm": alpha_grad_norm.detach().item()
            if self.auto_entropy and global_steps >= self.config.critic_warmup_steps
            else 0.0,
            "sac/replay_pool_size": len(self.replay_pool),
            "actor/loss": sum(actor_loss_list) / len(actor_loss_list) if actor_loss_list else 0.0,
            "actor/lr": self.actor_optimizer.param_groups[0]["lr"],
            "actor/grad_norm": actor_grad_norm.detach().item()
            if global_steps >= self.config.critic_warmup_steps
            else 0.0,
            "actor/logprob_mean": torch.cat(actor_logprobs_list).mean().detach().item() if actor_logprobs_list else 0.0,
            "critic/loss": sum(critic_loss_list) / len(critic_loss_list) if critic_loss_list else 0.0,
            "critic/grad_norm": critic_grad_norm.detach().item(),
        }

        return metrics

    def _optimizer_step(self) -> torch.Tensor:
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

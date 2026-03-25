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
from typing import Optional

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


def valid_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute the mean of tensor `x` over valid entries indicated by `valid` mask.

    Args:
        x: Tensor of shape (B, ...) containing values to average.
        valid: Tensor of shape (B,) indicating valid entries (1 for valid, 0 for invalid).

    Returns:
        Scalar tensor (mean over valid samples only)
    """
    x = x.squeeze(-1)
    valid_f = valid.float().to(x.device)
    denom = valid_f.sum().clamp_min(1.0)
    return (x * valid_f).sum() / denom


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

        self.replay_pool = SACReplayPool(
            single_pool_capacity=self.config.replay_pool_single_size,
            sample_device=self.device,
        )
        self.replay_pool.load(self.config.replay_pool_save_dir)

        self._init_alpha()
        self._init_critic()

        self.actor_ema_enabled = bool(self.config.get("actor_ema_enabled", True))
        self.actor_ema_decay = float(self.config.get("actor_ema_decay", 0.995))
        self.actor_ema_shadow: dict[str, torch.Tensor] = {}
        self.actor_ema_initialized = False
        self.bc_loss_coef = float(self.sac_config.get("bc_loss_coef", 0.5))

    def _init_critic(self):
        """Initialize the critic optimizer."""

        self.critic_optimizer = torch.optim.Adam(
            self.actor_module.sac_get_critic_parameters(),
            lr=self.config.critic_lr,
            weight_decay=self.config.critic_weight_decay,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ConstantLR(self.critic_optimizer, factor=1.0)

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

    def _init_actor_ema(self):
        if self.actor_ema_initialized:
            return

        self.actor_ema_shadow = {}

        if not self.actor_ema_enabled:
            self.actor_ema_initialized = True
            return

        for name, param in self.actor_module.sac_get_named_actor_parameters():
            self.actor_ema_shadow[name] = param.detach().clone().to(dtype=torch.float32)

        self.actor_ema_initialized = True

    @torch.no_grad()
    def _update_actor_ema(self):
        if not self.actor_ema_enabled:
            return

        one_minus_decay = 1.0 - self.actor_ema_decay
        for name, param in self.actor_module.sac_get_named_actor_parameters():
            shadow = self.actor_ema_shadow[name]
            shadow.mul_(self.actor_ema_decay).add_(param.detach().to(dtype=torch.float32), alpha=one_minus_decay)

    @torch.no_grad()
    def _apply_actor_ema_to_actor_module(self):
        if not self.actor_ema_enabled:
            return

        for name, param in self.actor_module.sac_get_named_actor_parameters():
            shadow = self.actor_ema_shadow[name]
            param.copy_(shadow.to(device=param.device, dtype=param.dtype))

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
        log_probs: Optional[torch.Tensor],
        q_values: torch.Tensor,
        valids: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate actor loss using the SAC loss function.

        Args:
            log_probs: Tensor of shape (B,) representing the log probabilities of actions.
            q_values: Tensor of shape (B,) representing the Q-values for the actions.
            valids: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).

        Returns:
            Tensor of shape (1,) representing the actor loss.
        """

        alpha = self._get_alpha()
        if log_probs is None:
            loss = -q_values
        else:
            loss = alpha * log_probs - q_values
        actor_loss = (loss * valids).sum() / (valids.sum().clamp_min(1.0))

        return actor_loss

    def _calculate_alpha_loss(self, log_probs: Optional[torch.Tensor], valids: torch.Tensor) -> torch.Tensor:
        """Calculate alpha loss for automatic entropy tuning.

        Args:
            log_probs: Tensor of shape (B,) representing the log probabilities of actions.
            valids: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).

        Returns:
            Tensor of shape (1,) representing the alpha loss.
        """

        if log_probs is None:
            return torch.tensor(0.0, device=valids.device)

        alpha_loss = -self._get_alpha() * (log_probs.detach() + self.target_entropy)
        alpha_loss = (alpha_loss * valids).sum() / (valids.sum().clamp_min(1.0))
        return alpha_loss

    def _calculate_critic_loss(
        self,
        q_predict: torch.Tensor,
        q_target: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_log_prob: Optional[torch.Tensor],
        valids: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate critic loss using the SAC loss function.

        Args:
            q_predict: Tensor of shape (B, critic_num) representing predicted Q-values.
            q_target: Tensor of shape (B,) representing target Q-values.
            rewards: Tensor of shape (B,) representing rewards.
            dones: Tensor of shape (B,) representing done flags.
            next_log_prob: Tensor of shape (B,) representing log probabilities of next actions.
            valids: Tensor of shape (B,) indicating valid samples (1 for valid, 0 for invalid).

        Returns:
            Tensor of shape (1,) representing the critic loss.
        """

        gamma = self.sac_config.gamma
        alpha = self._get_alpha()

        with torch.no_grad():
            if next_log_prob is None:
                y = rewards + gamma * (1.0 - dones) * q_target
            else:
                y = rewards + gamma * (1.0 - dones) * (q_target - alpha * next_log_prob)

        y = y.unsqueeze(1).expand_as(q_predict)  # (B, critic_num)
        valid_mask = valids.unsqueeze(1)
        mse = F.mse_loss(q_predict, y, reduction="none")
        per_critic = (mse * valid_mask).sum(dim=0) / valid_mask.sum().clamp_min(1.0)
        critic_loss = per_critic.sum()
        return critic_loss

    def _forward_critic(
        self, micro_batch: TensorDict, resample=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = get_dict_from_prefix(micro_batch, "s0.")
        s1 = get_dict_from_prefix(micro_batch, "s1.")
        a0 = get_dict_from_prefix(micro_batch, "a0.")
        a1 = get_dict_from_prefix(micro_batch, "a1.")

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            with torch.no_grad():
                s = merge_nested_dicts_or_tuples(s0, s1)
                state_features = self.actor_module.sac_forward_state_features(s)
                s0_state_features, s1_state_features = split_nested_dicts_or_tuples(state_features, 2)
                if resample:
                    a1_actions, log_probs_1, _ = self.actor_module.sac_forward_actor(
                        s1_state_features,
                        is_first_micro_batch=False,
                    )
                    a1 = {"full_action": a1_actions}
                else:
                    log_probs_1 = None

            q_values_0 = self.actor_module.sac_forward_critic(
                a0,
                s0_state_features,
                use_target_network=False,
                method="cat",
                requires_grad=True,
            )
            q_values_1 = self.actor_module.sac_forward_critic(
                a1,
                s1_state_features,
                use_target_network=True,
                method="min",
                requires_grad=False,
            )

            critic_loss = self._calculate_critic_loss(
                q_predict=q_values_0,
                q_target=q_values_1,
                rewards=micro_batch["rewards"],
                dones=micro_batch["dones"],
                next_log_prob=log_probs_1,
                valids=micro_batch["valids"],
            )
        return critic_loss, q_values_0, q_values_1

    def _forward_actor(
        self,
        micro_batch: TensorDict,
        is_first_micro_batch: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, dict[str, float]]:
        micro_batch = micro_batch.to(get_device_id())
        s0 = get_dict_from_prefix(micro_batch, "s0.")

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            s0_state_features = self.actor_module.sac_forward_state_features(s0)
            a0_actions, log_probs_0, actor_forward_metrics = self.actor_module.sac_forward_actor(
                s0_state_features,
                is_first_micro_batch=is_first_micro_batch,
            )
            q_values_0 = self.actor_module.sac_forward_critic(
                {"full_action": a0_actions},
                s0_state_features,
                use_target_network=False,
                method="min",
                requires_grad=False,
            )

            sac_loss = self._calculate_actor_loss(
                log_probs=log_probs_0,
                q_values=q_values_0,
                valids=micro_batch["valids"],
            )
            if self.bc_loss_coef > 0:
                bc_loss = self.actor_module.bc_loss(
                    state_features=s0_state_features,
                    actions={"full_action": a0_actions},
                    valids=micro_batch["valids"],
                )
                actor_loss = sac_loss + self.bc_loss_coef * bc_loss
            else:
                actor_loss = sac_loss
        return actor_loss, log_probs_0, q_values_0, actor_forward_metrics

    def _force_set_lr(self, opt: torch.optim.Optimizer, lr: float):
        for pg in opt.param_groups:
            pg["lr"] = lr

    @override
    def update_policy(self, data: DataProto):
        if not self.actor_ema_initialized:
            self._init_actor_ema()

        # self._force_set_lr(self.actor_optimizer, 5e-6)
        # self._force_set_lr(self.critic_optimizer, 1e-4)

        if "empty_batch" not in data.meta_info:
            task_ids = data.batch["task_ids"]
            self.replay_pool.add_batch(
                data.select(
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
                        "dones",
                        "valids",
                        "positive_sample_mask",
                    ]
                ).batch,
                task_ids=task_ids,
            )

        replay_positive_sample_ratio = float(self.sac_config.get("critic_replay_positive_sample_ratio", 0.5))
        critic_batch, critic_replay_sample_info = self.replay_pool.sample_batch(
            self.config.ppo_mini_batch_size,
            positive_sample_ratio=replay_positive_sample_ratio,
            return_sample_info=True,
        )
        micro_batches = critic_batch.split(self.config.ppo_micro_batch_size_per_gpu)
        global_steps = data.meta_info["global_steps"]
        grad_accum_steps = len(micro_batches) * torch.distributed.get_world_size()

        actor_logprobs_list, actor_qvalues_list = [], []
        critic_qvalues_0_list, critic_qvalues_1_list = [], []
        actor_loss_list, critic_loss_list, alpha_loss_list = [], [], []
        actor_forward_metrics: dict[str, float] = {}

        # Training critic
        self.critic_optimizer.zero_grad()
        for batch_idx, micro_batch in enumerate(micro_batches):
            logger.info(f"[{batch_idx + 1}/{len(micro_batches)}] critic micro batch ")

            micro_batch = micro_batch.to(get_device_id())
            raw_critic_loss, q_values_0, q_values_1 = self._forward_critic(micro_batch, resample=True)
            (raw_critic_loss / grad_accum_steps).backward()
            critic_loss_list.append(raw_critic_loss.detach().item())
            critic_qvalues_0_list.append(q_values_0.mean(dim=-1).detach())
            critic_qvalues_1_list.append(q_values_1.detach())
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_module.sac_get_critic_parameters(), max_norm=self.config.grad_clip
        )
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        update_actor = (
            global_steps >= self.config.critic_warmup_steps and global_steps % self.config.actor_update_interval == 0
        )
        if update_actor:
            replay_positive_sample_ratio = float(self.sac_config.get("actor_replay_positive_sample_ratio", 0.5))
            actor_batch, actor_replay_sample_info = self.replay_pool.sample_batch(
                self.config.ppo_mini_batch_size,
                positive_sample_ratio=replay_positive_sample_ratio,
                return_sample_info=True,
            )
            micro_batches = actor_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            # Training actor
            self.actor_optimizer.zero_grad()
            for batch_idx, micro_batch in enumerate(micro_batches):
                logger.info(f"[{batch_idx + 1}/{len(micro_batches)}] actor micro batch ")

                micro_batch = micro_batch.to(get_device_id())
                raw_actor_loss, log_probs, q_values, actor_forward_metrics_mb = self._forward_actor(
                    micro_batch,
                    is_first_micro_batch=(batch_idx == 0),
                )
                (raw_actor_loss / grad_accum_steps).backward()
                actor_loss_list.append(raw_actor_loss.detach().item())
                if log_probs is not None:
                    actor_logprobs_list.append(log_probs.detach())
                actor_qvalues_list.append(q_values.detach())
                actor_forward_metrics.update(actor_forward_metrics_mb)
            actor_grad_norm = self._optimizer_step()
            self._update_actor_ema()
            self._apply_actor_ema_to_actor_module()

            # Training alpha
            # NOTE: We reuse the log-probabilities computed during the actor forward pass
            # to update the entropy temperature (alpha), instead of re-forwarding
            # the actor after the policy update (saving compute).
            if self.auto_entropy and actor_logprobs_list:
                self.alpha_optimizer.zero_grad()
                for micro_batch, log_probs in zip(micro_batches, actor_logprobs_list, strict=False):
                    micro_batch = micro_batch.to(get_device_id())
                    raw_alpha_loss = self._calculate_alpha_loss(log_probs, micro_batch["valids"])
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
        positive_qvalue_mean = (
            torch.cat(critic_qvalues_0_list)[
                (critic_batch["positive_sample_mask"].to(torch.bool) & critic_batch["valids"].to(torch.bool)).to(
                    torch.cat(critic_qvalues_0_list).device
                )
            ]
            .mean()
            .detach()
            .item()
            if critic_qvalues_0_list
            and (critic_batch["positive_sample_mask"].to(torch.bool) & critic_batch["valids"].to(torch.bool)).any()
            else 0.0
        )
        negative_qvalue_mean = (
            torch.cat(critic_qvalues_0_list)[
                (~critic_batch["positive_sample_mask"].to(torch.bool) & critic_batch["valids"].to(torch.bool)).to(
                    torch.cat(critic_qvalues_0_list).device
                )
            ]
            .mean()
            .detach()
            .item()
            if critic_qvalues_0_list
            and (~critic_batch["positive_sample_mask"].to(torch.bool) & critic_batch["valids"].to(torch.bool)).any()
            else 0.0
        )
        metrics = {
            "data/reward_mean": valid_mean(critic_batch["rewards"], critic_batch["valids"]).detach().item(),
            "data/valid_ratio": critic_batch["valids"].float().mean().item(),
            "sac/critic_replay_sampled_ratio": critic_replay_sample_info["actual_positive_sample_ratio"],
            "sac/actor_replay_sampled_ratio": actor_replay_sample_info["actual_positive_sample_ratio"]
            if update_actor
            else 0.0,
            "sac/replay_pool_positive_size": critic_replay_sample_info["positive_size"],
            "sac/replay_pool_negative_size": critic_replay_sample_info["negative_size"],
            "sac/replay_task_count": critic_replay_sample_info["task_count"],
            "sac/alpha": self._get_alpha().detach().item(),
            "sac/actor_ema_enabled": float(self.actor_ema_enabled),
            "sac/actor_ema_decay": self.actor_ema_decay,
            "sac/replay_pool_size": len(self.replay_pool),
            "critic/loss": sum(critic_loss_list) / len(critic_loss_list) if critic_loss_list else 0.0,
            "critic/lr": self.critic_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": critic_grad_norm.detach().item(),
            "critic/qvalue0_mean": (
                valid_mean(torch.cat(critic_qvalues_0_list), critic_batch["valids"]).detach().item()
                if critic_qvalues_0_list
                else 0.0
            ),
            "critic/qvalue1_mean": (
                valid_mean(torch.cat(critic_qvalues_1_list), critic_batch["valids"]).detach().item()
                if critic_qvalues_1_list
                else 0.0
            ),
            "critic/positive_qvalue_mean": positive_qvalue_mean,
            "critic/negative_qvalue_mean": negative_qvalue_mean,
            "critic/diff_pos_neg_qvalue_mean": positive_qvalue_mean - negative_qvalue_mean,
        }
        if update_actor:
            metrics.update(
                {
                    "actor/loss": sum(actor_loss_list) / len(actor_loss_list),
                    "actor/lr": self.actor_optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm.detach().item(),
                    "actor/logprob_mean": (
                        valid_mean(torch.cat(actor_logprobs_list), actor_batch["valids"]).detach().item()
                        if actor_logprobs_list
                        else 0.0
                    ),
                    "actor/qvalue_mean": valid_mean(torch.cat(actor_qvalues_list), actor_batch["valids"])
                    .detach()
                    .item(),
                    "sac/alpha_lr": self.alpha_optimizer.param_groups[0]["lr"]
                    if self.auto_entropy and actor_logprobs_list
                    else 0.0,
                    "sac/alpha_loss": sum(alpha_loss_list) / len(alpha_loss_list)
                    if self.auto_entropy and alpha_loss_list
                    else 0.0,
                    "sac/alpha_grad_norm": alpha_grad_norm.detach().item()
                    if self.auto_entropy and actor_logprobs_list
                    else 0.0,
                }
            )
            metrics.update({f"actor/{k}": v for k, v in actor_forward_metrics.items()})

        return metrics

    def _optimizer_step(self) -> torch.Tensor:
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

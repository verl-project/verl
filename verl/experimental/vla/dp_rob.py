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

import torch
from tensordict.base import TensorDictBase
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.device import get_device_id, get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor import BasePPOActor

logger = logging.getLogger(__name__)

__all__ = ["RobDataParallelPPOActor"]


class RobDataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        logger.info(f"Actor use_remove_padding={self.use_remove_padding}")
        logger.info(f"PRM use dynamic bsz={self.config.get('use_dynamic_bsz', False)}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False  # self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def process_tensor(self, tensor, pad_id):
        mask = tensor != pad_id
        if not torch.all(mask == mask[0:1], dim=1).all():
            raise ValueError("Padding error!")
        base_mask = mask[0]
        valid_len = base_mask.sum().item()
        return tensor[:, base_mask], valid_len

    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,),
            traj_len:
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask

    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        """
        Args:
            log_probs: (batch_size, 7*8)
            entropy:   (batch_size, 7*8)
            # mask:      (batch_size, 8)
            mask:      (batch_size, 7*8)
        Returns:
            log_probs_masked:
            entropy_masked:
        """

        mask = mask.to(log_probs.device)
        log_probs_masked = torch.where(mask, log_probs, torch.zeros_like(log_probs, requires_grad=False))
        entropy_masked = torch.where(mask, entropy, torch.zeros_like(entropy, requires_grad=False))
        return log_probs_masked, entropy_masked

    def _forward_micro_batch(self, micro_batch, temperature) -> tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:

        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]

            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

            logits = self.actor_module(
                input_ids=input_ids_unpad,
                attention_mask=attention_mask_unpad,
                pixel_values=pixel_values,
            )  # prevent model thinks we are generating

            assert self.actor_module.vocab_size == 32000
            start_index = self.actor_module.vocab_size - 256
            logits = logits[..., -256 - 64 : -64]  # Shape: [batch_size, seq_len, 256]
            responses = responses - start_index
            # assert (0<=responses<=255).all()

            logits = logits.div(temperature)

            log_probs = logprobs_from_logits(logits, responses.to(logits.device))
            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            # assert len(log_probs.shape) == 2 and len(entropy.shape) == 2

            # TODO(caiyunke.astra): check here

            mask = micro_batch["response_mask"]
            log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)

            return entropy, log_probs

    def _forward_micro_batch_update(
        self, input_ids, attention_mask, pixel_values, responses, temperature
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

            logits = self.actor_module(
                input_ids=input_ids_unpad,
                attention_mask=attention_mask_unpad,
                pixel_values=pixel_values,
            )

            assert logits.requires_grad

            assert self.actor_module.vocab_size == 32000
            start_index = self.actor_module.vocab_size - 256
            logits = logits[..., -256 - 64 : -64]  # Shape: [batch_size, seq_len, 256]
            responses = responses - start_index

            logits = logits.div(temperature)

            log_probs = logprobs_from_logits(logits, responses)
            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]  # 256
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error # 1
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]  # trues
        self.pad_token_id = data.meta_info["pad_token_id"]

        select_keys = ["responses", "input_ids", "attention_mask", "pixel_values", "response_mask"]
        data = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return {"log_probs": log_probs, "entropys": entropys}

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "old_log_probs",
            "advantages",
        ]
        batch = data.select(batch_keys=select_keys).batch
        self.pad_token_id = data.meta_info["pad_token_id"]
        # TODO(caiyunke.astra): check here
        # assert self.config.ppo_micro_batch_size_per_gpu == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, mini_batch in enumerate(mini_batches):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for _, micro_batch in enumerate[DataProto | TensorDictBase](micro_batches):
                micro_batch = micro_batch.to(get_device_id())  # actor device is cpu when using offload
                responses = micro_batch["responses"]

                response_mask = micro_batch["response_mask"]  # (batch_size, traj_len)

                old_log_prob = micro_batch["old_log_probs"]
                advantages = micro_batch["advantages"]

                # clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low

                input_ids = micro_batch["input_ids"]
                attention_mask = micro_batch["attention_mask"]
                pixel_values = micro_batch["pixel_values"]
                responses = micro_batch["responses"]

                loss_info = {
                    "actor/pg_loss": 0,
                    "actor/pg_clipfrac": 0,
                    "actor/ppo_kl": 0,
                    "actor/pg_clipfrac_lower": 0,
                }

                _, log_prob = self._forward_micro_batch_update(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    responses=responses,
                    temperature=temperature,
                )

                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    cliprange_high=clip_ratio_high,
                    cliprange_low=clip_ratio_low,
                )
                loss = pg_loss / self.gradient_accumulation

                loss.backward()

                loss_info["actor/pg_loss"] = loss_info["actor/pg_loss"] + pg_loss.detach().item()
                loss_info["actor/pg_clipfrac"] = loss_info["actor/pg_clipfrac"] + pg_clipfrac.detach().item()
                loss_info["actor/ppo_kl"] = loss_info["actor/ppo_kl"] + ppo_kl.detach().item()
                loss_info["actor/pg_clipfrac_lower"] = (
                    loss_info["actor/pg_clipfrac_lower"] + pg_clipfrac_lower.detach().item()
                )
                append_to_dict(metrics, loss_info)

            grad_norm = self._optimizer_step()
            mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

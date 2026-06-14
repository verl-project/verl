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
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import logprobs_from_logits

from ..base import BaseRollout

__all__ = ["NaiveRollout"]


class NaiveRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        """A naive rollout. It requires the module to be compatible with huggingface APIs. That is:
        The module should define __call__ to receive input_ids, attention_mask and position_ids.
        It outputs a structure that contains logits field.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
        """
        super().__init__()
        self.config = config
        self.module = module

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()

        prev_attention_mask = torch.ones(size=(batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        logits_lst = []
        for _ in range(self.config.response_length):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            idx_cond = idx
            # forward the model to get the logits for the index in the sequence
            # we use huggingface APIs here
            output = self.module(input_ids=idx_cond, attention_mask=attention_mask, position_ids=position_ids)
            logits = output.logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.config.temperature  # (bs, vocab_size)
            # optionally crop the logits to only the top k options
            if self.config.top_k is not None:
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            if self.config.do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

            for token_id in eos_token_id:
                prev_attention_mask = torch.logical_and(idx_next != token_id, prev_attention_mask.bool())
            prev_attention_mask.to(attention_mask.dtype)

            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            logits_lst.append(logits)

        logits = torch.stack(logits_lst, dim=1)  # (bs, response_length, vocab_size)
        prompts = idx[:, :prompt_length]  # (bs, prompt_length)
        response = idx[:, prompt_length:]  # (bs, response_length)
        log_probs = logprobs_from_logits(logits=logits, labels=response)
        batch = TensorDict(
            {
                "input_ids": prompts,
                "responses": response,
                "sequences": idx,
                "old_log_probs": log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        self.module.train()

        return DataProto(batch=batch)

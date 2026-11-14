# Copyright 2026 Individual Contributor: LOGO127
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

"""Optimizer-minibatch token selection for the FSDP Clip-Cov actor."""

import torch
import torch.distributed as dist

from verl.trainer.ppo.core_algos import compute_clip_cov_mask
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.workers.utils.padding import no_padding_2_padding, response_to_nested


def compute_global_clip_cov_mask(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    clip_cov_ratio,
    clip_cov_lb,
    clip_cov_ub,
    cliprange_low,
    cliprange_high,
    dp_group=None,
):
    """Select across DP shards, then return only this rank's response-shaped mask."""
    valid = response_mask.to(bool)
    local = (
        old_log_prob[valid].detach().cpu(),
        log_prob[valid].detach().cpu(),
        advantages[valid].detach().cpu(),
    )
    shards = [local]
    rank = 0
    if dp_group is not None:
        shards = [None] * dist.get_world_size(dp_group)
        dist.all_gather_object(shards, local, group=dp_group)
        rank = dist.get_rank(dp_group)
    old_all = torch.cat([shard[0] for shard in shards])
    log_prob_all = torch.cat([shard[1] for shard in shards])
    advantages_all = torch.cat([shard[2] for shard in shards])
    selected = compute_clip_cov_mask(
        old_all,
        log_prob_all,
        advantages_all,
        torch.ones_like(advantages_all, dtype=torch.bool),
        clip_cov_ratio,
        clip_cov_lb,
        clip_cov_ub,
        cliprange_low,
        cliprange_high,
    )
    offset = sum(shard[0].numel() for shard in shards[:rank])
    result = torch.zeros_like(valid)
    result[valid] = selected[offset : offset + local[0].numel()].to(result.device)
    return result


def prepare_clip_cov_batch(engine, data, config):
    """Attach selection before microbatching, using the current actor parameters.

    The caller has entered train mode and supplied the training microbatch settings.
    Restore RNG states so the additional no-grad forward does not consume the
    training pass's random stream. Recompute for every optimizer update.
    """
    if tu.get(data, "distillation_only", default=False):
        return
    prepass_data = data.copy()
    # The selection pass only needs policy log probabilities, not teacher losses.
    tu.assign_non_tensor(prepass_data, distillation_use_topk=False)
    device_name = get_device_name()
    devices = [] if device_name == "cpu" else [get_device_id()]
    with torch.random.fork_rng(devices=devices, device_type=device_name):
        output = engine.infer_batch(prepass_data, loss_function=None)
    log_prob = no_padding_2_padding(output["model_output"]["log_probs"], data)
    padded = data.select("advantages", "response_mask", "old_log_probs").to_padded_tensor()
    policy_loss = config.policy_loss
    clip_cov_ratio = policy_loss.clip_cov_ratio if policy_loss.clip_cov_ratio is not None else 0.0002
    clip_cov_ub = policy_loss.clip_cov_ub if policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = policy_loss.clip_cov_lb if policy_loss.clip_cov_lb is not None else 1.0
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    mask = compute_global_clip_cov_mask(
        padded["old_log_probs"],
        log_prob,
        padded["advantages"],
        padded["response_mask"],
        clip_cov_ratio,
        clip_cov_lb,
        clip_cov_ub,
        cliprange_low,
        cliprange_high,
        dp_group=engine.get_data_parallel_group(),
    )
    if data["response_mask"].is_nested:
        mask = response_to_nested(mask, data["response_mask"])
    data["clip_cov_mask"] = mask

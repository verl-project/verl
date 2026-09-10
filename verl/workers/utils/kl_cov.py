# Copyright 2026 Individual Contributor: gss10282025
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

"""Optimizer-minibatch token selection for the FSDP KL-Cov actor."""

import torch
import torch.distributed as dist

from verl.trainer.ppo.core_algos import compute_kl_cov_mask
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.workers.utils.padding import no_padding_2_padding, response_to_nested


def compute_global_kl_cov_mask(advantages, log_prob, response_mask, kl_cov_ratio, dp_group=None):
    """Select across DP shards, then return only this rank's response-shaped mask."""
    valid = response_mask.to(bool)
    local = (advantages[valid].detach().cpu(), log_prob[valid].detach().cpu())
    shards = [local]
    rank = 0
    if dp_group is not None:
        shards = [None] * dist.get_world_size(dp_group)
        dist.all_gather_object(shards, local, group=dp_group)
        rank = dist.get_rank(dp_group)
    advantages_all = torch.cat([shard[0] for shard in shards])
    log_prob_all = torch.cat([shard[1] for shard in shards])
    selected = compute_kl_cov_mask(
        advantages_all, log_prob_all, torch.ones_like(advantages_all, dtype=torch.bool), kl_cov_ratio
    )
    offset = sum(shard[0].numel() for shard in shards[:rank])
    result = torch.zeros_like(valid)
    result[valid] = selected[offset : offset + local[0].numel()].to(result.device)
    return result


def prepare_kl_cov_batch(engine, data, config):
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
    padded = data.select("advantages", "response_mask").to_padded_tensor()
    kl_cov_ratio = config.policy_loss.kl_cov_ratio
    mask = compute_global_kl_cov_mask(
        padded["advantages"],
        log_prob,
        padded["response_mask"],
        0.0002 if kl_cov_ratio is None else kl_cov_ratio,
        dp_group=engine.get_data_parallel_group(),
    )
    if data["response_mask"].is_nested:
        mask = response_to_nested(mask, data["response_mask"])
    data["kl_cov_mask"] = mask

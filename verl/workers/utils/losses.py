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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_name
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding
from verl.workers.utils.tpu_static_packing import (
    flatten_tpu_loss_mask,
    select_and_pad_tpu_data,
    unpack_tpu_packed_data,
)


class DummyConfig:
    """Fallback configuration object when config is passed as None during SFT loss evaluation."""

    def __init__(self):
        self.global_batch_info = {}
        self.loss_scale_factor = None
        self.loss_agg_mode = "token-mean"


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute Supervised Fine-Tuning (SFT) loss for actor model.

    Supports both padded (RIGHT) and packed/nested sequence representations,
    including static sequence packing for Google TPU v6e.

    Args:
        config (ActorConfig): Configuration object for actor training.
        model_output (dict): Model forward outputs containing 'log_probs'.
        data (TensorDict): Data batch containing token IDs, masks, and metadata.
        dp_group: Data parallel process group (optional).

    Returns:
        tuple[torch.Tensor, dict]: Computed SFT loss and empty metrics dict.
    """
    is_tpu = get_device_name() == "tpu"
    if is_tpu:
        # TPU: Unpack sequence data if TPU static sequence packing is enabled
        data = unpack_tpu_packed_data(data)

    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)

    dp_size = tu.get_non_tensor_data(data=data, key="dp_size", default=1)
    batch_num_tokens = tu.get_non_tensor_data(data=data, key="batch_num_tokens", default=None)

    log_prob = model_output["log_probs"]

    if pad_mode in (DatasetPadMode.NO_PADDING, DatasetPadMode.TPU_BINNED_PACK, "tpu_binned_pack"):
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        if is_tpu and pad_mode in (DatasetPadMode.TPU_BINNED_PACK, "tpu_binned_pack"):
            # TPU: flatten loss mask to align with TPU binned static packed log_probs
            log_prob_flatten = log_prob.values()
            loss_mask = flatten_tpu_loss_mask(loss_mask, data, log_prob_flatten)

        log_prob = no_padding_2_padding(log_prob, data)

        # construct global batch info
        if config is None:
            config = DummyConfig()

        config.global_batch_info["dp_size"] = dp_size
        config.global_batch_info["batch_num_tokens"] = batch_num_tokens
        config.global_batch_info["global_batch_size"] = data["global_batch_size"]
        config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

        mask_key = "response_mask" if "response_mask" in data.keys() else "loss_mask"
        if is_tpu:
            # TPU: select and pad tensors with fixed target shape to avoid dynamic shape recompilation
            padded_dict = select_and_pad_tpu_data(data, mask_key, target_tensor=log_prob)
            response_mask = padded_dict[mask_key].to(bool)
        else:
            response_mask = data[mask_key].to(bool)

        # Bypass zero-sequence length empty tensor backward pass issues
        if log_prob.shape[1] == 0:
            log_prob = torch.zeros(
                (log_prob.shape[0], 1), device=log_prob.device, dtype=log_prob.dtype, requires_grad=True
            )
            response_mask = torch.zeros((response_mask.shape[0], 1), device=response_mask.device, dtype=torch.bool)

        loss = agg_loss(
            loss_mat=-log_prob, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )
    elif pad_mode == DatasetPadMode.RIGHT:
        if "response_mask" in data.keys():
            mask = data["response_mask"].to(bool)
            mask[:, -1] = False
        else:
            mask = data["loss_mask"].to(bool)
            mask = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], dim=1)
        loss = -masked_sum(log_prob, mask) / batch_num_tokens * dp_size
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute PPO policy gradient loss, entropy bonus, and KL divergence penalty.

    Args:
        config (ActorConfig): Configuration object for actor training.
        model_output (dict): Model forward outputs containing 'log_probs' and optional 'entropy'.
        data (TensorDict): Data batch containing old log probs, advantages, masks, and metadata.
        dp_group: Data parallel process group (optional).

    Returns:
        tuple[torch.Tensor, dict]: Total policy loss and metric dictionary containing policy metrics,
            entropy loss, and KL loss tracking.
    """
    is_tpu = get_device_name() == "tpu"
    if is_tpu:
        # TPU: Unpack sequence data if TPU static sequence packing is enabled
        data = unpack_tpu_packed_data(data)

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    dp_size = tu.get_non_tensor_data(data=data, key="dp_size", default=1)
    batch_num_tokens = tu.get_non_tensor_data(data=data, key="batch_num_tokens", default=None)
    config.global_batch_info["dp_size"] = dp_size
    config.global_batch_info["batch_num_tokens"] = batch_num_tokens
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")

    if is_tpu:
        # TPU: select and pad tensors with fixed target shape to avoid dynamic shape recompilation
        data = select_and_pad_tpu_data(data, *fields, target_tensor=log_prob)
    else:
        data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = Metric(value=config.kl_loss_coef, aggregation=metric_aggregation)

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """Compute critic value function loss with optional value clipping.

    Args:
        config (CriticConfig): Configuration object for critic training.
        model_output (dict): Model forward outputs containing predicted 'values'.
        data (TensorDict): Data batch containing target returns, old values, and masks.
        dp_group: Data parallel process group (optional).

    Returns:
        tuple[torch.Tensor, dict]: Value function loss and metric dictionary containing vf_loss,
            vf_clipfrac, and mean predicted values.
    """
    is_tpu = get_device_name() == "tpu"
    if is_tpu:
        # TPU: Unpack sequence data if TPU static sequence packing is enabled
        data = unpack_tpu_packed_data(data)

    vpreds = no_padding_2_padding(model_output["values"], data)

    # Normalize the value loss over the global mini-batch (dp_size / batch_num_tokens /
    # global_batch_size) instead of the local micro-batch, so the accumulated critic gradient is
    # invariant to how the mini-batch is split into micro-batches (as the actor's ppo_loss does).
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

    # When the loss is normalized over the global batch, each micro-batch contributes a partial sum,
    # so the loss metric must be aggregated with SUM to reflect the global-batch mean.
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    if is_tpu:
        # TPU: select and pad tensors with fixed target shape to avoid dynamic shape recompilation
        data = select_and_pad_tpu_data(data, "values", "returns", "response_mask")
    else:
        # select fields and convert to padded tensor
        data = data.select("values", "returns", "response_mask").to_padded_tensor()

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=config.loss_scale_factor,
    )

    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=metric_aggregation),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
    }

    return vf_loss, metrics

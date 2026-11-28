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

from verl.trainer.dpo.dpo_losses import get_dpo_loss_fn
from verl.trainer.dpo.utils import sum_logps_per_sequence
from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)
    sc_outputs = {
        key: no_padding_2_padding(model_output[key], data)
        for key in (
            "sc_correction",
            "sc_sampler_head_mass",
            "sc_train_head_mass",
            "sc_support_aligned",
            "sc_sampler_tail_mass",
            "sc_support_unknown",
        )
        if key in model_output
    }

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
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
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    policy_loss_kwargs = {"sc_correction": sc_outputs["sc_correction"]} if sc_outputs else {}
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        **policy_loss_kwargs,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    # Response tokens whose update-forward log-prob is not bitwise equal to old_log_probs. The count is
    # summed across microbatches; it is scaled by dp_size so the worker's DP mean yields the global count.
    old_logprobs_mismatch = ((log_prob.float() != old_log_prob.float()) & response_mask).sum()
    metrics["actor/old_logprobs_mismatch_count"] = Metric(
        value=old_logprobs_mismatch * config.global_batch_info["dp_size"], aggregation=AggregationType.SUM
    )
    if sc_outputs:
        for key in (
            "sc_sampler_head_mass",
            "sc_train_head_mass",
            "sc_support_aligned",
            "sc_sampler_tail_mass",
            "sc_support_unknown",
        ):
            if key not in sc_outputs:
                continue
            if key in ("sc_support_aligned", "sc_sampler_tail_mass", "sc_support_unknown"):
                # Sum globally normalized token contributions across microbatches; the
                # worker's DP mean then yields a token-weighted fraction, even for uneven rows.
                global_tokens = config.global_batch_info["batch_num_tokens"]
                value = agg_loss(
                    sc_outputs[key],
                    response_mask,
                    "token-mean",
                    dp_size=config.global_batch_info["dp_size"],
                    batch_num_tokens=global_tokens,
                )
                aggregation = AggregationType.SUM if global_tokens is not None else AggregationType.MEAN
            else:
                value = masked_mean(sc_outputs[key], response_mask)
                aggregation = AggregationType.MEAN
            metrics[f"actor/{key}"] = Metric(value=value, aggregation=aggregation)
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
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

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


def dpo_loss(config, model_output, data: TensorDict, dp_group=None):
    """DPO preference loss for ``TrainingWorker.set_loss_fn``.

    The micro-batch holds 2B sequences interleaved as [chosen_0, rejected_0, chosen_1, rejected_1, ...].
    Each sequence's response log-prob is summed, the selected loss variant is applied to the B pairs,
    and the result is scaled so that summing over micro-batches and averaging over data-parallel ranks
    gives the mean loss over the global batch of pairs, the same convention as ``agg_loss``.

    Args:
        config: ``verl.trainer.dpo.dpo_losses.DPOLossConfig`` (loss_type, beta, label_smoothing, reference_free).
        model_output: dict with "log_probs", a NestedTensor (2B, j). It is consumed and removed, because
            its 2B rows do not match the B-row indices the engine uses to restore dynamic batches.
        data: micro-batch TensorDict with "loss_mask" (NestedTensor, 2B rows), "ref_logps" ((2B,) summed
            reference log-probs, not needed when reference_free), and non-tensor "dp_size" and
            "global_batch_size" (the number of pairs in the global batch, across all data-parallel ranks).
        dp_group: unused.

    Returns:
        (loss, metrics). Every metric is pre-scaled like the loss, so its reported value is the
        sum over micro-batches, averaged over data-parallel ranks.
    """
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    if pad_mode != DatasetPadMode.NO_PADDING:
        raise NotImplementedError(f"dpo_loss requires pad_mode=no_padding, got {pad_mode}")
    dp_size = data["dp_size"]
    global_num_pairs = tu.get_non_tensor_data(data=data, key="global_batch_size", default=None)
    if global_num_pairs is None:
        raise ValueError("dpo_loss needs global_batch_size, the number of pairs in the global batch")

    log_probs = model_output.pop("log_probs")
    per_seq_logps = sum_logps_per_sequence(log_probs, data["loss_mask"])
    if per_seq_logps.shape[0] % 2 != 0:
        raise ValueError(f"DPO micro-batch must hold whole (chosen, rejected) pairs, got {per_seq_logps.shape[0]} rows")
    chosen_logps, rejected_logps = per_seq_logps[0::2], per_seq_logps[1::2]
    num_pairs = chosen_logps.shape[0]

    if config.reference_free and "ref_logps" not in data.keys():
        ref_logps = torch.zeros_like(per_seq_logps).detach()
    else:
        ref_logps = data["ref_logps"].to(per_seq_logps)
    ref_chosen_logps, ref_rejected_logps = ref_logps[0::2], ref_logps[1::2]

    loss_fn = get_dpo_loss_fn(config.loss_type)
    preference_loss, chosen_rewards, rejected_rewards = loss_fn(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=config.beta,
        label_smoothing=config.label_smoothing,
        reference_free=config.reference_free,
    )

    # Weight this micro-batch's mean by its share of the global pairs, then multiply by dp_size to undo the
    # average over data-parallel ranks (the reported loss and FSDP's gradient reduction both average).
    scale = num_pairs / global_num_pairs * dp_size
    loss = preference_loss * scale

    def _scaled(value: torch.Tensor) -> float:
        return value.detach().float().mean().item() * scale

    metrics = {
        "dpo/loss": preference_loss.detach().item() * scale,
        "dpo/accuracy": _scaled((chosen_rewards > rejected_rewards).float()),
        "dpo/chosen_reward": _scaled(chosen_rewards),
        "dpo/rejected_reward": _scaled(rejected_rewards),
        "dpo/reward_margin": _scaled(chosen_rewards - rejected_rewards),
        "dpo/chosen_logps": _scaled(chosen_logps),
        "dpo/rejected_logps": _scaled(rejected_logps),
    }
    return loss, metrics

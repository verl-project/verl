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

import hashlib
import os

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding

_ROLLOUT_CORR_DEBUG_COUNT = 0


def _rollout_corr_debug_limit() -> int:
    try:
        return max(0, int(os.getenv("VERL_ROLLOUT_CORR_DEBUG_LIMIT", "10")))
    except ValueError:
        return 5


def _debug_tensor_digest(value, row_idx: int | None = None, max_items: int = 2048) -> dict | None:
    if not isinstance(value, torch.Tensor):
        return None
    try:
        tensor = value[row_idx] if row_idx is not None else value
        tensor = tensor.detach()
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        flat = tensor.reshape(-1)
        numel = int(flat.numel())
        if numel == 0:
            return {"shape": shape, "dtype": dtype, "numel": 0, "sha1": "empty"}
        if numel > max_items:
            head = max_items // 2
            tail = max_items - head
            sample = torch.cat([flat[:head], flat[-tail:]])
        else:
            sample = flat
        if sample.dtype in (torch.bfloat16, torch.float16):
            sample = sample.float()
        sample_cpu = sample.contiguous().cpu()
        sha1 = hashlib.sha1(sample_cpu.numpy().tobytes()).hexdigest()[:16]
        out = {"shape": shape, "dtype": dtype, "numel": numel, "sampled": int(sample_cpu.numel()), "sha1": sha1}
        if numel <= 16 and tensor.dim() <= 2:
            out["values"] = tensor.detach().cpu().tolist()
        return out
    except Exception as exc:
        return {"error": type(exc).__name__}


def _debug_unwrap_non_tensor(value):
    if isinstance(value, NonTensorData):
        return value.data
    return value


def _debug_non_tensor_item(data: TensorDict, key: str, row_idx: int):
    try:
        if key not in data.keys():
            return None
        value = data.get(key)
        value = _debug_unwrap_non_tensor(value)
        if isinstance(value, NonTensorStack):
            return _debug_unwrap_non_tensor(value[row_idx])
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.detach().cpu().item()
            return (
                value[row_idx].detach().cpu().item()
                if value[row_idx].ndim == 0
                else value[row_idx].detach().cpu().tolist()
            )
        if hasattr(value, "shape") and hasattr(value, "__getitem__") and len(value) > row_idx:
            return value[row_idx]
        if isinstance(value, list | tuple) and len(value) > row_idx:
            return value[row_idx]
        return value
    except Exception as exc:
        return f"<{key}_failed:{type(exc).__name__}>"


def _debug_shorten(value, max_len: int = 180):
    try:
        if isinstance(value, dict):
            return {str(k): _debug_shorten(v, max_len=max_len) for k, v in value.items()}
        if isinstance(value, list | tuple):
            shown = [_debug_shorten(v, max_len=max_len) for v in list(value)[:8]]
            if len(value) > 8:
                shown.append(f"...(+{len(value) - 8})")
            return shown
        text = str(value)
        return text if len(text) <= max_len else text[:max_len] + "...<truncated>"
    except Exception as exc:
        return f"<repr_failed:{type(exc).__name__}>"


def _debug_collect_row_metadata(data: TensorDict) -> list[dict]:
    keys = (
        "debug_row_id",
        "__ref_debug_row_id",
        "uid",
        "request_id",
        "trajectory_role",
        "turn_number",
        "rollout_group_id",
        "data_source",
    )
    try:
        n_rows = int(data.shape[0])
    except Exception:
        return []
    rows = []
    for row_idx in range(n_rows):
        row = {}
        for key in keys:
            value = _debug_non_tensor_item(data, key, row_idx)
            if value is not None:
                row[key] = _debug_shorten(value)
        rows.append(row)
    return rows


def _debug_row_inputs(data: TensorDict, row_idx: int) -> dict:
    out = {}
    for key in ("input_ids", "prompts", "responses", "attention_mask", "position_ids", "ref_log_prob", "old_log_probs"):
        if key in data.keys():
            digest = _debug_tensor_digest(data[key], row_idx=row_idx)
            if digest is not None:
                out[f"{key}_digest"] = digest
    if "attention_mask" in data.keys() and isinstance(data["attention_mask"], torch.Tensor):
        try:
            out["attention_valid_len"] = int(data["attention_mask"][row_idx].detach().cpu().sum().item())
        except Exception:
            pass
    if "response_mask" in data.keys() and isinstance(data["response_mask"], torch.Tensor):
        try:
            out["response_valid_len"] = int(data["response_mask"][row_idx].detach().cpu().sum().item())
        except Exception:
            pass
    return out


def _debug_rollout_corr_logprob_alignment(
    data: TensorDict,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_mode: str,
    row_metadata: list[dict] | None = None,
) -> None:
    global _ROLLOUT_CORR_DEBUG_COUNT

    if loss_mode != "bypass_mode":
        return
    debug_limit = _rollout_corr_debug_limit()
    if _ROLLOUT_CORR_DEBUG_COUNT >= debug_limit:
        return
    if "responses" not in data or not response_mask.any():
        return

    with torch.no_grad():
        valid = response_mask.bool()
        current_valid = log_prob[valid].detach().float().cpu()
        rollout_valid = old_log_prob[valid].detach().float().cpu()
        diff_valid = (rollout_valid - current_valid).float()
        ref_log_prob = data.get("ref_log_prob", None)
        ref_valid = ref_log_prob[valid].detach().float().cpu() if ref_log_prob is not None else None
        actor_ref_diff = (current_valid - ref_valid).float() if ref_valid is not None else None
        responses = data["responses"].detach().cpu()

        seq_current = masked_mean(log_prob.detach().float(), response_mask, axis=-1).cpu()
        seq_rollout = masked_mean(old_log_prob.detach().float(), response_mask, axis=-1).cpu()
        seq_diff = seq_rollout - seq_current
        top_k = min(5, seq_diff.numel())
        top_values, top_indices = torch.topk(seq_diff, k=top_k) if top_k > 0 else ([], [])

        rows = []
        candidate_rows = list(range(min(3, response_mask.shape[0])))
        candidate_rows.extend(int(idx.item()) for idx in top_indices)
        seen_rows = set()
        for row_idx in candidate_rows:
            if row_idx in seen_rows or row_idx >= response_mask.shape[0]:
                continue
            seen_rows.add(row_idx)
            valid_cols = torch.nonzero(response_mask[row_idx].detach().cpu().bool(), as_tuple=False).flatten()
            if valid_cols.numel() == 0:
                continue
            cols = valid_cols[:12]
            rows.append(
                {
                    "row": int(row_idx),
                    "valid_tokens": int(valid_cols.numel()),
                    "seq_rollout_minus_current": float(seq_diff[row_idx].item()),
                    "metadata": row_metadata[row_idx] if row_metadata and row_idx < len(row_metadata) else {},
                    "input_digests": _debug_row_inputs(data, row_idx),
                    "tokens": [
                        {
                            "pos": int(col.item()),
                            "token_id": int(responses[row_idx, col].item()),
                            "current_logprob": float(log_prob[row_idx, col].detach().float().cpu().item()),
                            "rollout_logprob": float(old_log_prob[row_idx, col].detach().float().cpu().item()),
                            "rollout_minus_current": float(
                                (old_log_prob[row_idx, col] - log_prob[row_idx, col]).detach().float().cpu().item()
                            ),
                            **(
                                {
                                    "ref_logprob": float(ref_log_prob[row_idx, col].detach().float().cpu().item()),
                                    "current_minus_ref": float(
                                        (log_prob[row_idx, col] - ref_log_prob[row_idx, col])
                                        .detach()
                                        .float()
                                        .cpu()
                                        .item()
                                    ),
                                    "rollout_minus_ref": float(
                                        (old_log_prob[row_idx, col] - ref_log_prob[row_idx, col])
                                        .detach()
                                        .float()
                                        .cpu()
                                        .item()
                                    ),
                                }
                                if ref_log_prob is not None
                                else {}
                            ),
                        }
                        for col in cols
                    ],
                }
            )

        top_sequences = [
            {
                "row": int(idx.item()),
                "seq_rollout_minus_current": float(val.item()),
                "valid_tokens": int(response_mask[idx].sum().detach().cpu().item()),
                "metadata": row_metadata[int(idx.item())]
                if row_metadata and int(idx.item()) < len(row_metadata)
                else {},
            }
            for val, idx in zip(top_values, top_indices, strict=False)
        ]
        ref_stats = (
            f"ref_mean={ref_valid.mean().item():.6f} "
            f"ref_min={ref_valid.min().item():.6f} "
            f"ref_max={ref_valid.max().item():.6f} "
            f"actor_minus_ref_mean={actor_ref_diff.mean().item():.6f} "
            f"actor_minus_ref_min={actor_ref_diff.min().item():.6f} "
            f"actor_minus_ref_max={actor_ref_diff.max().item():.6f} "
            if ref_valid is not None and actor_ref_diff is not None
            else ""
        )

        print(
            "[RolloutCorrDebug][actor_logprob_alignment] "
            f"shape={tuple(log_prob.shape)} "
            f"valid_tokens={int(valid.sum().item())} "
            f"current_mean={current_valid.mean().item():.6f} "
            f"current_min={current_valid.min().item():.6f} "
            f"current_max={current_valid.max().item():.6f} "
            f"rollout_mean={rollout_valid.mean().item():.6f} "
            f"rollout_min={rollout_valid.min().item():.6f} "
            f"rollout_max={rollout_valid.max().item():.6f} "
            f"diff_mean={diff_valid.mean().item():.6f} "
            f"diff_min={diff_valid.min().item():.6f} "
            f"diff_max={diff_valid.max().item():.6f} "
            f"{ref_stats}"
            f"top_sequences={top_sequences} "
            f"sample_rows={rows}",
            flush=True,
        )
        _ROLLOUT_CORR_DEBUG_COUNT += 1


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

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    config.global_batch_info["global_rollout_count"] = tu.get_non_tensor_data(data, "global_rollout_count", None)

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
    if "rollout_loss_weights" in data:
        fields.append("rollout_loss_weights")
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    debug_enabled = loss_mode == "bypass_mode" and _ROLLOUT_CORR_DEBUG_COUNT < _rollout_corr_debug_limit()
    row_metadata = _debug_collect_row_metadata(data) if debug_enabled else None
    if loss_mode == "bypass_mode" and "responses" in data:
        fields.append("responses")
    if debug_enabled:
        for key in ("input_ids", "prompts", "attention_mask", "position_ids"):
            if key in data.keys():
                fields.append(key)
    data = data.select(*dict.fromkeys(fields)).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_loss_weights = data.get("rollout_loss_weights", None)
    rollout_is_weights = data.get("rollout_is_weights", None)
    config.global_batch_info["rollout_loss_weights"] = rollout_loss_weights

    loss_agg_mode = config.loss_agg_mode

    _debug_rollout_corr_logprob_alignment(data, log_prob, old_log_prob, response_mask, loss_mode, row_metadata)
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
        kl_loss_coeff = float(config.kl_loss_coef)
        if kl_loss_coeff == 0.0:
            with torch.no_grad():
                kld = kl_penalty(
                    logprob=log_prob.detach(),
                    ref_logprob=ref_log_prob,
                    kl_penalty=config.kl_loss_type,
                )
                kl_loss = agg_loss(
                    loss_mat=kld,
                    loss_mask=response_mask,
                    loss_agg_mode=config.loss_agg_mode,
                    **config.global_batch_info,
                )
        else:
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=response_mask,
                loss_agg_mode=config.loss_agg_mode,
                **config.global_batch_info,
            )
            policy_loss += kl_loss * kl_loss_coeff
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = kl_loss_coeff

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
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

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
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics

# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from copy import copy
from functools import partial
from typing import Callable

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import compute_value_loss
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.losses import ppo_loss, sft_loss, value_loss
from verl.workers.utils.padding import no_padding_2_padding

_DCP_TOKEN_DECOMPOSABLE_POLICY_LOSSES = {
    "vanilla",
    "dppo_tv",
    "dppo_kl",
    "sapo",
    "gpg",
    "cispo",
}


def _unwrap_partial(loss_function: Callable) -> Callable:
    """Return the callable beneath an arbitrarily nested partial."""
    while isinstance(loss_function, partial):
        loss_function = loss_function.func
    return loss_function


def _get_loss_config(loss_function: Callable):
    return _get_partial_keyword(loss_function, "config")


def _get_partial_keyword(loss_function: Callable, key: str, default=None):
    # Outer partial keywords take precedence over inner ones, matching call
    # semantics even for partial-like objects that have not been flattened.
    while isinstance(loss_function, partial):
        keywords = loss_function.keywords or {}
        if key in keywords:
            return keywords[key]
        loss_function = loss_function.func
    return default


def _replace_partial_keyword(loss_function: Callable, key: str, value) -> Callable:
    """Copy a partial chain while replacing one bound keyword."""
    if not isinstance(loss_function, partial):
        return loss_function
    keywords = dict(loss_function.keywords or {})
    if key in keywords:
        keywords[key] = value
        return partial(loss_function.func, *loss_function.args, **keywords)
    return partial(
        _replace_partial_keyword(loss_function.func, key, value),
        *loss_function.args,
        **keywords,
    )


def _config_get(config, key: str, default=None):
    """Read dataclass, DictConfig, or mapping-style loss configuration."""
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except (AttributeError, KeyError, TypeError):
            pass
    return getattr(config, key, default)


def _normalize_loss_mode(mode) -> str:
    return str(mode or "vanilla").lower().replace("-", "_")


def _validate_dcp_policy_mode(mode, *, source: str) -> None:
    """Reject policy objectives that need tokens hidden on other DCP ranks."""
    mode = _normalize_loss_mode(mode)
    if mode in _DCP_TOKEN_DECOMPOSABLE_POLICY_LOSSES:
        return

    raise NotImplementedError(
        f"Dynamic context parallelism does not support {source}={mode!r}: the objective has not been verified "
        "to decompose over token shards and may perform sequence- or batch-level statistics after sharding. "
        f"Use one of {sorted(_DCP_TOKEN_DECOMPOSABLE_POLICY_LOSSES)!r}, disable dynamic context parallelism, "
        "or add an explicit DCP-equivalence implementation and test."
    )


def validate_dcp_policy_loss(loss_function: Callable) -> None:
    """Fail early when a policy objective cannot be computed from DCP token shards."""
    base_loss_function = _unwrap_partial(loss_function)
    known_base_loss = any(base_loss_function is candidate for candidate in (sft_loss, ppo_loss, value_loss))
    if not known_base_loss:
        # Keep this import local: the distillation module imports ppo_loss and
        # importing it while this module is initialized would create a cycle.
        from verl.trainer.distillation.losses import distillation_ppo_loss

        if base_loss_function is not distillation_ppo_loss:
            name = getattr(base_loss_function, "__qualname__", repr(base_loss_function))
            raise NotImplementedError(
                f"Dynamic context parallelism does not support top-level loss callable {name!r}. "
                "Only sft_loss, ppo_loss, value_loss, and distillation_ppo_loss have verified DCP-equivalent "
                "implementations. Disable dynamic context parallelism or add an explicit DCP-equivalence "
                "implementation and test."
            )

    if base_loss_function is sft_loss or base_loss_function is value_loss:
        return

    config = _get_loss_config(loss_function)
    distillation_config = _get_partial_keyword(loss_function, "distillation_config", None)
    distillation_loss_config = _config_get(distillation_config, "distillation_loss", None)

    if base_loss_function is not ppo_loss:
        if distillation_loss_config is None:
            raise ValueError("DCP distillation_ppo_loss requires distillation_config.distillation_loss")
        from verl.trainer.distillation.losses import get_distillation_loss_settings

        distillation_loss_mode = _config_get(distillation_loss_config, "loss_mode", None)
        settings = get_distillation_loss_settings(distillation_loss_mode)
        if settings.dcp_compatible is not True:
            raise NotImplementedError(
                "Dynamic context parallelism does not support distillation loss mode "
                f"{distillation_loss_mode!r}: it has not been explicitly verified to decompose over token shards. "
                "Disable dynamic context parallelism or register the loss with dcp_compatible=True after adding "
                "a DCP-equivalence implementation and test."
            )

    actor_policy_is_used = distillation_loss_config is None or bool(
        _config_get(distillation_loss_config, "use_task_rewards", False)
        or _config_get(distillation_loss_config, "use_policy_gradient", False)
    )
    policy_config = _config_get(config, "policy_loss", None)
    if policy_config is not None and actor_policy_is_used:
        _validate_dcp_policy_mode(
            _config_get(policy_config, "loss_mode", "vanilla"),
            source="policy_loss.loss_mode",
        )

    if distillation_loss_config is not None and _config_get(distillation_loss_config, "use_policy_gradient", False):
        _validate_dcp_policy_mode(
            _config_get(distillation_loss_config, "policy_loss_mode", "vanilla"),
            source="distillation_loss.policy_loss_mode",
        )


def _resolve_dcp_loss_scale_factor(config, data: TensorDict):
    """Resolve and validate a rank-invariant normalized sequence horizon."""
    loss_scale_factor = _config_get(config, "loss_scale_factor", None)
    if loss_scale_factor is None:
        loss_scale_factor = tu.get_non_tensor_data(data, key="loss_scale_factor", default=None)
    if loss_scale_factor is None:
        loss_scale_factor = tu.get_non_tensor_data(data, key="max_response_len", default=None)
    if loss_scale_factor is None:
        raise ValueError(
            "Dynamic context parallelism with loss_agg_mode='seq-mean-token-sum-norm' requires a global "
            "loss_scale_factor or max_response_len; a routed local tensor width is not a stable denominator."
        )
    if isinstance(loss_scale_factor, torch.Tensor):
        if loss_scale_factor.numel() != 1:
            raise ValueError("DCP loss_scale_factor must be a positive integer scalar")
        loss_scale_factor = loss_scale_factor.item()
    if not isinstance(loss_scale_factor, int) or isinstance(loss_scale_factor, bool) or loss_scale_factor <= 0:
        raise ValueError("DCP loss_scale_factor must be a positive integer scalar")
    return loss_scale_factor


def validate_dcp_loss_normalization(loss_function: Callable, data: TensorDict) -> None:
    """Require a rank-invariant horizon for normalized sequence-sum losses."""
    if _unwrap_partial(loss_function) is sft_loss:
        return
    config = _get_loss_config(loss_function)
    if _config_get(config, "loss_agg_mode", None) != "seq-mean-token-sum-norm":
        return
    _resolve_dcp_loss_scale_factor(config, data)


def _prepare_dcp_loss_function(
    loss_function: Callable,
    data: TensorDict,
    response_token_counts: torch.Tensor,
) -> tuple[Callable, object]:
    """Bind a per-call config carrying DCP's global loss denominators."""
    config = _get_loss_config(loss_function)
    if config is None:
        raise ValueError("A scheduler-managed DCP loss requires a bound config")

    dcp_config = copy(config)
    global_batch_info = dict(getattr(config, "global_batch_info", {}))
    global_batch_info.update(
        {
            "dp_size": tu.get_non_tensor_data(data, key="dp_size", default=1),
            "batch_num_tokens": tu.get_non_tensor_data(data, key="batch_num_tokens", default=None),
            "global_batch_size": tu.get_non_tensor_data(data, key="global_batch_size", default=None),
            "sequence_token_counts": response_token_counts,
        }
    )
    if _config_get(config, "loss_agg_mode", None) == "seq-mean-token-sum-norm":
        loss_scale_factor = _resolve_dcp_loss_scale_factor(config, data)
        object.__setattr__(dcp_config, "loss_scale_factor", loss_scale_factor)
        global_batch_info["loss_scale_factor"] = loss_scale_factor
    object.__setattr__(dcp_config, "global_batch_info", global_batch_info)
    dcp_loss_function = _replace_partial_keyword(loss_function, "config", dcp_config)

    # distillation_loss() copies ActorConfig.global_batch_info into its nested
    # loss config. Keep that per-micro-batch mutation off the shared trainer
    # configuration, especially because sequence_token_counts varies by route.
    distillation_config = _get_partial_keyword(loss_function, "distillation_config", None)
    if distillation_config is not None:
        dcp_distillation_config = copy(distillation_config)
        distillation_loss_config = _config_get(distillation_config, "distillation_loss", None)
        if distillation_loss_config is None:
            raise ValueError("DCP distillation loss requires distillation_config.distillation_loss")
        dcp_distillation_loss_config = copy(distillation_loss_config)
        object.__setattr__(
            dcp_distillation_loss_config,
            "global_batch_info",
            dict(getattr(distillation_loss_config, "global_batch_info", {})),
        )
        object.__setattr__(dcp_distillation_config, "distillation_loss", dcp_distillation_loss_config)
        dcp_loss_function = _replace_partial_keyword(
            dcp_loss_function,
            "distillation_config",
            dcp_distillation_config,
        )

    return dcp_loss_function, dcp_config


def _dcp_sft_loss(config: ActorConfig, model_output: dict, data: TensorDict):
    """Handle scheduler-local dense SFT masks without changing shared losses."""
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    loss_mask = data["loss_mask"]
    if pad_mode != DatasetPadMode.NO_PADDING or loss_mask.is_nested:
        return sft_loss(config=config, model_output=model_output, data=data)

    log_prob = model_output["log_probs"]
    log_prob = no_padding_2_padding(log_prob, data) if log_prob.is_nested else log_prob
    if log_prob.shape != loss_mask.shape:
        raise ValueError(
            "DCP dense SFT loss_mask must match response-aligned log_probs: "
            f"{tuple(loss_mask.shape)} != {tuple(log_prob.shape)}"
        )
    batch_num_tokens = tu.get_non_tensor_data(data, key="batch_num_tokens", default=None)
    dp_size = tu.get_non_tensor_data(data, key="dp_size", default=None)
    if batch_num_tokens is None or batch_num_tokens <= 0:
        raise ValueError("Scheduler-managed DCP SFT requires a positive global batch_num_tokens")
    if dp_size is None or dp_size <= 0:
        raise ValueError("Scheduler-managed DCP SFT requires a positive dp_size")
    loss = -masked_sum(log_prob, loss_mask) / batch_num_tokens * dp_size
    return loss, {}


def _dcp_value_loss(config: CriticConfig, model_output: dict, data: TensorDict):
    """Compute critic loss with full-sequence denominators for DCP-local tokens."""
    vpreds = no_padding_2_padding(model_output["values"], data)
    response_token_counts = data["response_token_counts"]
    dp_size = tu.get_non_tensor_data(data, key="dp_size", default=1)
    # The generic value_loss reads both keys unconditionally (KeyError on absence).
    # Falling back to rank-local counts would silently change the denominator per
    # DCP shard, so missing metadata must fail loudly here as well.
    batch_num_tokens = tu.get_non_tensor_data(data, key="batch_num_tokens", default=None)
    global_batch_size = tu.get_non_tensor_data(data, key="global_batch_size", default=None)
    if batch_num_tokens is None or batch_num_tokens <= 0:
        raise ValueError("Scheduler-managed DCP value loss requires a positive global batch_num_tokens")
    if global_batch_size is None or global_batch_size <= 0:
        raise ValueError("Scheduler-managed DCP value loss requires a positive global_batch_size")
    loss_scale_factor = None
    if config.loss_agg_mode == "seq-mean-token-sum-norm":
        loss_scale_factor = _resolve_dcp_loss_scale_factor(config, data)

    padded = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = padded["values"]
    returns = padded["returns"]
    response_mask = padded["response_mask"].to(torch.bool)

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
        loss_scale_factor=loss_scale_factor,
        sequence_token_counts=response_token_counts,
    )
    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=AggregationType.SUM),
        "critic/vf_clipfrac": Metric(value=vf_clipfrac, aggregation=AggregationType.MEAN),
        "critic/vpred_mean": Metric(value=masked_mean(vpreds, response_mask), aggregation=AggregationType.MEAN),
    }
    return vf_loss, metrics


def call_megatron_loss(
    loss_function: Callable,
    model_output: dict,
    data: TensorDict,
    dp_group=None,
):
    """Call a shared loss while keeping DCP normalization in the Megatron backend."""
    dcp_scheduled = bool(tu.get_non_tensor_data(data, key="_dcp_scheduled", default=False))
    if not dcp_scheduled:
        return loss_function(model_output=model_output, data=data, dp_group=dp_group)

    validate_dcp_policy_loss(loss_function)
    validate_dcp_loss_normalization(loss_function, data)
    base_loss_function = _unwrap_partial(loss_function)
    config = _get_loss_config(loss_function)
    if base_loss_function is sft_loss:
        return _dcp_sft_loss(config, model_output, data)

    response_token_counts = data.get("response_token_counts", None)
    if response_token_counts is None:
        raise ValueError("Scheduler-managed DCP policy and critic losses require full response token counts")

    if base_loss_function is value_loss:
        return _dcp_value_loss(config, model_output, data)

    dcp_loss_function, _ = _prepare_dcp_loss_function(loss_function, data, response_token_counts)
    return dcp_loss_function(model_output=model_output, data=data, dp_group=dp_group)

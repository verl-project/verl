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

import hashlib
import inspect
import logging
import math
import os
from functools import partial
from typing import Any, Callable, ContextManager, Iterator

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.package_info import __version__
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.models.mcore import get_mcore_weight_converter
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    build_r3_replay_mask,
    merge_router_topk_indices,
    pp_gather,
    reorder_and_merge_vpp_layers,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import (
    vocab_parallel_entropy,
    vocab_parallel_entropy_with_chunking,
    vocab_parallel_log_probs_from_logits,
    vocab_parallel_sum_pi_squared,
)
from verl.utils.megatron_peft_utils import add_base_layer_suffix, build_peft_config_for_vllm
from verl.utils.megatron_utils import (
    check_mtp_config,
    get_megatron_module_device,
    get_megatron_mtp_loss,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    patch_engine_mtp,
    register_megatron_training_hooks,
    unwrap_model,
)
from verl.utils.metric import AggregationType, Metric
from verl.utils.model import extract_multi_modal_inputs, load_mcore_dist_weights
from verl.utils.seqlen_balancing import restore_dynamic_batch
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig, get_mcore_parallel_topology
from verl.workers.utils.padding import no_padding_2_padding

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from .losses import call_megatron_loss, validate_dcp_loss_normalization, validate_dcp_policy_loss
from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _resolve_dcp_transformer_overrides(
    engine_config: McoreEngineConfig, override_transformer_config: dict[str, Any]
) -> dict[str, Any]:
    """Apply the model-side invariants required by verl-managed DCP."""
    overrides = dict(override_transformer_config)
    if not engine_config.dynamic_context_parallel:
        return overrides

    static_cp_size = engine_config.context_parallel_size
    requested_cp_size = overrides.get("context_parallel_size", static_cp_size)
    if requested_cp_size != static_cp_size:
        raise ValueError(
            "Dynamic CP requires override_transformer_config.context_parallel_size to match "
            f"engine.context_parallel_size ({static_cp_size}), got {requested_cp_size}."
        )
    if overrides.get("calculate_per_token_loss") is False:
        raise ValueError("Dynamic CP requires calculate_per_token_loss=True for global token normalization.")

    overrides.update(
        {
            "calculate_per_token_loss": True,
            "context_parallel_size": static_cp_size,
            # verl schedules and routes before Megatron's pipeline schedule.
            "dynamic_context_parallel": False,
            "max_seqlen_per_dp_cp_rank": engine_config.max_seqlen_per_dp_cp_rank,
        }
    )
    return overrides


def _validate_resolved_dcp_transformer_config(engine_config: McoreEngineConfig, tf_config) -> None:
    """Verify that Bridge/provider finalization preserved DCP invariants."""
    if not engine_config.dynamic_context_parallel:
        return

    expected = {
        "calculate_per_token_loss": True,
        "context_parallel_size": engine_config.context_parallel_size,
        # verl invokes the MCore scheduler itself and must not schedule twice.
        "dynamic_context_parallel": False,
        "max_seqlen_per_dp_cp_rank": engine_config.max_seqlen_per_dp_cp_rank,
    }
    mismatches = {
        key: (expected_value, getattr(tf_config, key, None))
        for key, expected_value in expected.items()
        if getattr(tf_config, key, None) != expected_value
    }
    if mismatches:
        raise ValueError(f"Megatron-Bridge did not preserve the Dynamic CP transformer invariants: {mismatches}")

    moe_z_loss_coeff = getattr(tf_config, "moe_z_loss_coeff", None)
    if moe_z_loss_coeff is not None and moe_z_loss_coeff != 0:
        raise NotImplementedError(
            "Dynamic context parallelism does not yet support moe_z_loss_coeff: the resolved Megatron-Core "
            f"configuration enables it with coefficient {moe_z_loss_coeff}."
        )


def _nested_with_values_like(nested_tensor: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # Preserve cached jagged-length bounds. Without them,
    # nested_tensor_from_jagged conservatively pads to total values length.
    min_seqlen = getattr(nested_tensor, "_min_seqlen", None)
    max_seqlen = getattr(nested_tensor, "_max_seqlen", None)
    if min_seqlen is None or max_seqlen is None:
        seqlens = nested_tensor.offsets().diff()
        min_seqlen = int(seqlens.min().item())
        max_seqlen = int(seqlens.max().item())
    return torch.nested.nested_tensor_from_jagged(
        values,
        offsets=nested_tensor.offsets(),
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
    )


def _slice_dcp_response_field(data: TensorDict, key: str, response_len: int):
    if key not in data.keys():
        return
    value = data[key]
    if not isinstance(value, torch.Tensor) or value.is_nested or value.dim() < 2:
        return
    if value.shape[1] == response_len:
        return
    if value.shape[1] < response_len:
        raise ValueError(f"{key} is shorter than DCP model output: {value.shape[1]} < {response_len}")
    data[key] = value[:, :response_len]


def _validate_dcp_multi_modal_inputs(data: TensorDict, dcp_group=None) -> None:
    """Collectively reject VLM batches until DCP can route multimodal tensors."""
    has_multi_modal_inputs = False
    if "multi_modal_inputs" in data.keys():
        try:
            has_multi_modal_inputs = bool(extract_multi_modal_inputs(data.get("multi_modal_inputs", [])))
        except Exception:
            # Malformed multimodal inputs are unsupported by DCP too. Convert
            # local preprocessing failures into a collective rejection so peer
            # ranks do not continue into an all-to-all and hang.
            has_multi_modal_inputs = True

    if dcp_group is not None:
        has_multi_modal_inputs_tensor = torch.tensor(
            int(has_multi_modal_inputs), dtype=torch.int32, device=get_device_id()
        )
        torch.distributed.all_reduce(
            has_multi_modal_inputs_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=dcp_group,
        )
        has_multi_modal_inputs = bool(has_multi_modal_inputs_tensor.item())

    if has_multi_modal_inputs:
        raise NotImplementedError(
            "Dynamic context parallelism does not yet support multi_modal_inputs; "
            "disable dynamic_context_parallel for VLM batches."
        )


def _prepare_dcp_temperature(data: TensorDict, dcp_group=None) -> torch.Tensor | float | None:
    """Classify temperature as replicated metadata or a routed sample field.

    Tensor temperatures with a sample dimension are always routed, including
    one-element and empty tensors. Only Python scalars and zero-dimensional
    tensors remain replicated metadata. When a process group is supplied, all
    ranks validate the classification before the scheduler starts collectives.
    """
    temperature = tu.get_non_tensor_data(data, key="temperature", default=None)
    local_error = None
    route_per_sample = isinstance(temperature, torch.Tensor) and temperature.ndim > 0
    replicated_scalar = None

    if isinstance(temperature, torch.Tensor):
        if temperature.is_nested:
            local_error = "DCP temperature must be a dense scalar or one dense value per sample"
        elif route_per_sample and temperature.numel() != len(data):
            local_error = (
                "A per-sample DCP temperature tensor must have one value per sample: "
                f"got shape {tuple(temperature.shape)} for batch size {len(data)}."
            )
        elif route_per_sample:
            try:
                valid_temperature = bool(torch.isfinite(temperature).all().item() and torch.all(temperature > 0).item())
            except (RuntimeError, TypeError):
                valid_temperature = False
            if not valid_temperature:
                local_error = "DCP temperature values must be strictly positive and finite"
        else:
            try:
                replicated_scalar = float(temperature.item())
            except (TypeError, ValueError, RuntimeError):
                local_error = "DCP temperature metadata must be a real scalar"
            if replicated_scalar is not None and (not math.isfinite(replicated_scalar) or replicated_scalar <= 0):
                local_error = "DCP temperature values must be strictly positive and finite"
    elif temperature is not None:
        try:
            temperature = float(temperature)
            replicated_scalar = temperature
        except (TypeError, ValueError):
            local_error = "DCP temperature metadata must be a Python scalar or a tensor with one value per sample"
        if replicated_scalar is not None and (not math.isfinite(replicated_scalar) or replicated_scalar <= 0):
            local_error = "DCP temperature values must be strictly positive and finite"

    if dcp_group is not None:
        state = torch.tensor(
            [
                float(local_error is not None),
                float(route_per_sample),
                float(replicated_scalar is not None),
                replicated_scalar if replicated_scalar is not None else 0.0,
            ],
            dtype=torch.float64,
            device=get_device_id(),
        )
        states = [torch.empty_like(state) for _ in range(dcp_group.size())]
        torch.distributed.all_gather(states, state, group=dcp_group)
        if any(bool(peer[0].item()) for peer in states):
            raise ValueError(local_error or "Invalid DCP temperature was detected on another rank")
        route_states = {int(peer[1].item()) for peer in states}
        if len(route_states) != 1:
            raise ValueError(
                "DCP temperature must be per-sample on every rank or scalar metadata on every rank; "
                "mixed routing classifications would produce different collective schemas"
            )
        if not route_per_sample:
            scalar_presence = {int(peer[2].item()) for peer in states}
            if len(scalar_presence) != 1:
                raise ValueError(
                    "DCP scalar temperature metadata must be present on every rank or absent on every rank"
                )
            if scalar_presence == {1}:
                reference = states[0][3]
                if any(not torch.equal(peer[3], reference) for peer in states[1:]):
                    raise ValueError("DCP scalar temperature metadata must have the same value on every rank")
    elif local_error is not None:
        raise ValueError(local_error)

    if isinstance(temperature, torch.Tensor):
        if route_per_sample:
            # Replacing NonTensorData here is intentional: every tensor with a
            # sample dimension must appear in the scheduler's routed schema.
            data["temperature"] = temperature.reshape(len(data))
            return None
        # Store the scalar as a Python float: a zero-dimensional tensor must not
        # enter the routed schema, whose scalar fields are one value per sample.
        temperature = replicated_scalar
        tu.assign_non_tensor_data(data, "temperature", temperature)

    return temperature


def _validate_dcp_model_features(model_config: HFModelConfig, engine_config: McoreEngineConfig) -> None:
    """Reject model-side losses whose normalization is not DCP-equivalent."""
    mtp_config = model_config.mtp
    if engine_config.dynamic_context_parallel and mtp_config.enable and mtp_config.enable_train:
        raise NotImplementedError(
            "Dynamic context parallelism does not yet support MTP training: MTP normalizes each rank's local "
            "rolled-token loss before DP+CP gradient averaging, which is not equivalent for unequal DCP shards."
        )
    override_transformer_config = getattr(engine_config, "override_transformer_config", {}) or {}
    moe_z_loss_coeff = override_transformer_config.get("moe_z_loss_coeff")
    if engine_config.dynamic_context_parallel and moe_z_loss_coeff is not None and moe_z_loss_coeff != 0:
        raise NotImplementedError(
            "Dynamic context parallelism does not yet support moe_z_loss_coeff: Megatron-Core normalizes "
            "router z-loss by each rank's local token count, which is not equivalent for unequal DCP shards."
        )


def _apply_dcp_local_token_mask_for_loss(model_output: dict[str, torch.Tensor], data: TensorDict) -> None:
    """Apply DCP-local token ownership to Megatron loss inputs.

    DCP is a Megatron scheduling detail, so backend-agnostic loss functions should
    not know about ``_dcp_local_token_mask``. This helper rewrites the masks in
    the scheduled Megatron micro-batch before calling the shared losses.
    """
    local_token_mask = model_output.pop("_dcp_local_token_mask", None)
    if local_token_mask is None:
        return
    if not isinstance(local_token_mask, torch.Tensor) or not local_token_mask.is_nested:
        raise ValueError("_dcp_local_token_mask must be a nested tensor")
    tu.assign_non_tensor(
        data,
        _dcp_local_num_tokens=local_token_mask.values().sum().to(dtype=torch.int),
    )

    if "loss_mask" in data.keys():
        loss_mask = data["loss_mask"]
        if isinstance(loss_mask, torch.Tensor) and loss_mask.is_nested:
            if not torch.equal(loss_mask.offsets(), local_token_mask.offsets()):
                raise ValueError("DCP local token mask offsets must match loss_mask offsets")
            shifted_loss_mask = torch.roll(loss_mask.values(), shifts=-1, dims=0)
            shifted_loss_mask = shifted_loss_mask * local_token_mask.values().to(dtype=shifted_loss_mask.dtype)
            log_probs = model_output.get("log_probs", None)
            if isinstance(log_probs, torch.Tensor) and log_probs.is_nested:
                if torch.equal(log_probs.offsets(), local_token_mask.offsets()):
                    data["loss_mask"] = _nested_with_values_like(
                        loss_mask, torch.roll(shifted_loss_mask, shifts=1, dims=0)
                    )
                else:
                    compact_loss_mask = shifted_loss_mask[local_token_mask.values().to(dtype=torch.bool)]
                    if compact_loss_mask.numel() != log_probs.values().numel():
                        raise ValueError(
                            "DCP compact loss mask size must match compact log_probs size: "
                            f"{compact_loss_mask.numel()} != {log_probs.values().numel()}"
                        )
                    data["loss_mask"] = _nested_with_values_like(
                        log_probs, torch.roll(compact_loss_mask, shifts=1, dims=0)
                    )
            else:
                data["loss_mask"] = _nested_with_values_like(loss_mask, torch.roll(shifted_loss_mask, shifts=1, dims=0))

    if "response_mask" not in data.keys():
        return

    response_mask = data["response_mask"]
    if not isinstance(response_mask, torch.Tensor):
        return

    if "_dcp_response_mask_for_padding" not in data.keys():
        data["_dcp_response_mask_for_padding"] = response_mask.clone()
    full_response_mask = data["_dcp_response_mask_for_padding"]
    if full_response_mask.is_nested:
        response_token_counts = torch.stack([part.to(torch.bool).sum() for part in full_response_mask.unbind()], dim=0)
    else:
        response_token_counts = full_response_mask.sum(dim=-1)
    data["response_token_counts"] = response_token_counts

    local_response_mask = no_padding_2_padding(local_token_mask.to(dtype=torch.float32), data).to(torch.bool)
    response_len = local_response_mask.shape[1]

    loss_mask = data.get("loss_mask", None)
    if isinstance(loss_mask, torch.Tensor) and not loss_mask.is_nested:
        _slice_dcp_response_field(data, "loss_mask", response_len)
        data["loss_mask"] = data["loss_mask"] * local_response_mask.to(dtype=data["loss_mask"].dtype)

    if full_response_mask.is_nested:
        local_response_parts = []
        for sample_idx, part in enumerate(full_response_mask.unbind()):
            part_len = part.shape[0]
            if part_len > response_len:
                raise ValueError(
                    "DCP nested response mask is longer than the reconstructed response span: "
                    f"{part_len} > {response_len}"
                )
            local_response_parts.append(local_response_mask[sample_idx, :part_len])
        local_response_values = torch.cat(local_response_parts, dim=0)
        data["response_mask"] = _nested_with_values_like(
            response_mask,
            response_mask.values().to(torch.bool) & local_response_values,
        )
    else:
        if full_response_mask.shape[1] > response_len and full_response_mask[:, response_len:].any():
            raise ValueError("DCP response alignment would drop non-padding response tokens")

    for key in [
        "old_log_probs",
        "advantages",
        "rollout_is_weights",
        "ref_log_prob",
        "values",
        "returns",
    ]:
        _slice_dcp_response_field(data, key, response_len)

    if not response_mask.is_nested:
        _slice_dcp_response_field(data, "response_mask", response_len)
        data["response_mask"] = data["response_mask"].to(torch.bool) & local_response_mask


def _normalize_temperature_for_thd(
    temperature, input_ids: torch.Tensor, *, strict: bool = True
) -> tuple[torch.Tensor, float | None]:
    """Normalize temperature once for both fused and non-fused THD forwards.

    The strict mode backs the Dynamic CP path, whose temperatures were already
    validated collectively. The non-strict mode preserves the legacy engine
    behavior: a missing temperature is an error instead of a silent 1.0, and
    non-positive entries (such as padding zeros) are clamped, not rejected.
    """
    temperature = tu.unwrap_non_tensor_data(temperature)
    batch_size = input_ids.shape[0]
    if temperature is None:
        if not strict:
            raise ValueError("Megatron THD forward requires a 'temperature' entry in the batch")
        temperature = torch.ones(batch_size, device=input_ids.device, dtype=torch.float32)
    elif not isinstance(temperature, torch.Tensor):
        temperature = torch.full((batch_size,), float(temperature), device=input_ids.device, dtype=torch.float32)
    elif temperature.is_nested:
        raise ValueError("Megatron THD temperature must be a dense scalar or per-sample tensor")

    temperature = temperature.to(device=input_ids.device, dtype=torch.float32)
    if temperature.numel() == 0:
        temperature = torch.ones(batch_size, device=input_ids.device, dtype=torch.float32)
    elif temperature.numel() == 1:
        temperature = temperature.reshape(1).expand(batch_size)
    elif temperature.numel() == batch_size:
        temperature = temperature.reshape(batch_size)
    else:
        raise ValueError(
            "Megatron THD path expects a scalar or one temperature per sample. "
            f"Got shape {tuple(temperature.shape)} for batch size {batch_size}."
        )

    if not torch.isfinite(temperature).all().item():
        raise ValueError(f"Megatron THD temperature must be strictly positive and finite, got {temperature}")
    if not torch.all(temperature > 0).item():
        if strict:
            raise ValueError(f"Megatron THD temperature must be strictly positive and finite, got {temperature}")
        # avoid non-positive temperature such as padding (legacy engine behavior)
        temperature = temperature.clone()
        temperature[temperature <= 0] = 1e-8
    fused_temperature = None
    if torch.equal(temperature, temperature[:1].expand_as(temperature)):
        fused_temperature = float(temperature[0].item())
    return verl_F.expand_as_nested(temperature, input_ids), fused_temperature


def _aggregate_dcp_loss_for_logging(losses: list[float], loss_normalization_world_size: int, dcp_group) -> list[float]:
    """Turn per-rank DCP loss shards into the global loss reported by static CP."""
    loss = torch.tensor(losses, dtype=torch.float64, device=get_device_id()).sum()
    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=dcp_group)
    return [(loss / loss_normalization_world_size).item()]


def _aggregate_dcp_metrics_for_logging(
    outputs: list[dict], loss_normalization_world_size: int, dcp_group
) -> str | None:
    """Aggregate sharded DCP metrics before the normal DP logging path."""
    entries: dict[AggregationType, list[tuple[str, int, Metric, float]]] = {
        aggregation: [] for aggregation in AggregationType
    }
    schema = []
    for output_idx, output in enumerate(outputs):
        weight = float(output.pop("_dcp_metric_weight", 0.0))
        for key, metric in output.get("metrics", {}).items():
            if not isinstance(metric, Metric):
                continue
            metric_weight = float(getattr(metric, "dcp_weight", weight))
            entries[metric.aggregation].append((key, output_idx, metric, metric_weight))
            schema.append((output_idx, key, metric.aggregation.value, len(metric.values)))

    # Every rank must issue collectives with the same metric keys and shapes.
    # Compare a deterministic fixed-size signature before reducing any values so
    # a conditional/custom metric fails clearly instead of hanging NCCL or being
    # silently paired with a different key on another rank.
    schema.sort()
    schema_digest = hashlib.sha256(repr(schema).encode()).digest()
    schema_hash = int.from_bytes(schema_digest[:8], byteorder="little") & ((1 << 63) - 1)
    signature = torch.tensor([len(schema), schema_hash], dtype=torch.int64, device=get_device_id())
    signatures = [torch.empty_like(signature) for _ in range(dcp_group.size())]
    torch.distributed.all_gather(signatures, signature, group=dcp_group)
    if any(not torch.equal(peer, signature) for peer in signatures):
        return f"DCP metric schema differs across ranks; local schema={schema}"
    if any(len(metric.values) != 1 for metric_entries in entries.values() for _, _, metric, _ in metric_entries):
        return "Each DCP micro-batch metric must contain exactly one value"

    for aggregation, metric_entries in entries.items():
        if not metric_entries:
            continue
        metric_entries.sort(key=lambda entry: (entry[0], entry[1]))
        values = torch.tensor(
            [float(metric.values[0]) for _, _, metric, _ in metric_entries],
            dtype=torch.float64,
            device=get_device_id(),
        )
        weights = torch.tensor(
            [weight for _, _, _, weight in metric_entries], dtype=torch.float64, device=values.device
        )
        if aggregation == AggregationType.MEAN:
            reduced = torch.stack((values * weights, weights), dim=0)
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM, group=dcp_group)
            values = torch.where(reduced[1] > 0, reduced[0] / reduced[1], torch.zeros_like(reduced[0]))
        else:
            if aggregation == AggregationType.MIN:
                values = torch.where(weights > 0, values, torch.full_like(values, torch.inf))
            elif aggregation == AggregationType.MAX:
                values = torch.where(weights > 0, values, torch.full_like(values, -torch.inf))
            reduce_op = {
                AggregationType.SUM: torch.distributed.ReduceOp.SUM,
                AggregationType.MIN: torch.distributed.ReduceOp.MIN,
                AggregationType.MAX: torch.distributed.ReduceOp.MAX,
            }[aggregation]
            torch.distributed.all_reduce(values, op=reduce_op, group=dcp_group)
            if aggregation == AggregationType.SUM:
                values /= loss_normalization_world_size
            elif aggregation in {AggregationType.MIN, AggregationType.MAX}:
                values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))

        for (_, _, metric, _), value in zip(metric_entries, values.tolist(), strict=True):
            metric.values[:] = [value]
            if hasattr(metric, "dcp_weight"):
                delattr(metric, "dcp_weight")
    return None


def _synchronize_dcp_metric_error(error: str | None, model_parallel_group) -> None:
    """Make every PP stage fail together after a last-stage metric validation error."""
    error_flag = torch.tensor(int(error is not None), dtype=torch.int32, device=get_device_id())
    torch.distributed.all_reduce(error_flag, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group)
    if error_flag.item():
        raise RuntimeError(error or "DCP metric validation failed on another pipeline stage")


class MegatronEngine(BaseEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"
        _validate_dcp_model_features(self.model_config, self.engine_config)
        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None

        # QAT configuration
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            if self.engine_config.vanilla_mbridge:
                raise ValueError(
                    "QAT requires non-vanilla Megatron bridge. "
                    "Please set 'use_mbridge=True' and 'vanilla_mbridge=False'."
                )
            logger.info(f"QAT enabled in MegatronEngine: mode={self._qat_config.mode}")

        # Router replay configuration for MoE models
        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        logger.info(f"enable_routing_replay in MegatronEngine: {self.enable_routing_replay}")
        if self.enable_routing_replay:
            apply_router_replay_patch()
            self.mini_layer_topk_idx_list = []
        # Apply checkpoint patch for MoE models
        from verl.utils.device import is_cuda_available, is_npu_available

        if is_npu_available and __version__ >= "0.16.0":
            from verl.models.mcore.patch import apply_mtp_inference_patch

            apply_mtp_inference_patch()

        if is_cuda_available:
            from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

            apply_patch_megatron_recomputation_backward()

    def _init_device_mesh(self):
        # TODO: set different parallelism for actor, critic, ref
        extra_args = dict()

        if self.engine_config.dynamic_context_parallel:
            if "dynamic_context_parallel" not in inspect.signature(mpu.initialize_model_parallel).parameters:
                raise RuntimeError("Dynamic CP requires a Megatron-Core build containing NVIDIA/Megatron-LM#5154.")
            if not self.engine_config.use_remove_padding:
                raise ValueError("dynamic_context_parallel requires use_remove_padding=True")
            # Dataclass construction validates the immutable per-rank sequence limit.
            from verl.utils.dynamic_cp_scheduler import _get_megatron_dynamic_cp_scheduler_cls

            _get_megatron_dynamic_cp_scheduler_cls()
            extra_args["dynamic_context_parallel"] = self.engine_config.dynamic_context_parallel

            model_parallel_size = (
                self.engine_config.tensor_model_parallel_size * self.engine_config.pipeline_model_parallel_size
            )
            world_size = torch.distributed.get_world_size()
            if world_size % model_parallel_size != 0:
                raise ValueError(
                    "Dynamic CP requires world_size to be divisible by tensor_model_parallel_size * "
                    f"pipeline_model_parallel_size, got {world_size} % {model_parallel_size}."
                )
            dpcp_world_size = world_size // model_parallel_size
            if dpcp_world_size < 2 or (dpcp_world_size & (dpcp_world_size - 1)) != 0:
                raise ValueError(
                    "Dynamic CP requires the DPxCP world size to be a power of two: the supported "
                    "Megatron-Core build only creates dynamic CP process groups for power-of-two sizes "
                    f"while its scheduler assigns power-of-two CP group sizes, got {dpcp_world_size}."
                )

        if mpu.is_initialized():
            if self.engine_config.dynamic_context_parallel:
                expected_topology = get_mcore_parallel_topology(self.engine_config)
                actual_topology = {
                    "tensor": mpu.get_tensor_model_parallel_world_size(),
                    "pipeline": mpu.get_pipeline_model_parallel_world_size(),
                    "virtual_pipeline": mpu.get_virtual_pipeline_model_parallel_world_size(),
                    "context": mpu.get_context_parallel_world_size(),
                    "expert": mpu.get_expert_model_parallel_world_size(),
                    "expert_tensor": mpu.get_expert_tensor_parallel_world_size(),
                }
                if actual_topology != expected_topology:
                    raise ValueError(
                        "All colocated Megatron engines must use the same TP/PP/VPP/CP/EP/ETP topology when "
                        "Dynamic CP is "
                        f"enabled: expected={expected_topology}, initialized={actual_topology}."
                    )
                try:
                    mpu.get_dynamic_data_context_parallel_groups(group_size=1)
                except (AssertionError, AttributeError, KeyError) as exc:
                    raise RuntimeError(
                        "Dynamic CP was enabled after Megatron model-parallel groups had already been initialized "
                        "without dynamic DPxCP groups. Enable Dynamic CP consistently for colocated ref/actor/critic "
                        "engines."
                    ) from exc
            return

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
            **extra_args,
        )

    def _build_tf_config(self):
        from verl.utils.megatron_utils import mapping_string_to_attn_backend
        from verl.utils.torch_dtypes import PrecisionType

        self.is_value_model = self.model_config.model_type == "value_model"
        self.share_embeddings_and_output_weights = self.model_config.share_embeddings_and_output_weights

        check_mtp_config(self.model_config, self.engine_config)

        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        override_transformer_config = mapping_string_to_attn_backend({**self.engine_config.override_transformer_config})
        override_transformer_config = _resolve_dcp_transformer_overrides(
            self.engine_config, override_transformer_config
        )
        if self.is_value_model:
            # A value head cannot share weights with the vocabulary embedding. This must
            # be set before either bridge creates and finalizes its Megatron config.
            self.model_config.hf_config.tie_word_embeddings = False
            self.share_embeddings_and_output_weights = False
            override_transformer_config["share_embeddings_and_output_weights"] = False
        self.provider = None
        self.vanilla_bridge = self.engine_config.vanilla_mbridge

        if self.vanilla_bridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            tf_config.fp16 = self.param_dtype == torch.float16
            tf_config.bf16 = self.param_dtype == torch.bfloat16
        else:
            from verl.models.mcore.bridge import AutoBridge

            # Use Megatron-Bridge to convert HF config to Megatron config
            bridge = AutoBridge.from_hf_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            # Get Megatron provider and configure it
            provider = bridge.to_megatron_provider(load_weights=False)

            # Match verl implementation (need variable_seq_lengths)
            from megatron.core.transformer.enums import AttnBackend

            virtual_pipeline_model_parallel_size = self.engine_config.virtual_pipeline_model_parallel_size
            provider_overrides = {
                "tensor_model_parallel_size": self.engine_config.tensor_model_parallel_size,
                "pipeline_model_parallel_size": self.engine_config.pipeline_model_parallel_size,
                "expert_model_parallel_size": self.engine_config.expert_model_parallel_size,
                "expert_tensor_parallel_size": self.engine_config.expert_tensor_parallel_size,
                "virtual_pipeline_model_parallel_size": virtual_pipeline_model_parallel_size,
                "context_parallel_size": self.engine_config.context_parallel_size,
                "sequence_parallel": self.engine_config.sequence_parallel,
                "overlap_p2p_comm": (
                    virtual_pipeline_model_parallel_size is not None and virtual_pipeline_model_parallel_size > 1
                ),
                "batch_p2p_comm": False,
                "variable_seq_lengths": True,
                "attention_backend": AttnBackend.flash,
                "moe_token_dispatcher_type": "alltoall",
                "moe_router_load_balancing_type": "none",
            }
            for key, value in override_transformer_config.items():
                provider_overrides[key] = value
            if (
                self.model_config.hf_config.model_type == "deepseek_v4"
                and not self.model_config.mtp.enable
                and getattr(provider, "mtp_num_layers", 0)
            ):
                provider_overrides["mtp_num_layers"] = 0
                csa_compress_ratios = getattr(provider, "csa_compress_ratios", None)
                if csa_compress_ratios is not None:
                    provider_overrides["csa_compress_ratios"] = csa_compress_ratios[: provider.num_layers]
            if self.enable_routing_replay:
                if hasattr(provider, "moe_enable_routing_replay"):
                    provider_overrides["moe_enable_routing_replay"] = True
                else:
                    provider_overrides["enable_routing_replay"] = True

            if self._qat_enabled:
                from megatron.bridge.models.gpt_provider import modelopt_transformer_layer_spec

                provider.transformer_layer_spec = modelopt_transformer_layer_spec

            provider.apply_overrides_and_finalize(
                dtype=self.param_dtype,
                overrides=provider_overrides,
            )
            self.provider = provider
            tf_config = None  # Will be set after model creation
        self.bridge = bridge

        _validate_resolved_dcp_transformer_config(
            self.engine_config,
            tf_config if tf_config is not None else self.provider,
        )

        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.model_config.hf_config, self.dtype)

        # Set router replay directly on tf_config instead of passing through
        # override_transformer_config, because dataclass subclasses like MLATransformerConfig
        # generate their own __init__ and may not accept compatibility kwargs.
        if self.enable_routing_replay and tf_config is not None:
            if hasattr(tf_config, "moe_enable_routing_replay"):
                tf_config.moe_enable_routing_replay = True
            else:
                tf_config.enable_routing_replay = True

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.tf_config = tf_config

        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.model_config, bridge=self.bridge, provider=self.provider, dtype=self.param_dtype
        )

    def _resolve_override_ddp_config(self):
        """Apply optimizer dtype and Dynamic CP gradient-reduction invariants.

        When the precision-aware optimizer is opted into with a sub-fp32
        ``main_grads_dtype``, the DDP grad bucket must reduce grads in the same
        dtype, so inject ``grad_reduce_in_fp32=False`` unless the user set it
        explicitly via ``override_ddp_config``. Default (opt-out) leaves the fp32
        grad bucket untouched, preserving prior behavior.
        """
        from verl.utils.torch_dtypes import PrecisionType

        override_ddp_config = dict(self.engine_config.override_ddp_config or {})
        if self.engine_config.dynamic_context_parallel:
            if override_ddp_config.get("average_in_collective") is True:
                raise ValueError(
                    "Dynamic CP requires override_ddp_config.average_in_collective=False because "
                    "calculate_per_token_loss=True performs token-count normalization after gradient reduction."
                )
            override_ddp_config["average_in_collective"] = False
        opt_cfg = self.optimizer_config
        if (
            opt_cfg is not None
            and getattr(opt_cfg, "use_precision_aware_optimizer", False)
            and PrecisionType.to_dtype(getattr(opt_cfg, "main_grads_dtype", "fp32")) != torch.float32
            and "grad_reduce_in_fp32" not in override_ddp_config
        ):
            override_ddp_config["grad_reduce_in_fp32"] = False
        return override_ddp_config

    def _build_megatron_module(self):
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        if self.engine_config.forward_only:
            wrap_with_ddp = False
        else:
            wrap_with_ddp = True

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=self.is_value_model,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_megatron_fsdp=self.engine_config.use_megatron_fsdp,
        )
        override_ddp_config = self._resolve_override_ddp_config()

        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.model_config.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=self.engine_config.override_mcore_model_config,
            override_ddp_config=override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.model_config.get("lora", None),
        )
        self.tf_config = updated_tf_config
        _validate_resolved_dcp_transformer_config(self.engine_config, self.tf_config)
        print(f"module: {len(module)}")

        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(
                module, self.engine_config.dist_checkpointing_path, is_value_model=self.is_value_model
            )
        else:
            if self.vanilla_bridge:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                allowed_mismatched_params = []
                if self.is_value_model:
                    allowed_mismatched_params = ["output_layer.weight"]
                self.bridge.load_hf_weights(
                    module, self.model_config.local_path, allowed_mismatched_params=allowed_mismatched_params
                )

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        if self.enable_routing_replay:
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module

    def _maybe_enable_fused_kernels(self):
        if not self.engine_config.use_fused_kernels:
            return

        if self.is_value_model or self.model_config.mtp.enable:
            logger.warning_once(
                "Fused kernels are not supported for value models or when MTP is enabled in Megatron engine; disabling."
            )
            self.engine_config.use_fused_kernels = False
            return

        from verl.models.mcore.model_forward_fused import patch_fused_forward

        for model in self.module:
            patch_fused_forward(model)

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
            bf16=self.param_dtype == torch.bfloat16,
        )
        optimizer = get_megatron_optimizer(model=self.module, config=optim_config_megatron)
        register_megatron_training_hooks(self.module, optimizer)
        return optimizer

    def _build_lr_scheduler(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer_param_scheduler

        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.optimizer_config
        )
        return optimizer_scheduler

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )

    def initialize(self):
        self._build_tf_config()

        self.module = self._build_megatron_module()

        if self._qat_enabled and not self.engine_config.forward_only:
            from verl.utils.modelopt import apply_qat_to_modules

            self.module = apply_qat_to_modules(self.module, self._qat_config)

        self._maybe_enable_fused_kernels()

        if self.model_config.mtp.enable:
            patch_engine_mtp(self.module, self.model_config)
        elif (
            self.engine_config.forward_only
            and self.engine_config.override_transformer_config.get("mtp_num_layers") == 0
        ):
            from verl.models.mcore.mtp_patch import patch_postprocess

            for model in self.module:
                patch_postprocess(model)

        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            self.to(device="cpu", model=self._is_offload_param, optimizer=False, grad=False)
            log_gpu_memory_usage("After offload model during init (forward_only)", logger=logger)
            return

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        full_reshardable = self.engine_config.dist_ckpt_optim_fully_reshardable
        mem_eff = self.engine_config.distrib_optim_fully_reshardable_mem_efficient

        tmp_config = OmegaConf.create(
            {
                "model": {"path": self.model_config.local_path},
                "megatron": {
                    "dist_ckpt_optim_fully_reshardable": full_reshardable,
                    "distrib_optim_fully_reshardable_mem_efficient": mem_eff,
                },
            }
        )

        role = "actor" if not self.is_value_model else "critic"

        self.checkpoint_mananager = MegatronCheckpointManager(
            config=tmp_config,
            checkpoint_config=self.checkpoint_config,
            model_config=self.model_config.hf_config,
            transformer_config=self.tf_config,
            role=role,
            model=self.module,
            arch=self.model_config.architectures[0],
            hf_config=self.model_config.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            processing_class=self.model_config.get_processor(),
            optimizer=self.optimizer,
            optimizer_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
            bridge=self.bridge,
            provider=self.provider,
            peft_cls=self.peft_cls,
            use_megatron_fsdp=self.engine_config.use_megatron_fsdp,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        return EngineEvalModeCtx(self, **kwargs)

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        self.optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        # forward_kl_topk leaves large fp32 vocab tensors until backward ends;
        # free cached blocks before grad-norm all_reduce to reduce OOM on tight VRAM.
        if getattr(self, "_distillation_use_topk_active", False):
            get_torch_device().empty_cache()
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        self.lr_scheduler.step(1)
        return get_megatron_last_lr(self.optimizer)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_megatron_model_to_gpu(self.module, load_grad=grad)
            if optimizer and self.optimizer is not None:
                load_megatron_optimizer(self.optimizer)
        elif device == "cpu":
            if model:
                offload_megatron_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_megatron_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def get_data_parallel_rank(self):
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self, with_context_parallel: bool = False):
        return mpu.get_data_parallel_group(with_context_parallel=with_context_parallel)

    def get_model_parallel_group(self):
        return mpu.get_model_parallel_group()

    def get_context_parallel_group(self):
        return mpu.get_context_parallel_group()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        origin_module_device = get_megatron_module_device(self.module)
        if self._is_offload_param or origin_module_device == "cpu":
            load_megatron_model_to_gpu(self.module, load_grad=True)
        self.checkpoint_mananager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: str | None = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

    def _routed_num_tokens(self, data: TensorDict) -> torch.Tensor:
        """Real (unpadded) tokens fed to the MoE router: attention_mask in the padded RL
        path, else the packed input_ids count in the no-padding SFT path. Not loss_mask,
        which counts response tokens only and would under-normalize the router loss."""
        attention_mask = data.get("attention_mask", None)
        if attention_mask is not None:
            return attention_mask.sum()
        input_ids = data["input_ids"]
        if input_ids.is_nested:
            return input_ids.offsets()[-1]
        return torch.tensor(input_ids.numel(), device=input_ids.device)

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        self._distillation_use_topk_active = tu.get_non_tensor_data(data, key="distillation_use_topk", default=False)
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

        # Reject unsupported objectives on every pipeline stage before any
        # collective or model forward. Failing only on the last PP stage can
        # otherwise leave the remaining ranks waiting in Megatron collectives.
        if self.engine_config.dynamic_context_parallel and loss_function is not None:
            validate_dcp_policy_loss(loss_function)
            validate_dcp_loss_normalization(loss_function, data)

        # compute num_tokens in global batch for loss normalization
        normalization_mask = data.get("loss_mask", data.get("response_mask", None))
        if normalization_mask is not None:
            if isinstance(normalization_mask, torch.Tensor) and normalization_mask.is_nested:
                batch_num_tokens = normalization_mask.values().sum().to(get_device_id())
            else:
                batch_num_tokens = normalization_mask.sum().to(get_device_id())
        else:
            batch_num_tokens = torch.tensor(1.0, device=get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        # Verl losses scale partial contributions by logical DP. The legacy
        # callback supplies its CP factor in the pipeline schedule, while the
        # per-token callback supplies it through the returned token counts.
        loss_normalization_world_size = self.get_data_parallel_size()
        tu.assign_non_tensor(data, dp_size=loss_normalization_world_size)

        # Global routed-token count for the per-token-loss regime (consumed in
        # postprocess_micro_batch_func). Real tokens are CP-replicated, so a single
        # all-reduce over the DP group gives the global value.
        if self.tf_config is not None and self.tf_config.calculate_per_token_loss:
            routed_num_tokens = self._routed_num_tokens(data).to(get_device_id())
            torch.distributed.all_reduce(
                routed_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
            tu.assign_non_tensor(data, routed_num_tokens=routed_num_tokens.item())

        # BSHD path only: pad every micro-batch to the mini-batch's global max seq_len so the
        # padded `s_q` is shared -> cuDNN plan built once per shape. Raw (unaligned)
        # max; TP/CP/FP8 alignment is applied inside preprocess_bshd_engine.
        pad_bshd_to_minibatch_max = self.engine_config.pad_bshd_to_minibatch_max
        global_max_seqlen = None
        if pad_bshd_to_minibatch_max and not self.engine_config.use_remove_padding and "input_ids" in data.keys():
            input_ids_for_max = data["input_ids"]
            if input_ids_for_max.is_nested:
                global_max_seqlen = int(input_ids_for_max.offsets().diff().max().item())

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage
        else:
            num_batches_divided_by = None

        dcp_routing_info = None
        if self.engine_config.dynamic_context_parallel:
            dp_group = self.get_data_parallel_group()
            dcp_group = self.get_data_parallel_group(with_context_parallel=True)
            _validate_dcp_multi_modal_inputs(data, dcp_group=dcp_group)
            temperature = _prepare_dcp_temperature(data, dcp_group=dcp_group)
            # Collect non-tensor data to propagate to micro-batches
            non_tensor_data = {}
            for key in [
                "use_dynamic_bsz",
                "use_fused_kernels",
                "calculate_entropy",
                "calculate_sum_pi_squared",
                "pad_mode",
                "sp_size",
                "batch_num_tokens",
                "dp_size",
                "routed_num_tokens",
                "enable_routing_replay",
                "distillation_use_topk",
                "distillation_only",
                "global_batch_size",
                "mini_batch_size",
                "max_response_len",
                "compute_loss",
                "loss_scale_factor",
            ]:
                val = tu.get_non_tensor_data(data, key=key, default=None)
                if val is not None:
                    non_tensor_data[key] = val
            if temperature is not None:
                non_tensor_data["temperature"] = temperature
            non_tensor_data.setdefault("compute_loss", loss_function is not None)
            micro_batches, dcp_routing_info = prepare_micro_batches(
                data=data,
                dp_group=dp_group,
                dynamic_context_parallel=True,
                dcp_group=dcp_group,
                max_seqlen_per_dp_cp_rank=self.engine_config.max_seqlen_per_dp_cp_rank,
                cp_size=mpu.get_context_parallel_world_size(),
                num_batches_divided_by=num_batches_divided_by,
                non_tensor_data=non_tensor_data,
            )
            indices = None
        else:
            micro_batches, indices = prepare_micro_batches(
                data=data,
                dp_group=self.get_data_parallel_group(),
                num_batches_divided_by=num_batches_divided_by,
                same_micro_num_in_dp=True,
                min_num_micro_batch=None,
            )

        if num_batches_divided_by is not None:
            assert len(micro_batches) % num_batches_divided_by == 0, (
                f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                f"{num_batches_divided_by} for megatron backend"
            )

        # Broadcast dynamic CP metadata to middle PP stages
        if dcp_routing_info is not None and mpu.get_pipeline_model_parallel_world_size() > 2:
            from verl.utils.dynamic_cp_scheduler import broadcast_dcp_metadata_to_pp

            micro_batches = broadcast_dcp_metadata_to_pp(micro_batches, mpu.get_pipeline_model_parallel_group())

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)
            if global_max_seqlen is not None:
                tu.assign_non_tensor(micro_batch, forced_max_seqlen=global_max_seqlen)

        forward_backward_func = get_forward_backward_func()

        postprocess_micro_batch_func = partial(
            self.postprocess_micro_batch_func,
            forward_only=forward_only,
            loss_function=loss_function,
        )

        tu.assign_non_tensor(data, num_micro_batch=n_micro_batch)

        forward_step = partial(
            self.forward_step,
            logits_processor_func=loss_function,
            postprocess_micro_batch_func=postprocess_micro_batch_func,
            forward_only=forward_only,
        )

        enable_routing_replay = tu.get_non_tensor_data(data, key="enable_routing_replay", default=False)

        if enable_routing_replay:
            # Set to REPLAY mode: for R3 mode or actor update phase in R2 mode
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            if forward_only and self.engine_config.router_replay.mode == "R2":
                # In R2 mode, forward_only calls (e.g., compute_log_probs) need to record routing information
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.module,
            num_microbatches=n_micro_batch,
            seq_length=1,  # the communication shape is obtained via p2p comm
            micro_batch_size=1,  # the communication shape is obtained via p2p comm
            forward_only=forward_only,
        )

        if self.model_config.mtp.enable and mpu.is_pipeline_last_stage(ignore_virtual=True):
            # All CP ranks must participate in the all_reduce inside get_megatron_mtp_loss,
            # because save_loss_to_tracker uses avg_group=DP+CP group.
            # Only collect metrics on the src rank afterward.
            metrics = get_megatron_mtp_loss(n_micro_batch)
            if self.is_mp_src_rank_with_outputs():
                if "metrics" not in losses_reduced[0]:
                    losses_reduced[0]["metrics"] = {}
                losses_reduced[0]["metrics"].update(metrics)

        dcp_metric_error = None
        if dcp_routing_info is not None and mpu.is_pipeline_last_stage(ignore_virtual=True):
            dcp_metric_error = _aggregate_dcp_metrics_for_logging(
                losses_reduced,
                loss_normalization_world_size,
                self.get_data_parallel_group(with_context_parallel=True),
            )
        if dcp_routing_info is not None:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                _synchronize_dcp_metric_error(dcp_metric_error, self.get_model_parallel_group())
            elif dcp_metric_error is not None:
                raise RuntimeError(dcp_metric_error)

        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                # config = self.actor_module[0].module.module.config
                vp_size = len(self.module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                topk_idx_td = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
                )
            else:
                tensors = [tensor for nt in self.mini_layer_topk_idx_list for tensor in nt.unbind()]
                topk_idx_td = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            self.mini_layer_topk_idx_list = []

            layers_topk_idx = pp_gather(topk_idx_td.to(torch.uint8), self.tf_config)
            use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
            if use_dynamic_bsz and indices is not None:
                layers_topk_idx = restore_dynamic_batch(layers_topk_idx, indices)

        output = {}
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            output = postprocess_batch_func(output_lst=losses_reduced, indices=indices, data=data)
            if dcp_routing_info is not None and output.get("loss"):
                output["loss"] = _aggregate_dcp_loss_for_logging(
                    output["loss"],
                    loss_normalization_world_size,
                    self.get_data_parallel_group(with_context_parallel=True),
                )
            if RouterReplayHelper.is_r2_record_action(self.tf_config):
                output["model_output"]["routed_experts"] = layers_topk_idx
            # Reverse-route outputs back to original DP ranks after dynamic CP scheduling
            if dcp_routing_info is not None:
                from verl.utils.dynamic_cp_scheduler import reverse_route_outputs

                merge_duplicate_gids = bool(output.pop("_dcp_merge_duplicate_gids", False))
                output["model_output"] = reverse_route_outputs(
                    output.get("model_output", {}),
                    dcp_routing_info,
                    self.get_data_parallel_group(),
                    self.get_data_parallel_group(with_context_parallel=True),
                    merge_duplicate_gids=merge_duplicate_gids,
                )
        if enable_routing_replay:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()
        return output

    def get_per_tensor_param(self, base_sync_done=False, **kwargs):
        peft_config = None
        non_merge_lora_sync = self.peft_cls is not None and not self.model_config.lora.get("merge", False)
        adapter_only = base_sync_done and non_merge_lora_sync
        if non_merge_lora_sync:
            peft_config = build_peft_config_for_vllm(self.model_config.lora)
        # when lora adapter only, we only load adapter weights when base sync is done, otherwise load all weights
        load_megatron_model_to_gpu(self.module, load_grad=False, load_frozen_params=not adapter_only)
        if self.vanilla_bridge:
            per_tensor_param = self.bridge.export_weights(self.module)
        elif adapter_only:
            per_tensor_param = self.bridge.export_adapter_weights(self.module)
        else:
            per_tensor_param = (
                self.bridge.export_hf_weights(self.module, merge_adapter_weights=False)
                if non_merge_lora_sync
                else self.bridge.export_hf_weights(self.module)
            )
            if non_merge_lora_sync:
                per_tensor_param = add_base_layer_suffix(
                    per_tensor_param, model_type=self.model_config.hf_config.model_type
                )

        # QAT: process weights through QATWeightExporter for quantized weight sync to vLLM
        if self._qat_enabled:
            from verl.utils.modelopt import export_qat_weights

            per_tensor_param = export_qat_weights(per_tensor_param, self.module, self._qat_config.mode, self.bridge)

        return per_tensor_param, peft_config

    def disable_adapter(self) -> ContextManager:
        return self.peft_cls.disable_adapter(self.module)

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func, forward_only=False):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        raise NotImplementedError("postprocess_micro_batch_func must be implemented in subclass")


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        input_ids = batch["input_ids"]
        loss_mask_val = batch.get("loss_mask", None)
        loss_mask = loss_mask_val.to(bool) if loss_mask_val is not None else None
        multi_modal_inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))

        routed_experts = batch.get("routed_experts", None)

        return {
            "input_ids": input_ids,
            "attention_mask": batch.get("attention_mask", None),
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
            "routed_experts": routed_experts,
        }

    def prepare_model_outputs(self, output: dict, data: TensorDict):
        return output

    def _lm_head_logits_processor(
        self,
        logits,
        label,
        temperature,
        *,
        calculate_sum_pi_squared: bool,
        calculate_entropy: bool,
        distillation_use_topk: bool,
        distillation_only: bool,
        logits_processor_func: Callable,
        batch: TensorDict,
        data_format: str,
    ):
        assert logits.shape[:2] == label.shape[:2]
        # avoid non-positive temperature such as padding
        temperature[temperature <= 0] = 1e-8
        assert torch.all(temperature > 0).item(), f"temperature tensor must be positive. Got {temperature}"
        logits.div_(temperature.unsqueeze(dim=-1).to(logits.dtype))
        ret = {}
        # sum_pi_squared is non-destructive — must run before vocab_parallel_entropy.
        if calculate_sum_pi_squared:
            ret["sum_pi_squared"] = vocab_parallel_sum_pi_squared(logits)
        if calculate_entropy:
            logits_bak = logits.clone()
            # # disable the hint until the fused_kernel is optimized for triton>=3.3
            # if torch.distributed.get_rank() == 0:
            #     logger.warning_once(
            #         "For memory-efficient computation, enable fused kernels via "
            #         "`actor_rollout_ref.model.use_fused_kernels=True`. "
            #         "The current `clone()` operation ensures correctness but increases memory usage."
            #     )
            if self.engine_config.entropy_from_logits_with_chunking:
                entropy = vocab_parallel_entropy_with_chunking(
                    logits,
                    chunk_size=self.engine_config.entropy_from_logits_chunk_size,
                )
            else:
                entropy = vocab_parallel_entropy(logits)

            ret["entropy"] = entropy
        else:
            logits_bak = logits

        # logits_processor_func return tensors with shape (1, total_nnz/cp_size)
        if distillation_use_topk:
            ret.update(logits_processor_func(student_logits=logits_bak, data=batch, data_format=data_format))
        if not distillation_only:
            ret["log_probs"] = vocab_parallel_log_probs_from_logits(logits_bak, label)

        return ret

    def forward_step(
        self,
        batch_iter: Iterator[TensorDict],
        model,
        logits_processor_func,
        postprocess_micro_batch_func,
        forward_only=False,
    ):
        batch: TensorDict = next(batch_iter)

        if self.engine_config.dynamic_context_parallel:
            _already_scheduled = tu.get_non_tensor_data(data=batch, key="local_cp_size", default=None)
            if _already_scheduled is None:
                raise RuntimeError(
                    "Dynamic CP micro-batches must be prepared by DynamicCPScheduler before model forward"
                )

        batch = batch.to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(batch, key="calculate_sum_pi_squared", default=False)
        distillation_use_topk = tu.get_non_tensor_data(batch, key="distillation_use_topk", default=False)
        distillation_only = tu.get_non_tensor_data(batch, key="distillation_only", default=False)
        if distillation_use_topk:
            # compute_topk_loss preprocesses the teacher tensors outside the
            # model forward; record the model's FP8 padding so the teacher THD
            # stream is padded exactly like the student's.
            tu.assign_non_tensor(
                batch,
                _distillation_use_fp8_padding=getattr(self.tf_config, "fp8", None) in ("e4m3", "hybrid"),
            )

        if calculate_sum_pi_squared and use_fused_kernels:
            raise NotImplementedError(
                "calculate_sum_pi_squared=True is not supported with use_fused_kernels=True: "
                "fused kernels do not materialize the full logits tensor needed for Σπ²."
            )
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = tu.get_non_tensor_data(batch, key="temperature", default=None)
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        local_cp_size = tu.get_non_tensor_data(data=batch, key="local_cp_size", default=None)
        is_dcp_micro_batch = self.engine_config.dynamic_context_parallel and local_cp_size is not None
        loss_mask = model_inputs.get("loss_mask", None)
        temperature, fused_temperature = _normalize_temperature_for_thd(
            temperature, input_ids, strict=self.engine_config.dynamic_context_parallel
        )

        unwrapped_model = unwrap_model(model)
        if hasattr(unwrapped_model, "vp_stage"):
            vp_rank = unwrapped_model.vp_stage
        else:
            vp_rank = 0

        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            layers_topk_idx = model_inputs["routed_experts"]
            replay_mask = None
            if self.engine_config.router_replay.mode == "R3":
                replay_mask = build_r3_replay_mask(input_ids, batch["response_mask"])
            set_router_replay_data(
                layers_topk_idx,
                None,
                self.tf_config,
                vp_rank,
                replay_mask=replay_mask,
                local_cp_size=local_cp_size,
            )

        if pad_mode == DatasetPadMode.NO_PADDING:
            label = input_ids.clone()
        else:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

        if use_fused_kernels:
            if not self.engine_config.use_remove_padding:
                logger.warning_once(
                    "Fused kernels require `use_remove_padding=True` for Megatron engine. Falling back to non-fused."
                )
                use_fused_kernels = False
            elif fused_temperature is None:
                logger.warning_once("Fused kernels require one temperature for the whole batch; using non-fused.")
                use_fused_kernels = False

        if use_fused_kernels:
            from verl.models.mcore import get_mcore_forward_fused_model_engine_fn

            fused_forward_fn = get_mcore_forward_fused_model_engine_fn(self.model_config.hf_config)
            output = fused_forward_fn(
                model=model,
                input_ids=input_ids,
                labels=label,
                multi_modal_inputs=multi_modal_inputs,
                temperature=fused_temperature,
                calculate_entropy=calculate_entropy,
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                local_cp_size=local_cp_size,
                compact_dcp_output=is_dcp_micro_batch and forward_only and logits_processor_func is None,
            )
        else:
            from verl.models.mcore import get_mcore_engine_forward_fn

            forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)
            data_format = "thd" if self.engine_config.use_remove_padding else "bshd"

            logits_processor = partial(
                self._lm_head_logits_processor,
                calculate_sum_pi_squared=calculate_sum_pi_squared,
                calculate_entropy=calculate_entropy,
                distillation_use_topk=distillation_use_topk,
                distillation_only=distillation_only,
                logits_processor_func=logits_processor_func,
                batch=batch,
                data_format=data_format,
            )

            response_attention_mask = None
            if attention_mask is not None and loss_mask is not None and not loss_mask.is_nested:
                response_attention_mask = attention_mask[:, -loss_mask.shape[-1] :]
            logits_processor_args = {
                "label": label,
                "temperature": temperature,
                "loss_mask": loss_mask,
                "response_attention_mask": response_attention_mask,
            }

            output = forward_fn(
                model,
                input_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                data_format=data_format,
                mtp_enable_train=self.model_config.mtp.enable and self.model_config.mtp.enable_train,
                local_cp_size=local_cp_size,
                forced_max_seqlen=tu.get_non_tensor_data(data=batch, key="forced_max_seqlen", default=None),
                compact_dcp_output=is_dcp_micro_batch and forward_only and logits_processor_func is None,
            )
        # Router replay: record routing decisions for R2 mode
        if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
            merge_router_topk_indices(
                None,
                input_ids,
                self.mini_layer_topk_idx_list,
                self.tf_config,
                vp_rank,
                local_cp_size=local_cp_size,
            )

        # Router replay: switch to backward replay mode for next backward pass
        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        return output, partial(
            postprocess_micro_batch_func,
            data=batch,
            local_cp_size=local_cp_size,
        )

    def postprocess_micro_batch_func(
        self,
        output,
        data: TensorDict,
        forward_only: bool,
        loss_function,
        local_cp_size=None,
    ):
        # For memory efficiency
        # We move calculation of entropy to compute_log_probs, forward_only == True
        device = data["input_ids"].device
        model_output = self.prepare_model_outputs(output, data)

        if loss_function is not None:
            _apply_dcp_local_token_mask_for_loss(model_output, data)
            # TODO(baiyan): How to support hybrid context parallel with dp_group,
            # now the dp_group is not used, so just leave it as is, but what if we need to use it?
            loss, metrics = call_megatron_loss(
                loss_function,
                model_output=model_output,
                data=data,
                dp_group=self.get_data_parallel_group(),
            )
            # scale loss by num_micro_batch because megatron will scale loss
            # by n_micro_batch inside pp schedule
            scaled_loss = loss * data["num_micro_batch"]
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            scaled_loss = loss
            metrics = {}

        _dcp_scheduled = bool(tu.get_non_tensor_data(data=data, key="_dcp_scheduled", default=False))
        if local_cp_size is not None and not _dcp_scheduled:
            raise RuntimeError("A Dynamic CP micro-batch is missing its scheduler marker")

        output = {
            "loss": loss.detach().item(),
            "metrics": metrics,
        }
        if _dcp_scheduled and "response_mask" in data.keys():
            metric_mask = data["response_mask"]
            metric_weight = metric_mask.values().sum() if metric_mask.is_nested else metric_mask.sum()
            output["_dcp_metric_weight"] = metric_weight.detach().item()
        if forward_only or not _dcp_scheduled:
            output["model_output"] = model_output
        if forward_only and _dcp_scheduled:
            output["_dcp_merge_duplicate_gids"] = True

        # calculate_per_token_loss=True puts Megatron in its per-token regime: loss_func
        # must return (loss_sum, num_tokens,
        # output), and finalize_model_grads divides every gradient by the accumulated
        # total_num_tokens. That division is what cancels the MoE router's pre-multiplication
        # of the aux/z loss by num_tokens; a 2-tuple leaves total_num_tokens=0, so the factor
        # is never cancelled (the ~1e4 grad_norm blow-up at CP>1).
        if self.tf_config is not None and self.tf_config.calculate_per_token_loss and loss_function is not None:
            # Static CP lacks the full-sequence denominator needed by
            # seq-mean-token-mean. Scheduler-managed DCP supplies that denominator
            # through response_token_counts in the Megatron loss adapter.
            if hasattr(loss_function, "keywords") and "config" in loss_function.keywords:
                _agg_mode = getattr(loss_function.keywords["config"], "loss_agg_mode", None)
                if _agg_mode == "seq-mean-token-mean" and not _dcp_scheduled:
                    raise ValueError(
                        "loss_agg_mode='seq-mean-token-mean' is incompatible with "
                        "calculate_per_token_loss=True. The per-sequence inner division by n_s requires "
                        "local-shard counts that diverge from global under CP. Use one "
                        "of: 'token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm'."
                    )
            # verl never passes a router padding_mask, so the MoE router normalizes the
            # aux/z loss by logits.shape[0]. THD packs padding out -> that equals the real
            # token count; BSHD leaves it at B*S (padding-inclusive), while gradients are
            # divided by the real token count -> a padding-ratio mis-normalization.
            if not self.engine_config.use_remove_padding:
                raise ValueError(
                    "calculate_per_token_loss=True requires use_remove_padding=True. "
                    "verl does not pass a padding_mask to the MoE router, so in BSHD it "
                    "normalizes the aux/z loss by the padding-inclusive token count (B*S) "
                    "while gradients are divided by the real token count. Use THD "
                    "(use_remove_padding=True) or disable CP."
                )
            # finalize_model_grads all-reduces the returned token count over the
            # DP*CP group. DCP uses the exact local ownership count because a
            # real sequence length need not divide evenly by its dynamic CP size.
            if _dcp_scheduled:
                local_num_tokens = tu.get_non_tensor_data(data, key="_dcp_local_num_tokens", default=None)
                if local_num_tokens is None:
                    raise ValueError("Scheduler-managed DCP loss is missing its local token ownership count")
                local_num_tokens = torch.as_tensor(local_num_tokens, device=device, dtype=torch.int)
            else:
                cp_size = self.engine_config.context_parallel_size
                if cp_size < 1:
                    raise ValueError(f"Context-parallel size must be positive, got {cp_size}")
                local_num_tokens = (self._routed_num_tokens(data) // cp_size).to(torch.int)
            # n_i is the global routed-token count (all-reduced in forward_backward_batch);
            # scaling loss by the same value makes Sum(L_i)/Sum(n_i) recover the loss. Falls
            # back to local counts when not plumbed (single-rank / tests).
            routed_num_tokens = tu.get_non_tensor_data(data, key="routed_num_tokens", default=None)
            if routed_num_tokens is None:
                routed_num_tokens = self._routed_num_tokens(data)
            dp_size = tu.get_non_tensor_data(data, key="dp_size", default=1)
            local_sum = loss * routed_num_tokens / dp_size
            return local_sum, local_num_tokens, output

        # return loss and stats
        return scaled_loss, output


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngineWithLMHead):
    # for value head
    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func, forward_only=False):
        batch: TensorDict = next(batch_iter)
        if self.engine_config.dynamic_context_parallel:
            _already_scheduled = tu.get_non_tensor_data(data=batch, key="local_cp_size", default=None)
            if _already_scheduled is None:
                raise RuntimeError(
                    "Dynamic CP micro-batches must be prepared by DynamicCPScheduler before model forward"
                )

        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        local_cp_size = tu.get_non_tensor_data(data=batch, key="local_cp_size", default=None)
        is_dcp_micro_batch = self.engine_config.dynamic_context_parallel and local_cp_size is not None

        from verl.models.mcore import get_mcore_engine_forward_fn

        forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            value_model=True,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            data_format="thd" if self.engine_config.use_remove_padding else "bshd",
            forced_max_seqlen=tu.get_non_tensor_data(data=batch, key="forced_max_seqlen", default=None),
            local_cp_size=local_cp_size,
            compact_dcp_output=is_dcp_micro_batch and forward_only and logits_processor_func is None,
        )

        return output, partial(
            postprocess_micro_batch_func,
            data=batch,
            local_cp_size=local_cp_size,
        )

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        if isinstance(output, dict):
            return output
        return {"values": output}

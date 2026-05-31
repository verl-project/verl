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

import os
import random

import numpy as np
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import is_npu_available
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches, restore_dynamic_batch

_DYNAMIC_BATCH_DEBUG_COUNT = 0
_DYNAMIC_BATCH_RESTORE_DEBUG_COUNT = 0


def _rollout_corr_debug_limit() -> int:
    try:
        return max(0, int(os.getenv("VERL_ROLLOUT_CORR_DEBUG_LIMIT", "10")))
    except ValueError:
        return 10


def _dynamic_batch_debug_limit() -> int:
    try:
        return max(0, int(os.getenv("VERL_DYNAMIC_BATCH_DEBUG_LIMIT", "2")))
    except ValueError:
        return 2


def _debug_non_tensor_preview(data: TensorDict, key: str, indices: list[int] | None = None, limit: int = 12):
    try:
        if key not in data.keys():
            return None
        value = data.get(key)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if hasattr(value, "data"):
            value = value.data
        if indices is not None:
            return [str(value[i]) for i in indices[:limit]]
        return [str(v) for v in list(value)[:limit]]
    except Exception as exc:
        return f"<{key}_failed:{type(exc).__name__}>"


def _debug_seq_len_summary(data: TensorDict) -> dict:
    try:
        input_ids = data["input_ids"]
        if getattr(input_ids, "is_nested", False):
            seq_lens = input_ids.offsets().diff().detach().cpu()
        else:
            seq_lens = data["attention_mask"].sum(dim=1).detach().cpu()
        seq_lens_f = seq_lens.float()
        return {
            "count": int(seq_lens.numel()),
            "sum": int(seq_lens.sum().item()),
            "min": int(seq_lens.min().item()) if seq_lens.numel() else 0,
            "max": int(seq_lens.max().item()) if seq_lens.numel() else 0,
            "mean": float(seq_lens_f.mean().item()) if seq_lens.numel() else 0.0,
            "first": [int(x) for x in seq_lens[:12].tolist()],
        }
    except Exception as exc:
        return {"error": type(exc).__name__}


def _debug_dynamic_batch_summary(
    data: TensorDict,
    micro_batches: list[TensorDict],
    batch_idx_list,
    *,
    use_dynamic_bsz: bool,
    max_token_len=None,
    micro_batch_size_per_gpu=None,
) -> None:
    global _DYNAMIC_BATCH_DEBUG_COUNT
    limit = _dynamic_batch_debug_limit()
    if limit <= 0 or _DYNAMIC_BATCH_DEBUG_COUNT >= limit:
        return
    _DYNAMIC_BATCH_DEBUG_COUNT += 1
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
        micro_rows = []
        for idx, micro_batch in enumerate(micro_batches[:16]):
            source_indices = batch_idx_list[idx] if batch_idx_list is not None and idx < len(batch_idx_list) else None
            micro_rows.append(
                {
                    "micro_idx": idx,
                    "rows": len(micro_batch),
                    "source_indices": list(source_indices[:24]) if source_indices is not None else None,
                    "seq_lens": _debug_seq_len_summary(micro_batch),
                    "debug_row_ids": _debug_non_tensor_preview(
                        data if source_indices is not None else micro_batch,
                        "debug_row_id",
                        source_indices if source_indices is not None else None,
                    ),
                }
            )
        flat_indices = [i for part in batch_idx_list for i in part] if batch_idx_list is not None else None
        print(
            "[RolloutCorrDebug][dynamic_batching] "
            f"rank={rank} count={_DYNAMIC_BATCH_DEBUG_COUNT - 1} "
            f"use_dynamic_bsz={use_dynamic_bsz} max_token_len={max_token_len} "
            f"micro_batch_size_per_gpu={micro_batch_size_per_gpu} batch_rows={len(data)} "
            f"num_micro_batches={len(micro_batches)} batch_seq_lens={_debug_seq_len_summary(data)} "
            f"indices_flat_first={flat_indices[:48] if flat_indices is not None else None} "
            f"indices_is_identity={flat_indices == list(range(len(data))) if flat_indices is not None else None} "
            f"batch_debug_row_ids={_debug_non_tensor_preview(data, 'debug_row_id')} "
            f"micro_rows={micro_rows}",
            flush=True,
        )
    except Exception as exc:
        print(f"[RolloutCorrDebug][dynamic_batching] failed={type(exc).__name__}: {exc}", flush=True)


def _debug_dynamic_restore_summary(data: TensorDict, indices, model_output: dict) -> None:
    global _DYNAMIC_BATCH_RESTORE_DEBUG_COUNT
    limit = _dynamic_batch_debug_limit()
    if limit <= 0 or _DYNAMIC_BATCH_RESTORE_DEBUG_COUNT >= limit:
        return
    _DYNAMIC_BATCH_RESTORE_DEBUG_COUNT += 1
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
        flat_indices = [i for part in indices for i in part] if indices is not None else None
        output_summary = {}
        for key, value in model_output.items():
            try:
                if isinstance(value, torch.Tensor):
                    if getattr(value, "is_nested", False):
                        offsets = value.offsets().detach().cpu()
                        output_summary[key] = {
                            "nested": True,
                            "rows": int(value.shape[0]),
                            "seq_lens_first": [int(x) for x in offsets.diff()[:12].tolist()],
                        }
                    else:
                        output_summary[key] = {"shape": tuple(value.shape), "dtype": str(value.dtype)}
                else:
                    output_summary[key] = str(type(value).__name__)
            except Exception as exc:
                output_summary[key] = {"error": type(exc).__name__}
        print(
            "[RolloutCorrDebug][dynamic_batching_restore] "
            f"rank={rank} count={_DYNAMIC_BATCH_RESTORE_DEBUG_COUNT - 1} "
            f"use_dynamic_bsz={tu.get_non_tensor_data(data=data, key='use_dynamic_bsz', default=True)} "
            f"indices_flat_first={flat_indices[:48] if flat_indices is not None else None} "
            f"indices_is_identity={flat_indices == list(range(len(data))) if flat_indices is not None else None} "
            f"batch_debug_row_ids={_debug_non_tensor_preview(data, 'debug_row_id')} "
            f"output_summary={output_summary}",
            flush=True,
        )
    except Exception as exc:
        print(f"[RolloutCorrDebug][dynamic_batching_restore] failed={type(exc).__name__}: {exc}", flush=True)


def enable_full_determinism(seed: int):
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    if is_npu_available:
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ["HCCL_DETERMINISTIC"] = "true"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if is_npu_available:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def prepare_micro_batches(
    data: TensorDict,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    """
    Prepare micro batches from data.
    """
    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    sp_size = tu.get_non_tensor_data(data=data, key="sp_size", default=1)

    force_group_size = tu.get_non_tensor_data(data=data, key="force_group_size", default=1)

    if use_dynamic_bsz:
        assert "max_token_len_per_gpu" in data.keys(), "max_token_len_per_gpu must be set when use_dynamic_bsz is True"
        max_token_len_per_gpu = data["max_token_len_per_gpu"]
        max_token_len = max_token_len_per_gpu * sp_size
        micro_batches, batch_idx_list = rearrange_micro_batches(
            data,
            max_token_len=max_token_len,
            dp_group=dp_group,
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=same_micro_num_in_dp,
            min_num_micro_batch=min_num_micro_batch,
            use_dynamic_bsz_balance=use_dynamic_bsz_balance,
            force_group_size=force_group_size,
        )
        _debug_dynamic_batch_summary(
            data,
            micro_batches,
            batch_idx_list,
            use_dynamic_bsz=True,
            max_token_len=max_token_len,
            micro_batch_size_per_gpu=None,
        )
    else:
        total_data_size = len(data)
        micro_batch_size_per_gpu = data["micro_batch_size_per_gpu"]
        assert total_data_size % (force_group_size * micro_batch_size_per_gpu) == 0, (
            "data size must be divisible by force_group_size * micro_batch_size_per_gpu"
        )
        micro_batches = tu.chunk_tensordict(data, total_data_size // (micro_batch_size_per_gpu * force_group_size))
        batch_idx_list = None
        _debug_dynamic_batch_summary(
            data,
            micro_batches,
            batch_idx_list,
            use_dynamic_bsz=False,
            max_token_len=None,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        )
    return micro_batches, batch_idx_list


def postprocess_batch_func(output_lst, indices, data: TensorDict):
    """postprocess the output of a forward_backward_batch.
    output_lst is a list of dict containing outputs for each micro-batch
    reorder entropy and outputs. Return None for other pp ranks
    only on last rank. It should be on every tp rank

    each losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    """

    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    assert pad_mode == DatasetPadMode.NO_PADDING, "postprocess_batch_func only support NO_PADDING pad_mode"

    # losses_reduced is a list of dict containing outputs for each micro-batch
    # reorder entropy and outputs. Return None for other pp ranks
    # only on last rank. It should be on every tp rank

    # losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    # We perform reverse

    model_output = {}
    losses = []
    aggregated_metrics = {}

    # model output
    for o in output_lst:
        if "model_output" in o:
            for key, val in o["model_output"].items():
                if key not in model_output:
                    model_output[key] = []
                model_output[key].append(val)

    # concat results from micro batches
    for key, val in model_output.items():
        if pad_mode == DatasetPadMode.NO_PADDING:
            tensors = [tensor for nt in model_output[key] for tensor in nt.unbind()]
            model_output[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
        else:
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        # reverse with dynamic bsz
        if use_dynamic_bsz:
            model_output[key] = restore_dynamic_batch(model_output[key], indices)

    _debug_dynamic_restore_summary(data, indices, model_output)

    if "__ref_debug_row_id" in data.keys():
        row_ids = data["__ref_debug_row_id"]
        model_output["__ref_debug_row_id"] = row_ids.detach() if isinstance(row_ids, torch.Tensor) else row_ids
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
            print(
                "[RolloutCorrDebug][ref_row_id_postprocess] "
                f"rank={rank} batch_size={len(row_ids)} "
                f"first_row_ids={row_ids[:5].detach().cpu().tolist()} "
                f"use_dynamic_bsz={use_dynamic_bsz}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[RolloutCorrDebug][ref_row_id_postprocess] print_failed={type(exc).__name__}",
                flush=True,
            )

    # loss
    for o in output_lst:
        if "loss" in o:
            losses.append(o["loss"])

    # metrics
    for o in output_lst:
        if "metrics" in o:
            metrics = o["metrics"]
            append_to_dict(aggregated_metrics, metrics)

    output = {
        "model_output": model_output,
        "loss": losses,
        "metrics": aggregated_metrics,
    }

    return output

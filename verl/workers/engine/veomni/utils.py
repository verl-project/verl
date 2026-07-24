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

import torch

from verl.utils.device import get_device_id, get_torch_device
from verl.workers.engine.utils import _prodshape

VL_TYPE2INDEX = {
    "qwen2_5_vl": {
        "IMAGE_INPUT_INDEX": 151655,
        "VIDEO_INPUT_INDEX": 151656,
    },
    "qwen3_vl": {
        "IMAGE_INPUT_INDEX": 151655,
        "VIDEO_INPUT_INDEX": 151656,
    },
    "qwen3_vl_moe": {
        "IMAGE_INPUT_INDEX": 151655,
        "VIDEO_INPUT_INDEX": 151656,
    },
    "qwen3_5": {
        "IMAGE_INPUT_INDEX": 248056,
        "VIDEO_INPUT_INDEX": 248057,
    },
    "qwen3_5_moe": {
        "IMAGE_INPUT_INDEX": 248056,
        "VIDEO_INPUT_INDEX": 248057,
    },
}


@torch.no_grad()
def offload_veomni_model_to_cpu(model, empty_cache: bool = True):
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None:
            continue
        fsdp_param_group = state._fsdp_param_group

        if fsdp_param_group is None:
            continue

        fsdp_param_group._training_state = TrainingState.IDLE

    model.reshard()
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_veomni_model_to_gpu(model):
    device = get_device_id()
    model.to(device)


@torch.no_grad()
def offload_veomni_optimizer(optimizer):
    optimizers = []
    # Check if this is a MultiOptimizer (for ep and non-ep parameters when ep+fsdp2 is enabled)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    else:
        optimizers.append(optimizer)

    for opt in optimizers:
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_veomni_optimizer(optimizer, device_id):
    optimizers = []
    # Check if this is a MultiOptimizer (for ep and non-ep parameters when ep+fsdp2 is enabled)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    else:
        optimizers.append(optimizer)

    for opt in optimizers:
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device_id, non_blocking=True)


def _map_moe_params_common(name, tensor, expert_id_base):
    for i in range(tensor.size(0)):
        idx = expert_id_base + i
        new_key = name.replace("mlp.experts.", f"mlp.experts.{idx}.") + ".weight"
        yield new_key, tensor[i].to(get_device_id(), non_blocking=True)


def default_moe_param_handler(name, tensor, expert_id_base):
    """Project a stacked expert tensor ``[k, ...]`` (dim 0 = a contiguous run of
    experts) onto per-expert HF-named tensors. ``expert_id_base`` is the global
    expert id of slice 0: an ep rank's local stack passes
    ``ep_rank * experts_per_rank``; an already-global stack passes ``0``."""
    if "gate_up_proj" in name:
        gate, up = tensor.chunk(2, dim=1)
        params = {
            name.replace("gate_up_proj", "gate_proj"): gate,
            name.replace("gate_up_proj", "up_proj"): up,
        }
    else:
        params = {name: tensor}

    for key, value in params.items():
        yield from _map_moe_params_common(key, value, expert_id_base)


# Overrides the default MoE parameter mapping per model_type. Handlers follow the
# ``default_moe_param_handler`` contract: (name, stacked_tensor, expert_id_base).
MOE_PARAM_HANDERS = {}


# ---- EP delta export (veomni-specific converter machinery) ----------------
# The NaN row probe and the converter entry builder for fused expert params;
# the DTensor-generic delta pipeline lives in verl.workers.engine.utils.

NO_SLOTS_MSG = (
    "converter specs without an enumerable slot table (hf_slots) are not supported by "
    "the sender-side HF delta export; rewrite the converter as a dim-0-separable "
    "to_hf_chunk + hf_slots (see #7060)"
)


def convert_row_to_hf(name, spec, r: int, pos_in_row, vals, ref):
    """NaN-probe one dim-0 row through the spec's converter: scatter the row's
    (within-row position, value) pairs into a NaN-filled row buffer, run
    ``to_hf_chunk`` on it, and extract each output slot's surviving positions and
    values (the converter is a pure permutation, so non-NaN survivors are exactly
    the input pairs in final HF coordinates). Returns
    ``[(slot_offset_in_row, idx_int32, val), ...]``, skipping empty slots."""
    full_shape = spec.full_shape
    inner = max(_prodshape(full_shape[1:]), 1)
    slots_per_row = len(spec.hf_slots) // int(full_shape[0])
    buf = torch.full((inner,), float("nan"), dtype=ref.dtype, device=ref.device)
    buf[pos_in_row] = vals
    outs = spec.to_hf_chunk(int(r), buf.view(1, *full_shape[1:]))
    assert len(outs) == slots_per_row, (
        f"{name}: to_hf_chunk gave {len(outs)} outputs/row, slot table expects {slots_per_row}"
    )
    res = []
    for s_i, (_hf_name, hf_tensor) in enumerate(outs):
        fl = hf_tensor.reshape(-1)
        p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        if p_.numel():
            res.append((s_i, p_.to(torch.int32), fl[p_]))
    return res


def hf_entry_converter(name, spec, place, lidx, lval):
    """Turn one converter param's shard-local delta into its final HF-coordinate
    entry ``(slots, dtype_str, counts, idx_concat, val_concat)``: only the touched
    dim-0 rows go through the NaN probe; every rank enumerates the same slot list
    (zero counts when untouched) so the engine's batched gather stays aligned."""
    from verl.workers.engine.spec import translate_flat_indices

    full_shape = spec.full_shape
    inner = max(_prodshape(full_shape[1:]), 1)
    K = len(spec.hf_slots)
    slots_per_row = K // int(full_shape[0])
    counts = torch.zeros(K, dtype=torch.int64)
    idx_pieces: list = []
    val_pieces: list = []
    if lidx.numel():
        g = translate_flat_indices(lidx, place)
        order = torch.argsort(g)
        g, gv = g[order], lval[order]
        rows = torch.div(g, inner, rounding_mode="floor")
        urows, rcounts = torch.unique_consecutive(rows, return_counts=True)
        pos = 0
        for r, cnt in zip(urows.tolist(), rcounts.tolist(), strict=False):
            sel_g = g[pos : pos + cnt]
            sel_v = gv[pos : pos + cnt]
            pos += cnt
            for s_i, pidx, pval in convert_row_to_hf(name, spec, r, sel_g - r * inner, sel_v, lval):
                counts[int(r) * slots_per_row + s_i] = pidx.numel()
                idx_pieces.append(pidx)
                val_pieces.append(pval)
    if idx_pieces:
        my_idx = torch.cat(idx_pieces)
        my_val = torch.cat(val_pieces)
    else:
        my_idx = torch.empty(0, dtype=torch.int32, device=lval.device)
        my_val = torch.empty(0, dtype=lval.dtype, device=lval.device)
    return spec.hf_slots, str(lval.dtype).replace("torch.", ""), counts, my_idx, my_val

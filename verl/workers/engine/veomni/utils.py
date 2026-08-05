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
import json
import os

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


def load_safetensors_index(model_path: str) -> dict[str, int]:
    path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        weight_map = json.load(f)["weight_map"]
    index = {fqn: int(filename.split("-")[1]) for fqn, filename in weight_map.items()}
    return index


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


def passthrough_moe_param_handler(name, tensor, expert_id_base):
    """Keep a fused MoE checkpoint tensor unchanged."""
    del expert_id_base
    yield name, tensor


# Overrides the default MoE parameter mapping per model_type. Handlers follow the
# ``default_moe_param_handler`` contract: (name, stacked_tensor, expert_id_base).
MOE_PARAM_HANDERS = {"gpt_oss": passthrough_moe_param_handler}


def get_moe_param_handler(model_type, ep_enabled):
    """Resolve the checkpoint export handler for the current MoE layout."""
    # GPT-OSS stores all experts in fused tensors with gate/up values interleaved
    # along the last dimension. When EP is disabled, the tensor is already global
    # and vLLM expects this packed checkpoint layout unchanged. EP requires a
    # shard-aware vLLM loader and must not pass local shards under the same name.
    if model_type == "gpt_oss" and ep_enabled:
        raise NotImplementedError("VeOmni GPT-OSS does not support EP weight updates")

    return MOE_PARAM_HANDERS.get(model_type, default_moe_param_handler)


# ---- EP delta export (veomni-specific converter machinery) ----------------
# The NaN row probe and the converter entry builder for fused expert params;
# the DTensor-generic delta pipeline lives in verl.workers.engine.utils.

NO_SLOTS_MSG = (
    "converter specs without an enumerable slot table (hf_slots) are not supported by "
    "the sender-side HF delta export; rewrite the converter as a dim-0-separable "
    "to_hf_chunk + hf_slots (see #7060)"
)


_SLOT_TABLES: dict = {}


def enumerate_hf_slots(process_func, fqn, full_shape, dtype, device=None) -> list:
    """Let the handler itself reveal its slot table: run it once per expert row
    on a zero dummy row and record the ``(hf_name, hf_shape)`` sequence in the
    handler's own output order. Same philosophy as the mcore probe -- the
    converter owns its output structure, nothing is hand-copied -- so ANY
    handler following the ``(name, segment, expert_id_base)`` contract is
    supported, not just the default one.

    The table is static metadata (names/shapes depend on the arguments only,
    never on weight values), so it is enumerated once per (handler, fqn, shape,
    dtype) and cached for the process lifetime -- the export calls this every
    sync. Pass the param's device: the handler moves its outputs to the
    accelerator, so an on-device dummy makes that a no-op view instead of a
    per-row H2D copy."""
    n_experts = int(full_shape[0])
    row_shape = tuple(int(dim) for dim in full_shape[1:])
    cache_key = (process_func, fqn, (n_experts, *row_shape), dtype)
    cached = _SLOT_TABLES.get(cache_key)
    if cached is not None:
        return cached

    dummy_row = torch.zeros((1, *row_shape), dtype=dtype, device=device)
    slots = []
    for expert_id in range(n_experts):
        for hf_name, hf_tensor in process_func(fqn, dummy_row, expert_id_base=expert_id):
            slots.append((hf_name, tuple(hf_tensor.shape)))
    _SLOT_TABLES[cache_key] = slots
    return slots


def convert_row_to_hf(name, spec, expert_id: int, pos_in_row, vals, like):
    """NaN-probe one dim-0 row through the spec's converter: scatter the row's
    (within-row position, value) pairs into a NaN-filled row buffer, run
    ``to_hf_chunk`` on it, and extract each output slot's surviving positions and
    values (the converter is a pure permutation, so non-NaN survivors are exactly
    the input pairs in final HF coordinates). ``like`` only lends its
    dtype/device to the buffer. Returns
    ``[(slot_in_row, idx_int32, val), ...]``, skipping empty slots.

    NaN is the sentinel, so a delta whose NEW value is literally NaN is
    indistinguishable from "untouched" and silently dropped -- acceptable
    because NaN weights are already a fatal training failure, and the
    idempotence verify sweep surfaces the divergence on its next pass."""
    row_shape = spec.full_shape[1:]
    row_numel = max(_prodshape(row_shape), 1)
    slots_per_row = len(spec.hf_slots) // int(spec.full_shape[0])

    row_buf = torch.full((row_numel,), float("nan"), dtype=like.dtype, device=like.device)
    row_buf[pos_in_row] = vals
    hf_outputs = spec.to_hf_chunk(int(expert_id), row_buf.view(1, *row_shape))
    assert len(hf_outputs) == slots_per_row, (
        f"{name}: to_hf_chunk gave {len(hf_outputs)} outputs/row, slot table expects {slots_per_row}"
    )

    converted = []
    for slot_in_row, (_hf_name, hf_tensor) in enumerate(hf_outputs):
        flat = hf_tensor.reshape(-1)
        if flat.device != like.device:
            # handlers move outputs to the accelerator; under CPUOffloadPolicy the
            # shard (and the rest of this entry) lives on CPU -- normalize so the
            # gather queue never cats mixed-device pieces.
            flat = flat.to(like.device)
        survivor_pos = (~torch.isnan(flat)).nonzero(as_tuple=False).view(-1)
        if survivor_pos.numel():
            converted.append((slot_in_row, survivor_pos.to(torch.int32), flat[survivor_pos]))
    return converted


def hf_entry_converter(name, spec, place, lidx, lval):
    """Turn one converter param's shard-local delta into its final HF-coordinate
    entry ``(slots, dtype_str, counts, idx_concat, val_concat)``: only the touched
    dim-0 rows go through the NaN probe; every rank enumerates the same slot list
    (zero counts when untouched) so the engine's batched gather stays aligned."""
    from verl.workers.engine.spec import translate_flat_indices

    row_numel = max(_prodshape(spec.full_shape[1:]), 1)
    n_slots = len(spec.hf_slots)
    slots_per_row = n_slots // int(spec.full_shape[0])
    counts = torch.zeros(n_slots, dtype=torch.int64)
    idx_pieces: list = []
    val_pieces: list = []
    if lidx.numel():
        # shard-local flat positions -> full-tensor flat positions, sorted so
        # every touched expert row becomes one contiguous run
        full_idx = translate_flat_indices(lidx, place)
        order = torch.argsort(full_idx)
        full_idx, full_val = full_idx[order], lval[order]
        row_of = torch.div(full_idx, row_numel, rounding_mode="floor")
        touched_rows, run_lengths = torch.unique_consecutive(row_of, return_counts=True)
        cursor = 0
        for expert_id, run_len in zip(touched_rows.tolist(), run_lengths.tolist(), strict=False):
            run_idx = full_idx[cursor : cursor + run_len]
            run_val = full_val[cursor : cursor + run_len]
            cursor += run_len
            pos_in_row = run_idx - expert_id * row_numel
            for slot_in_row, hf_idx, hf_val in convert_row_to_hf(name, spec, expert_id, pos_in_row, run_val, lval):
                counts[int(expert_id) * slots_per_row + slot_in_row] = hf_idx.numel()
                idx_pieces.append(hf_idx)
                val_pieces.append(hf_val)
    if idx_pieces:
        entry_idx = torch.cat(idx_pieces)
        entry_val = torch.cat(val_pieces)
    else:
        entry_idx = torch.empty(0, dtype=torch.int32, device=lval.device)
        entry_val = torch.empty(0, dtype=lval.dtype, device=lval.device)
    return spec.hf_slots, str(lval.dtype).replace("torch.", ""), counts, entry_idx, entry_val


def veomni_shard_export(module):
    """Build the per-rank shard export ``(gen, None)`` for
    :meth:`VeOmniEngine.get_per_tensor_param_shard`.

    Dense params pass their FSDP DTensor placement through verbatim (flat
    ``Shard(0)`` fast path). Grouped expert params are split twice: the expert
    dim (0) is split *manually* across ``ep_group`` (outside DTensor), and the
    local expert block is FSDP-sharded as a DTensor. That combination is not
    expressible as DTensor placements, so the spec carries an explicit
    :class:`BlockPlacement` in a virtual full tensor of all experts
    (dim-0 offset = ``ep_rank * n_local_experts`` + the DTensor block offset)
    with the world group as the gather group, and a ``to_hf_chunk`` closure over
    the veomni MoE param handler (pure slice + rename: fused -> per-expert HF
    names) plus its ``hf_slots`` output enumeration. ``ep_size=1`` is the same
    geometry with the manual split degenerate; Replicate dims on the block mesh
    (newer veomni HSDPs the expert block) dedupe via ``spec.contributes``.
    """
    import torch.distributed as dist
    from veomni.distributed import parallel_state

    from verl.utils.model import convert_weight_keys

    params = module.state_dict()
    params = convert_weight_keys(params, getattr(module, "_fsdp_wrapped_module", module))

    model_type = getattr(module.config, "model_type", "default")
    if model_type == "gpt_oss":
        raise NotImplementedError("VeOmni GPT-OSS does not support delta_sharded weight updates")

    ps = parallel_state.get_parallel_state()
    process_func = MOE_PARAM_HANDERS.get(model_type, default_moe_param_handler)

    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    from ..spec import BlockPlacement, ShardSpec

    # Structural fact source: ParallelPlan.apply hangs fqn -> SpecInfo on the
    # model itself (it survives state_dict, unlike per-param attributes). Every
    # param gets an entry when EP is on -- non-sliced ones with
    # placement=Replicate() and the SAME para_name -- so the discriminator is
    # the placement type, never ``para_name`` (veomni's own DCP checkpointer
    # keys off ``isinstance(placement, Shard)`` too).
    fqn2si = getattr(module, "_fqn2spec_info", None) or getattr(
        getattr(module, "_fsdp_wrapped_module", module), "_fqn2spec_info", None
    )

    def _is_ep_param(name, param):
        if fqn2si is not None:
            si = fqn2si.get(name)
            if si is not None:
                from torch.distributed.tensor import Shard

                return isinstance(si.placement, Shard)
        # fallback: key conversion may have renamed the fqn; historical name match.
        return "mlp.experts." in name and any(p in name for p in ["down_proj", "gate_proj", "up_proj", "gate_up_proj"])

    def _gen():
        for name, param in params.items():
            # NOT gated on ep_enabled: the fused expert stack exists (and needs
            # the per-expert HF renaming, like the seed path does) even with
            # ep_size=1 -- that is just the degenerate ep_rank=0 geometry.
            is_expert = _is_ep_param(name, param)
            if param.is_floating_point() and param.dtype != torch.bfloat16:
                # mixed-precision keeps fp32 master weights; the wire (and the
                # rollout side) speak bf16, so cast before diffing.
                param = param.to(torch.bfloat16)
            if not is_expert:
                spec = ShardSpec.from_param(param)
                local = param.to_local() if isinstance(param, DTensor) else param
                yield name, local.reshape(-1), spec
                continue

            # local fused block: [n_local_experts, ...] (ep split already applied)
            contributes = True
            if isinstance(param, DTensor):
                lshape, goff = compute_local_shape_and_global_offset(
                    param.shape, param.device_mesh, list(param.placements)
                )
                local = param.to_local()
                # Replicate mesh dims (newer veomni HSDPs the expert block:
                # (ep_replicate, ep_fsdp) mesh) yield identical blocks on every
                # replica -- offsets are already 0 there, so the block math is
                # untouched; only coord-0 ranks contribute, the rest keep
                # lockstep with an empty delta.
                coord = param.device_mesh.get_coordinate()
                rep_dims = [d for d, p in enumerate(param.placements) if p.is_replicate()]
                if coord is not None:
                    contributes = all(int(coord[d]) == 0 for d in rep_dims)
            else:
                lshape, goff = tuple(param.shape), (0,) * param.ndim
                local = param
            ep_size = ps.ep_size if ps.ep_enabled else 1
            ep_rank = ps.ep_rank if ps.ep_enabled else 0
            n_local = int(param.shape[0])
            full_shape = (n_local * ep_size,) + tuple(param.shape[1:])
            offset = (ep_rank * n_local + int(goff[0]),) + tuple(int(o) for o in goff[1:])
            block = BlockPlacement(tuple(int(x) for x in lshape), offset, full_shape)

            # the blocks of one expert tensor are spread over ep x (block mesh,
            # replicas included); v1 requires that to cover the whole trainer
            # world so the world group doubles as the gather group (rank 0 ==
            # engine master). Replicas are deduplicated via ``contributes``.
            mesh_world = param.device_mesh.size() if isinstance(param, DTensor) else 1
            assert ep_size * mesh_world == dist.get_world_size(), (
                f"expert blocks must cover the trainer world: ep={ep_size} x "
                f"block mesh={mesh_world} != world={dist.get_world_size()}"
            )
            spec = ShardSpec(
                full_shape=full_shape,
                place=block,
                gather_group=dist.group.WORLD,
                contributes=contributes,
                # dim-0-separable: a segment is a run of whole experts starting at
                # global expert id ``start`` -- exactly the handler's contract.
                to_hf_chunk=lambda start, seg, _f=process_func, _n=name: list(_f(_n, seg, expert_id_base=start)),
                # slot table enables sender-side conversion; probed once from
                # the handler itself, so any handler is supported.
                hf_slots=enumerate_hf_slots(process_func, name, full_shape, torch.bfloat16, device=local.device),
            )
            yield name, local.reshape(-1), spec

    return _gen(), None

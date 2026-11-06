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
"""Per-rank slicing of a Kimi-K3.1 LoRA adapter for the colocated rollout sync (``VERL_K31_LORA_RANK_SLICE=1``).

Without it every vLLM worker receives the WHOLE adapter (Kimi-K3.1 with expert LoRA: ~49.6 GB, almost all of it the
per-expert factors of 896 experts x 92 layers), copies it, and the Megatron-LM mLite compat layer stages it again
through host memory and ``/dev/shm`` before ``add_lora``.  Each worker keeps only a slice of the expert factors:

* **EP layouts** (``expert_parallel_size > 1``, e.g. TP8+EP8): the worker keeps the expert block ``[r*E/R, (r+1)*E/R)``.
  The mLite exporter already sends only that block, headed by ``EP_TAG = [r, R, E]`` (``lora_expert_slice``, requested
  in ``engine_workers.update_weights``).  vLLM accepts local-sized factors, but mLite compat refuses the
  tag; :func:`localize_received` checks it against this worker's own placement and strips it.
* **TP layouts without EP** (``tp8``): vLLM shards every expert's intermediate dim over TP.  :func:`tp_slice_adapter`
  (send side) narrows ``experts.w{1,3}.lora_B`` (dim 1) and ``experts.w2.lora_A`` (dim 2) to the worker's shard and
  heads the payload with ``TP_TAG = [tp_rank, tp_size]``.  The receiver checks the tag and every sliced shape against
  its ``FusedMoEWithLoRA`` and installs :func:`install_presliced_tp_passthrough`, so vLLM's ``set_lora`` does not slice
  the shard a second time (it would take ``[tp_rank*s:(tp_rank+1)*s]`` of an ``s``-wide tensor: empty for every rank
  but 0, i.e. silently zero expert LoRA).

Shared factors (``[1, ...]``: w1/w3 lora_A, w2 lora_B) and all dense tensors are sent whole either way.
Every mismatch raises: a wrong block or shard would load other experts' (or other ranks') factors without error.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator

import torch

logger = logging.getLogger(__name__)

ENV = "VERL_K31_LORA_RANK_SLICE"
# megatron.lite.model.k31.vllm.checkpoint.LORA_EXPERT_SLICE_TAG (vLLM's own key on builds that have one).
try:  # pragma: no cover - depends on the vLLM build
    from vllm.lora.lora_model import LORA_EXPERT_SLICE_KEY as EP_TAG
except Exception:  # noqa: BLE001
    EP_TAG = "__k31_lora_expert_slice__"
TP_TAG = "__k31_lora_expert_tp_slice__"

# (name suffix, dim sharded over TP by vLLM's FusedMoEWithLoRA: _slice_w13_b / _slice_w2_a)
_TP_SHARDED = (
    (".experts.w1.lora_B.weight", 1),
    (".experts.w3.lora_B.weight", 1),
    (".experts.w2.lora_A.weight", 2),
)


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def tp_sharded_dim(name: str) -> int | None:
    for suffix, dim in _TP_SHARDED:
        if name.endswith(suffix):
            return dim
    return None


# ----------------------------------------------------------------------------------------------- send side (trainer)
def tp_slice_adapter(
    weights: Iterable[tuple[str, torch.Tensor]], tp_rank: int, tp_size: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``TP_TAG`` then the adapter with every TP-sharded expert factor narrowed to ``tp_rank``'s shard.

    The shard is a contiguous copy (the sender packs tensors into flat buckets); everything else passes through.
    """
    if not (0 <= tp_rank < tp_size) or tp_size < 2:
        raise ValueError(f"tp_slice_adapter: bad TP rank {tp_rank}/{tp_size}")
    it = iter(weights)
    first = next(it, None)
    device = first[1].device if first is not None else torch.device("cpu")
    yield TP_TAG, torch.tensor([tp_rank, tp_size], dtype=torch.int64, device=device)
    if first is None:
        return
    for name, t in _chain(first, it):
        dim = tp_sharded_dim(name)
        if dim is not None:
            if t.dim() != 3:
                raise ValueError(f"tp_slice_adapter: {name} is {tuple(t.shape)}, expected a 3D expert factor")
            n = t.shape[dim]
            if n % tp_size:
                raise ValueError(f"tp_slice_adapter: {name} dim {dim} = {n} is not divisible by TP {tp_size}")
            s = n // tp_size
            t = t.narrow(dim, tp_rank * s, s).contiguous()
        yield name, t


def _chain(first, rest):
    yield first
    yield from rest


# ----------------------------------------------------------------------------------------------- receive side (vLLM)
def _moe_lora_modules(worker) -> list:
    mgr = worker.model_runner.lora_manager._adapter_manager
    return [m for m in getattr(mgr, "modules", {}).values() if hasattr(m, "w13_lora_b_stacked")]


def localize_received(weights: dict, worker) -> dict:
    """Check and strip a slice tag (EP or TP) from a received adapter; return it ready for ``add_lora``.

    A payload without a tag is returned unchanged (whole adapter: vLLM slices it itself).
    """
    has_ep, has_tp = EP_TAG in weights, TP_TAG in weights
    if not (has_ep or has_tp):
        return weights
    if has_ep and has_tp:
        raise RuntimeError("K31 adapter carries both an EP and a TP slice tag")
    mgr = worker.model_runner.lora_manager._adapter_manager
    moes = _moe_lora_modules(worker)
    if not moes:
        raise RuntimeError("expert-sliced K31 adapter reached a vLLM worker without MoE LoRA layers")
    if has_ep:
        moe = moes[0]
        placement = getattr(getattr(moe, "base_layer", None), "expert_placement_strategy", None)
        mpc = getattr(getattr(moe, "moe_config", None), "moe_parallel_config", None)
        return localize_ep(
            weights, getattr(mgr, "moe_ep_load_spec", None), placement, bool(getattr(mpc, "enable_eplb", False))
        )
    return localize_tp(weights, moes)


def localize_ep(weights: dict, ep_spec, placement=None, eplb: bool = False) -> dict:
    """EP block ``[r, R, E]``: check it against this worker's expert placement and drop the tag."""
    rank, size, num_experts = (int(v) for v in weights[EP_TAG].detach().to("cpu").reshape(-1).tolist())
    if ep_spec is None:
        raise RuntimeError(
            f"expert-sliced K31 adapter (EP rank {rank}/{size}) reached a vLLM worker without "
            "expert parallelism for its LoRA MoE layers"
        )
    if placement not in (None, "linear") or eplb:
        raise RuntimeError(
            "expert-sliced K31 adapter needs vLLM's linear expert placement without EPLB, got "
            f"placement={placement!r} eplb={eplb}"
        )
    want = (
        ep_spec.ep_rank,
        ep_spec.global_num_experts // max(ep_spec.local_num_experts, 1),
        ep_spec.global_num_experts,
    )
    if (rank, size, num_experts) != want or num_experts % size:
        raise RuntimeError(
            f"expert-sliced K31 adapter for EP rank {rank}/{size} ({num_experts} experts) reached a "
            f"vLLM worker with EP rank {want[0]}/{want[1]} ({want[2]} experts): placement mismatch"
        )
    local = num_experts // size
    out = {}
    for name, t in weights.items():
        if name == EP_TAG:
            continue
        if ".experts." in name and t.dim() == 3 and t.shape[0] not in (1, local):
            raise RuntimeError(
                f"expert-sliced K31 adapter tensor {name} has expert dim {t.shape[0]}, expected 1 "
                f"(shared) or {local} (EP rank {rank}/{size})"
            )
        out[name] = t
    return out


def localize_tp(weights: dict, moes: list) -> dict:
    """TP shard ``[tp_rank, tp_size]``: check against every MoE LoRA layer, install the pass-through, drop the tag."""
    rank, size = (int(v) for v in weights[TP_TAG].detach().to("cpu").reshape(-1).tolist())
    got = {(int(m.tp_rank), int(m.tp_size), int(m._tp_shard_size), bool(m.use_ep)) for m in moes}
    if len(got) != 1:
        raise RuntimeError(f"TP-sliced K31 adapter: MoE LoRA layers disagree on their TP shard: {sorted(got)}")
    w_rank, w_size, shard, use_ep = got.pop()
    if use_ep or (rank, size) != (w_rank, w_size) or size < 2:
        raise RuntimeError(
            f"TP-sliced K31 adapter for TP rank {rank}/{size} reached a vLLM worker whose MoE LoRA "
            f"layers are TP rank {w_rank}/{w_size} (use_ep={use_ep}): trainer/rollout layout mismatch"
        )
    out = {}
    n_sliced = 0
    for name, t in weights.items():
        if name == TP_TAG:
            continue
        dim = tp_sharded_dim(name)
        if dim is not None:
            if t.dim() != 3 or t.shape[dim] != shard:
                raise RuntimeError(
                    f"TP-sliced K31 adapter tensor {name} {tuple(t.shape)}: dim {dim} should be this "
                    f"worker's expert shard {shard} (TP rank {rank}/{size})"
                )
            n_sliced += 1
        out[name] = t
    install_presliced_tp_passthrough()
    logger.info("TP-sliced K31 adapter for TP rank %d/%d (%d expert factors at shard %d)", rank, size, n_sliced, shard)
    return out


def install_presliced_tp_passthrough() -> bool:
    """Make ``FusedMoEWithLoRA``'s TP slicing keep a factor that already has this rank's shard width.

    vLLM shards ``w13_lora_b`` (dim 1) and ``w2_lora_a`` (dim 2) by ``_tp_shard_size``; a whole factor has
    ``_tp_shard_size * tp_size`` there, a pre-sliced one ``_tp_shard_size``: with TP >= 2 the two never coincide, so
    whole adapters (incl. vLLM's warm-up dummy) keep vLLM's own path.  Idempotent.
    """
    from vllm.lora.layers import fused_moe as fm

    changed = False
    for cls_name in ("FusedMoEWithLoRA", "FusedMoE3DWithLoRA"):
        cls = getattr(fm, cls_name, None)
        if cls is None:
            continue
        for meth, dim in (("_slice_w13_b", 1), ("_slice_w2_a", 2)):
            orig = cls.__dict__.get(meth)
            if orig is None or getattr(orig, "_k31_presliced", False):
                continue

            def wrapped(self, t, _orig=orig, _dim=dim):
                if self.tp_size > 1 and t is not None and t.dim() == 3 and t.shape[_dim] == self._tp_shard_size:
                    return t
                return _orig(self, t)

            wrapped._k31_presliced = True
            setattr(cls, meth, wrapped)
            changed = True
    return changed

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
"""Unified benchmark & numerical accuracy suite for `linear_cross_entropy`.

Compares production Eager implementations against fused kernels (`triton`, `tilelang`) across
`TP in {1, 4}` and `topk in {1, 129}` (`1` target + `128` top-k):

- `TP = 1` Eager follows `FSDPEngineWithLMHead.prepare_model_outputs` in
  `verl/workers/engine/fsdp/transformer_impl.py` (`F.linear` + `_scale_logits_by_temperature` +
  `entropy_from_logits` + `topk_log_probs_from_logits` + `logprobs_from_logits`).
- `TP = 4` Eager follows `MegatronEngineWithLMHead._lm_head_logits_processor` in
  `verl/workers/engine/megatron/transformer_impl.py` (`ColumnParallelLinear` +
  `vocab_parallel_entropy` + `topk_log_probs_from_logits` + `vocab_parallel_log_probs_from_logits`).

Usage:
    # Run full TP={1,4}, topk={1,129} suite automatically (spawns workers per TP):
    python tests/utils/kernel/linear_cross_entropy/benchmark_linear_cross_entropy.py

    # Run TP=1 only:
    python tests/utils/kernel/linear_cross_entropy/benchmark_linear_cross_entropy.py --tp 1

    # Run TP=4 via torchrun:
    torchrun --standalone --nproc_per_node=4 \\
        tests/utils/kernel/linear_cross_entropy/benchmark_linear_cross_entropy.py --tp 4
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.util
import os
import tempfile
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    is_device_available,
)
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

GiB = 1024**3


@dataclasses.dataclass
class PerfRecord:
    arch: str
    tp: int
    shape_str: str
    topk: int
    backend_label: str
    oom: bool = False
    skipped: bool = False
    fwd_ms: float = 0.0
    bwd_ce_ms: float = 0.0
    bwd_cee_ms: float = 0.0
    peak_gib: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.fwd_ms + self.bwd_cee_ms


@dataclasses.dataclass
class ErrorRecord:
    arch: str
    tp: int
    shape_str: str
    topk: int
    backend_label: str
    skipped: bool = False
    oom: bool = False
    lp_err: str = "- / -"
    ent_err: str = "- / -"
    dh_err: str = "- / -"
    dw_err: str = "- / -"


def _detect_arch_label(device_name: str, device: torch.device) -> str:
    dev_mod = get_torch_device()
    if device_name == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return f"SM{major}{minor}"
    if device_name == "npu":
        raw_name = dev_mod.get_device_name(device)
        if "950" in raw_name:
            return f"Ascend 950 ({raw_name})"
        if "910" in raw_name:
            return f"Ascend 910 ({raw_name})"
        return raw_name
    return device_name


def _backend_available(backend: str, device_name: str) -> bool:
    if backend == "eager":
        return True
    if backend == "triton":
        return importlib.util.find_spec("triton") is not None and device_name in ("cuda", "npu")
    if backend == "tilelang":
        return importlib.util.find_spec("tilelang") is not None and device_name in ("cuda", "npu")
    return False


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in msg or "alloc" in msg and "memory" in msg


def _format_error_pair(actual: torch.Tensor, ref: torch.Tensor) -> str:
    a64 = actual.detach().float().cpu().double()
    r64 = ref.detach().float().cpu().double()
    diff = (a64 - r64).abs()
    max_abs = diff.max().item()
    denom = torch.linalg.vector_norm(r64).item()
    rel_l2 = torch.linalg.vector_norm(diff).item() / max(denom, 1e-30)
    return f"{max_abs:.2e} / {rel_l2:.2e}"


def _format_tp_sharded_error_pair(
    actual_local: torch.Tensor,
    ref_local: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup],
    device: torch.device,
) -> str:
    a64 = actual_local.detach().float().cpu().double()
    r64 = ref_local.detach().float().cpu().double()
    diff = (a64 - r64).abs()
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        max_abs = diff.max().item()
        rel_l2 = torch.linalg.vector_norm(diff).item() / max(torch.linalg.vector_norm(r64).item(), 1e-30)
        return f"{max_abs:.2e} / {rel_l2:.2e}"

    max_t = torch.tensor([diff.max().item()], device=device, dtype=torch.float32)
    dist.all_reduce(max_t, op=dist.ReduceOp.MAX, group=tp_group)
    sq_t = torch.tensor([(diff**2).sum().item(), (r64**2).sum().item()], device=device, dtype=torch.float32)
    dist.all_reduce(sq_t, op=dist.ReduceOp.SUM, group=tp_group)
    rel_l2 = (sq_t[0].double().sqrt() / sq_t[1].double().clamp_min(1e-30).sqrt()).item()
    return f"{max_t.item():.2e} / {rel_l2:.2e}"


def _max_over_group(val: float, tp_group: Optional[dist.ProcessGroup], device: torch.device) -> float:
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return val
    t = torch.tensor([val], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=tp_group)
    return t.item()


def _sync_all(dev_mod, tp_group: Optional[dist.ProcessGroup]) -> None:
    dev_mod.synchronize()
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.barrier(group=tp_group)
        dev_mod.synchronize()


def _clean_memory(dev_mod) -> None:
    gc.collect()
    dev_mod.empty_cache()


def _chunked_fp32_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    dlp: torch.Tensor,
    dent: torch.Tensor,
    temperature: float,
    tp_group: Optional[dist.ProcessGroup] = None,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact chunked FP32 golden reference for `(logprobs, entropy, d_hidden, d_weight)`."""
    num_tokens, hidden_size = hidden.shape
    local_vocab_size = weight.shape[0]
    rank = 0 if tp_group is None else dist.get_rank(tp_group)
    tp_size = 1 if tp_group is None else dist.get_world_size(tp_group)
    rcp_temp = 1.0 / float(temperature)

    lp_out = torch.empty(labels.shape, device=hidden.device, dtype=torch.float32)
    ent_out = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    dh_out = torch.empty((num_tokens, hidden_size), device=hidden.device, dtype=torch.float32)
    dw_acc = torch.zeros((local_vocab_size, hidden_size), device=hidden.device, dtype=torch.float32)
    w_f = weight.detach().float()

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        h_c = hidden[start:end].detach().float()
        logits_f = F.linear(h_c, w_f)
        if temperature != 1.0:
            logits_f.mul_(rcp_temp)

        lbl_2d = labels[start:end].unsqueeze(-1) if labels.ndim == 1 else labels[start:end]
        dlp_2d = dlp[start:end].unsqueeze(-1) if labels.ndim == 1 else dlp[start:end]

        if tp_size == 1:
            lse = torch.logsumexp(logits_f, dim=-1)
            t_logits = logits_f.gather(-1, lbl_2d)
            lp_c = t_logits - lse.unsqueeze(-1)
            probs = torch.softmax(logits_f, dim=-1)
            mean_logits = (probs * logits_f).sum(dim=-1)
            ent_c = lse - mean_logits

            dl_f = (
                -dlp_2d.float().sum(dim=-1, keepdim=True)
                - dent[start:end].float().unsqueeze(-1) * (logits_f - mean_logits.unsqueeze(-1))
            ) * probs
            dl_f.scatter_add_(1, lbl_2d, dlp_2d.float())
            if temperature != 1.0:
                dl_f.mul_(rcp_temp)
            dh_c = dl_f @ w_f
        else:
            local_max = logits_f.max(dim=-1).values
            global_max = local_max.clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
            exp_l = torch.exp(logits_f - global_max.unsqueeze(-1))
            local_sum = exp_l.sum(dim=-1)
            local_wsum = (exp_l * logits_f).sum(dim=-1)

            local_lbl = lbl_2d - rank * local_vocab_size
            in_shard = (local_lbl >= 0) & (local_lbl < local_vocab_size)
            local_tl = torch.where(in_shard, logits_f.gather(-1, local_lbl.clamp(0, local_vocab_size - 1)), 0.0)

            packed = torch.cat([local_sum.unsqueeze(-1), local_wsum.unsqueeze(-1), local_tl], dim=-1)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=tp_group)
            g_sum, g_wsum, g_tl = packed[:, 0], packed[:, 1], packed[:, 2:]

            lse = global_max + torch.log(g_sum)
            mean_logits = g_wsum / g_sum
            ent_c = lse - mean_logits
            lp_c = g_tl - lse.unsqueeze(-1)

            probs = exp_l / g_sum.unsqueeze(-1)
            dl_f = (
                -dlp_2d.float().sum(dim=-1, keepdim=True)
                - dent[start:end].float().unsqueeze(-1) * (logits_f - mean_logits.unsqueeze(-1))
            ) * probs
            dl_f.scatter_add_(
                1,
                local_lbl.clamp(0, local_vocab_size - 1),
                torch.where(in_shard, dlp_2d.float(), 0.0),
            )
            if temperature != 1.0:
                dl_f.mul_(rcp_temp)
            dh_c = dl_f @ w_f
            dist.all_reduce(dh_c, op=dist.ReduceOp.SUM, group=tp_group)

        if labels.ndim == 1:
            lp_c = lp_c.squeeze(-1)
        lp_out[start:end] = lp_c
        ent_out[start:end] = ent_c
        dh_out[start:end] = dh_c
        dw_acc.add_(dl_f.T @ h_c)
        del logits_f, probs, dl_f, dh_c

    if rank == 0:
        lp_out_cpu, ent_out_cpu, dh_out_cpu = lp_out.cpu(), ent_out.cpu(), dh_out.cpu()
    else:
        lp_out_cpu = ent_out_cpu = dh_out_cpu = torch.empty(0)
    dw_acc_cpu = dw_acc.cpu()
    del lp_out, ent_out, dh_out, dw_acc, w_f
    return lp_out_cpu, ent_out_cpu, dh_out_cpu, dw_acc_cpu


def _build_fsdp_eager_runner(
    device_name: str,
    entropy_with_chunking: bool = False,
    compile_entropy: Optional[bool] = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float, bool], tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Build the exact TP=1 FSDP Eager runner from `verl/workers/engine/fsdp/transformer_impl.py`."""
    import verl.utils.torch_functional as verl_F
    from verl.trainer.ppo.score_centering import topk_log_probs_from_logits
    from verl.workers.engine.fsdp.transformer_impl import _scale_logits_by_temperature

    if compile_entropy is None:
        compile_entropy = device_name == "cuda"

    entropy_fn = verl_F.entropy_from_logits_with_chunking if entropy_with_chunking else verl_F.entropy_from_logits
    compute_entropy_from_logits = torch.compile(entropy_fn, dynamic=True) if compile_entropy else entropy_fn

    def _run(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        calculate_entropy: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits_rmpad = F.linear(hidden, weight)
        temperature_is_one = temperature == 1.0
        temp_t = (
            None
            if temperature_is_one
            else torch.full((hidden.shape[0], 1), temperature, device=hidden.device, dtype=torch.float32)
        )
        logits_rmpad = _scale_logits_by_temperature(
            logits_rmpad,
            temp_t,
            is_unit_temperature=temperature_is_one,
        )

        score_centering = labels.ndim == 2 and labels.shape[-1] > 1
        target_labels = labels[:, 0] if score_centering else labels

        entropy_rmpad = None
        if calculate_entropy:
            if entropy_with_chunking:
                entropy_rmpad = compute_entropy_from_logits(logits_rmpad, chunk_size=2048)
            else:
                entropy_rmpad = compute_entropy_from_logits(logits_rmpad)

        topk_log_probs = None
        if score_centering:
            topk_log_probs = topk_log_probs_from_logits(
                logits_rmpad,
                labels[:, 1:],
                process_group=None,
            )

        inplace_backward = not (calculate_entropy or score_centering)
        log_probs = verl_F.logprobs_from_logits(
            logits=logits_rmpad,
            labels=target_labels,
            inplace_backward=inplace_backward,
        )
        if score_centering:
            log_probs = torch.cat([log_probs.float().unsqueeze(-1), topk_log_probs], dim=-1)
        return log_probs, entropy_rmpad

    return _run


def _build_megatron_eager_runner(
    weight: torch.Tensor,
    global_vocab_size: int,
    tp_group: dist.ProcessGroup,
    entropy_with_chunking: bool = False,
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor, float, bool], tuple[torch.Tensor, Optional[torch.Tensor]]],
    torch.nn.Parameter,
]:
    """Build the exact TP>1 Megatron Eager runner from `verl/workers/engine/megatron/transformer_impl.py`."""
    from megatron.core import ModelParallelConfig
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    from verl.trainer.ppo.score_centering import topk_log_probs_from_logits
    from verl.utils.megatron.tensor_parallel import (
        vocab_parallel_entropy,
        vocab_parallel_entropy_with_chunking,
        vocab_parallel_log_probs_from_logits,
    )

    hidden_size = weight.shape[1]
    config = ModelParallelConfig(
        params_dtype=weight.dtype,
        use_cpu_initialization=False,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
    )
    output_layer = ColumnParallelLinear(
        hidden_size,
        global_vocab_size,
        config=config,
        init_method=torch.nn.init.xavier_normal_,
        bias=False,
        gather_output=False,
        tp_group=tp_group,
    ).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        output_layer.weight.copy_(weight.detach())

    def _run(
        hidden: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        calculate_entropy: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits, _ = output_layer(hidden)
        if temperature != 1.0:
            logits.div_(temperature)

        score_centering = labels.ndim == 2 and labels.shape[-1] > 1
        target_labels = labels[:, 0] if score_centering else labels

        entropy = None
        if calculate_entropy:
            logits_bak = logits.clone()
            if entropy_with_chunking:
                entropy = vocab_parallel_entropy_with_chunking(logits.unsqueeze(0), chunk_size=2048).squeeze(0)
            else:
                entropy = vocab_parallel_entropy(logits)
        else:
            logits_bak = logits

        topk_log_probs = None
        if score_centering:
            topk_log_probs = topk_log_probs_from_logits(
                logits_bak,
                labels[:, 1:],
                process_group=tp_group,
            )

        log_probs = vocab_parallel_log_probs_from_logits(
            logits_bak.clone() if score_centering else logits_bak,
            target_labels,
        )
        if score_centering:
            log_probs = torch.cat([log_probs.float().unsqueeze(-1), topk_log_probs], dim=-1)
        return log_probs, entropy

    return _run, output_layer.weight


def _benchmark_one_backend(
    *,
    arch: str,
    tp: int,
    shape_str: str,
    topk: int,
    backend_label: str,
    fwd_fn: Callable[[bool], tuple[torch.Tensor, Optional[torch.Tensor]]],
    bwd_params: tuple[torch.Tensor, torch.Tensor],
    post_bwd_hook: Optional[Callable[[torch.Tensor], None]],
    dlp: torch.Tensor,
    dent: torch.Tensor,
    ref_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    warmup: int,
    iters: int,
    dev_mod,
    device: torch.device,
    tp_group: Optional[dist.ProcessGroup],
) -> tuple[PerfRecord, ErrorRecord]:
    perf = PerfRecord(arch=arch, tp=tp, shape_str=shape_str, topk=topk, backend_label=backend_label)
    err = ErrorRecord(arch=arch, tp=tp, shape_str=shape_str, topk=topk, backend_label=backend_label)

    _clean_memory(dev_mod)
    dev_mod.reset_peak_memory_stats(device)
    oom_local = 0
    lp_cand = ent_cand = dh_cand = dw_cand = None
    try:
        # 1. Measure Peak Memory & capture outputs/grads on a clean full step (Fwd + Bwd CE+Ent)
        lp, ent = fwd_fn(True)
        dh, dw = torch.autograd.grad((lp, ent), bwd_params, (dlp.to(lp.dtype), dent.to(ent.dtype)))
        if post_bwd_hook is not None:
            post_bwd_hook(dh)
        dev_mod.synchronize()
        local_peak_gib = dev_mod.max_memory_allocated(device) / GiB
        lp_cand = lp.detach().float().cpu()
        ent_cand = ent.detach().float().cpu()
        dh_cand = dh.detach().float().cpu()
        dw_cand = dw.detach().float().cpu()
        del lp, ent, dh, dw
        _clean_memory(dev_mod)
    except RuntimeError as exc:
        if _is_oom_error(exc):
            oom_local = 1
        else:
            raise

    if _max_over_group(float(oom_local), tp_group, device) > 0.0:
        _clean_memory(dev_mod)
        perf.oom = True
        err.oom = True
        return perf, err

    perf.peak_gib = _max_over_group(local_peak_gib, tp_group, device)

    lp_ref, ent_ref, dh_ref, dw_ref = ref_tensors
    if tp_group is None or dist.get_rank(tp_group) == 0:
        err.lp_err = _format_error_pair(lp_cand, lp_ref)
        err.ent_err = _format_error_pair(ent_cand, ent_ref)
        err.dh_err = _format_error_pair(dh_cand, dh_ref)
    err.dw_err = _format_tp_sharded_error_pair(dw_cand, dw_ref, tp_group, device)
    del lp_cand, ent_cand, dh_cand, dw_cand

    try:
        # 2. Warmup full Fwd + Bwd CE+Ent
        for _ in range(warmup):
            lp, ent = fwd_fn(True)
            dh, dw = torch.autograd.grad((lp, ent), bwd_params, (dlp.to(lp.dtype), dent.to(ent.dtype)))
            if post_bwd_hook is not None:
                post_bwd_hook(dh)
            del lp, ent, dh, dw
        _sync_all(dev_mod, tp_group)

        # 3. Time Forward (`calculate_entropy=True`)
        start_ev, end_ev = dev_mod.Event(enable_timing=True), dev_mod.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iters):
            lp, ent = fwd_fn(True)
            del lp, ent
        end_ev.record()
        dev_mod.synchronize()
        perf.fwd_ms = _max_over_group(start_ev.elapsed_time(end_ev) / iters, tp_group, device)

        # 4. Time Backward CE only (`calculate_entropy=False`, `retain_graph=False` per step)
        bwd_ce_samples: list[float] = []
        for step in range(warmup + iters):
            lp, _ = fwd_fn(False)
            dlp_cast = dlp.to(lp.dtype)
            _sync_all(dev_mod, tp_group)
            start_ev.record()
            dh, dw = torch.autograd.grad((lp,), bwd_params, (dlp_cast,))
            if post_bwd_hook is not None:
                post_bwd_hook(dh)
            end_ev.record()
            dev_mod.synchronize()
            if step >= warmup:
                bwd_ce_samples.append(start_ev.elapsed_time(end_ev))
            del lp, dh, dw
        perf.bwd_ce_ms = _max_over_group(sum(bwd_ce_samples) / iters, tp_group, device)
        _clean_memory(dev_mod)

        # 5. Time Backward CE + Entropy (`calculate_entropy=True`, `retain_graph=False` per step)
        bwd_cee_samples: list[float] = []
        for step in range(warmup + iters):
            lp, ent = fwd_fn(True)
            dlp_cast = dlp.to(lp.dtype)
            dent_cast = dent.to(ent.dtype)
            _sync_all(dev_mod, tp_group)
            start_ev.record()
            dh, dw = torch.autograd.grad((lp, ent), bwd_params, (dlp_cast, dent_cast))
            if post_bwd_hook is not None:
                post_bwd_hook(dh)
            end_ev.record()
            dev_mod.synchronize()
            if step >= warmup:
                bwd_cee_samples.append(start_ev.elapsed_time(end_ev))
            del lp, ent, dh, dw
        perf.bwd_cee_ms = _max_over_group(sum(bwd_cee_samples) / iters, tp_group, device)
    except RuntimeError as exc:
        if _is_oom_error(exc):
            oom_local = 1
        else:
            raise

    if _max_over_group(float(oom_local), tp_group, device) > 0.0:
        perf.oom = True
    _clean_memory(dev_mod)
    return perf, err


def _print_markdown_tables(perf_records: list[PerfRecord], err_records: list[ErrorRecord]) -> None:
    eager_totals: dict[tuple[int, int], PerfRecord] = {
        (r.tp, r.topk): r for r in perf_records if r.backend_label.startswith("Eager")
    }

    print("\n### Performance & Memory Benchmark\n", flush=True)
    print(
        "| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend "
        "| Fwd (ms) | Bwd CE (ms) | Bwd CE+Ent (ms) | Total (ms) | Peak Mem (GiB) | Speedup vs Eager |",
        flush=True,
    )
    print("| :--- | :-: | :--- | :-: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)
    for r in perf_records:
        if r.skipped:
            print(
                f"| **{r.arch}** | {r.tp} | `{r.shape_str}` | {r.topk} | {r.backend_label} | - | - | - | - | - | - |",
                flush=True,
            )
            continue
        if r.oom:
            print(
                f"| **{r.arch}** | {r.tp} | `{r.shape_str}` | {r.topk} | {r.backend_label} "
                "| OOM | OOM | OOM | OOM | OOM | - |",
                flush=True,
            )
            continue
        eager_r = eager_totals.get((r.tp, r.topk))
        if r.backend_label.startswith("Eager"):
            speedup_str = "1.00x"
        elif eager_r is None or eager_r.skipped:
            speedup_str = "-"
        elif eager_r.oom:
            speedup_str = "Eager OOM"
        else:
            speedup_str = f"{eager_r.total_ms / r.total_ms:.2f}x"
        print(
            f"| **{r.arch}** | {r.tp} | `{r.shape_str}` | {r.topk} | {r.backend_label} "
            f"| {r.fwd_ms:.2f} | {r.bwd_ce_ms:.2f} | {r.bwd_cee_ms:.2f} "
            f"| {r.total_ms:.2f} | {r.peak_gib:.2f} | {speedup_str} |",
            flush=True,
        )

    print("\n### Numerical Error vs Chunked FP32 Reference (`MaxAbs / RelL2`)\n", flush=True)
    print(
        "| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend "
        "| `logprobs` MaxAbs / RelL2 | `entropy` MaxAbs / RelL2 "
        "| `d_hidden` MaxAbs / RelL2 | `d_weight` MaxAbs / RelL2 |",
        flush=True,
    )
    print("| :--- | :-: | :--- | :-: | :--- | :---: | :---: | :---: | :---: |", flush=True)
    for e in err_records:
        if e.skipped:
            print(
                f"| **{e.arch}** | {e.tp} | `{e.shape_str}` | {e.topk} | {e.backend_label} "
                "| - / - | - / - | - / - | - / - |",
                flush=True,
            )
        elif e.oom:
            print(
                f"| **{e.arch}** | {e.tp} | `{e.shape_str}` | {e.topk} | {e.backend_label} | OOM | OOM | OOM | OOM |",
                flush=True,
            )
        else:
            print(
                f"| **{e.arch}** | {e.tp} | `{e.shape_str}` | {e.topk} | {e.backend_label} "
                f"| `{e.lp_err}` | `{e.ent_err}` | `{e.dh_err}` | `{e.dw_err}` |",
                flush=True,
            )


def _run_tp_worker(
    rank: int,
    tp_size: int,
    init_file: Optional[str],
    args: argparse.Namespace,
    result_file: Optional[str] = None,
) -> tuple[list[PerfRecord], list[ErrorRecord]]:
    device_name = get_device_name()
    if device_name == "npu":
        import megatron_adaptor  # noqa: F401

    dev_mod = get_torch_device()
    local_rank = int(os.environ.get("LOCAL_RANK", rank)) if init_file is None and tp_size > 1 else rank
    device = torch.device(f"{device_name}:{local_rank}")
    dev_mod.set_device(device)

    owns_pg = False
    tp_group: Optional[dist.ProcessGroup] = None
    if tp_size > 1:
        if not dist.is_initialized():
            if init_file is not None:
                dist.init_process_group(
                    backend=get_nccl_backend(),
                    init_method=f"file://{init_file}",
                    rank=rank,
                    world_size=tp_size,
                )
            else:
                dist.init_process_group(backend=get_nccl_backend())
                rank = dist.get_rank()
            owns_pg = True
        from megatron.core import parallel_state as mpu

        mpu.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        tp_group = mpu.get_tensor_model_parallel_group()

    arch = _detect_arch_label(device_name, device)
    assert args.vocab_size % tp_size == 0, f"vocab_size ({args.vocab_size}) must be divisible by tp ({tp_size})"
    local_vocab_size = args.vocab_size // tp_size
    shape_str = f"({args.num_tokens}, {local_vocab_size}, {args.hidden_size})"

    torch.manual_seed(args.seed)
    hidden = (torch.randn(args.num_tokens, args.hidden_size, device=device, dtype=torch.bfloat16) * 0.1).detach()
    if tp_group is not None:
        dist.broadcast(hidden, src=0, group=tp_group)
    hidden.requires_grad_(True)

    torch.manual_seed(args.seed + 1000 + rank)
    weight = (torch.randn(local_vocab_size, args.hidden_size, device=device, dtype=torch.bfloat16) * 0.1).detach()
    weight.requires_grad_(True)

    fsdp_runner = None
    megatron_runner = None
    megatron_weight = None
    if "eager" in args.backends:
        if tp_size == 1:
            fsdp_runner = _build_fsdp_eager_runner(
                device_name=device_name,
                entropy_with_chunking=args.entropy_with_chunking,
                compile_entropy=args.compile_entropy,
            )
        else:
            megatron_runner, megatron_weight = _build_megatron_eager_runner(
                weight=weight,
                global_vocab_size=args.vocab_size,
                tp_group=tp_group,
                entropy_with_chunking=args.entropy_with_chunking,
            )

    perf_records: list[PerfRecord] = []
    err_records: list[ErrorRecord] = []

    backend_display = {
        "eager": "Eager (FSDP)" if tp_size == 1 else "Eager (Megatron)",
        "triton": "Triton",
        "tilelang": "TileLang",
    }

    for topk in args.topk:
        if rank == 0:
            print(f"\n[Benchmark] Arch={arch} | TP={tp_size} | Shape={shape_str} | topk={topk}", flush=True)

        torch.manual_seed(args.seed + 200 + topk)
        label_shape = (args.num_tokens,) if topk == 1 else (args.num_tokens, topk)
        labels = torch.randint(0, args.vocab_size, label_shape, device=device, dtype=torch.int64)
        dlp = torch.randn(label_shape, device=device, dtype=torch.float32) * 0.1
        dent = torch.randn((args.num_tokens,), device=device, dtype=torch.float32) * 0.1
        if tp_group is not None:
            dist.broadcast(labels, src=0, group=tp_group)
            dist.broadcast(dlp, src=0, group=tp_group)
            dist.broadcast(dent, src=0, group=tp_group)

        ref_tensors = _chunked_fp32_reference(
            hidden=hidden,
            weight=weight,
            labels=labels,
            dlp=dlp,
            dent=dent,
            temperature=args.temperature,
            tp_group=tp_group,
            chunk_size=args.ref_chunk_size,
        )
        _clean_memory(dev_mod)

        for backend in args.backends:
            label_str = backend_display[backend]
            if not _backend_available(backend, device_name):
                perf_records.append(
                    PerfRecord(
                        arch=arch, tp=tp_size, shape_str=shape_str, topk=topk, backend_label=label_str, skipped=True
                    )
                )
                err_records.append(
                    ErrorRecord(
                        arch=arch, tp=tp_size, shape_str=shape_str, topk=topk, backend_label=label_str, skipped=True
                    )
                )
                continue

            if backend == "eager":
                if tp_size == 1:
                    fwd_fn = lambda calc_ent, _lbl=labels: fsdp_runner(hidden, weight, _lbl, args.temperature, calc_ent)
                    bwd_params = (hidden, weight)
                    post_bwd_hook = None
                else:
                    fwd_fn = lambda calc_ent, _lbl=labels: megatron_runner(hidden, _lbl, args.temperature, calc_ent)
                    bwd_params = (hidden, megatron_weight)
                    post_bwd_hook = None
            else:
                fwd_fn = lambda calc_ent, _b=backend, _lbl=labels: linear_cross_entropy(
                    hidden,
                    weight,
                    _lbl,
                    args.temperature,
                    "none",
                    dist_process_group=tp_group,
                    impl_backend=_b,
                )
                bwd_params = (hidden, weight)
                post_bwd_hook = (
                    (lambda dh: dist.all_reduce(dh, op=dist.ReduceOp.SUM, group=tp_group))
                    if tp_group is not None
                    else None
                )

            perf_rec, err_rec = _benchmark_one_backend(
                arch=arch,
                tp=tp_size,
                shape_str=shape_str,
                topk=topk,
                backend_label=label_str,
                fwd_fn=fwd_fn,
                bwd_params=bwd_params,
                post_bwd_hook=post_bwd_hook,
                dlp=dlp,
                dent=dent,
                ref_tensors=ref_tensors,
                warmup=args.warmup,
                iters=args.iters,
                dev_mod=dev_mod,
                device=device,
                tp_group=tp_group,
            )
            perf_records.append(perf_rec)
            err_records.append(err_rec)

            if rank == 0:
                if perf_rec.oom:
                    print(f"  -> {label_str}: OOM", flush=True)
                else:
                    print(
                        f"  -> {label_str}: Fwd={perf_rec.fwd_ms:.2f} ms, Bwd_CE={perf_rec.bwd_ce_ms:.2f} ms, "
                        f"Bwd_CE+Ent={perf_rec.bwd_cee_ms:.2f} ms, Total={perf_rec.total_ms:.2f} ms, "
                        f"PeakMem={perf_rec.peak_gib:.2f} GiB | lp={err_rec.lp_err}, ent={err_rec.ent_err}",
                        flush=True,
                    )

        del ref_tensors, labels, dlp, dent
        _clean_memory(dev_mod)

    if rank == 0 and result_file is not None:
        torch.save((perf_records, err_records), result_file)

    if tp_size > 1:
        from megatron.core import parallel_state as mpu

        mpu.destroy_model_parallel()
        if owns_pg and dist.is_initialized():
            dist.destroy_process_group()

    return perf_records, err_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark & verify linear_cross_entropy across TP and topk.")
    parser.add_argument("--num-tokens", type=int, default=65536, help="Number of tokens M (default: 65536).")
    parser.add_argument("--vocab-size", type=int, default=129280, help="Global vocabulary size V (default: 129280).")
    parser.add_argument("--hidden-size", type=int, default=5120, help="Hidden dimension H (default: 5120).")
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 4], help="TP sizes to benchmark (default: 1 4).")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 129], help="Query topk list (default: 1 129).")
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["eager", "triton", "tilelang"],
        choices=["eager", "triton", "tilelang"],
        help="Backends to benchmark (default: eager triton tilelang).",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (default: 1.0).")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10).")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations (default: 20).")
    parser.add_argument(
        "--ref-chunk-size", type=int, default=2048, help="Chunk size for FP32 reference (default: 2048)."
    )
    parser.add_argument("--entropy-with-chunking", action="store_true", help="Enable chunked entropy in Eager.")
    parser.add_argument(
        "--compile-entropy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use torch.compile on FSDP Eager entropy (default: True on CUDA, False on NPU).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device_name = get_device_name()
    if device_name not in ("cuda", "npu") or not is_device_available():
        raise RuntimeError(f"A supported accelerator (CUDA or Ascend NPU) is required, got device={device_name!r}.")

    # Mode 1: Launched under torchrun (`WORLD_SIZE` set in environment)
    if "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if len(args.tp) == 1 and args.tp[0] != world_size:
            raise ValueError(f"--tp {args.tp[0]} does not match torchrun WORLD_SIZE={world_size}.")
        perf_recs, err_recs = _run_tp_worker(
            rank=local_rank,
            tp_size=world_size,
            init_file=None,
            args=args,
            result_file=None,
        )
        if local_rank == 0:
            _print_markdown_tables(perf_recs, err_recs)
        return

    # Mode 2: Standalone launcher — run each TP group in isolated spawned worker(s)
    dev_count = get_torch_device().device_count()
    all_perf: list[PerfRecord] = []
    all_err: list[ErrorRecord] = []

    for tp_size in args.tp:
        if tp_size > dev_count:
            print(f"[Skip] TP={tp_size} requires {tp_size} devices, but only {dev_count} available.", flush=True)
            continue
        with tempfile.NamedTemporaryFile(delete=False) as init_f, tempfile.NamedTemporaryFile(delete=False) as res_f:
            init_path, res_path = init_f.name, res_f.name
        try:
            mp.spawn(
                _run_tp_worker,
                args=(tp_size, init_path, args, res_path),
                nprocs=tp_size,
                join=True,
            )
            perf_recs, err_recs = torch.load(res_path, weights_only=False)
            all_perf.extend(perf_recs)
            all_err.extend(err_recs)
        finally:
            for p in (init_path, res_path):
                if os.path.exists(p):
                    os.remove(p)

    if all_perf:
        _print_markdown_tables(all_perf, all_err)


if __name__ == "__main__":
    main()

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
"""E2E test for Megatron ``suspend_via_parallel_state`` / ``resume_via_parallel_state``.

Initializes Megatron parallel_state, warms each group with a representative
collective, then verifies: discovery is non-empty, suspend frees driver
memory, suspend is idempotent, resume reclaims within 5%, post-resume
collectives still work. Skips on NCCL < 2.29.7 or missing megatron-core.

Requires >= 4 GPUs (default TP=2 PP=2). Env vars: TP_SIZE, PP_SIZE, CP_SIZE,
EP_SIZE, ETP_SIZE.

Usage:
    NCCL_NVLS_ENABLE=0 torchrun --nproc-per-node=4 --standalone \\
        tests/special_distributed/test_nccl_suspend_megatron.py

    EP_SIZE=2 ETP_SIZE=2 NCCL_NVLS_ENABLE=0 torchrun --nproc-per-node=8 --standalone \\
        tests/special_distributed/test_nccl_suspend_megatron.py

RFC: https://github.com/verl-project/verl/issues/6266
"""

import os

os.environ.setdefault("NCCL_DEBUG", "WARN")

import sys

import torch
import torch.distributed as dist


def _log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[test_nccl_suspend_megatron] {msg}", flush=True)


def _gpu_used_mb() -> float:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


# Per-group canonical collective. NCCL allocates differently-sized channel
# buffers depending on the first collective fired, so picking the dominant
# production op makes the warmed state representative of real training.
_GROUP_TO_WARMUP_OP = {
    "tp": "allreduce",
    "dp": "allreduce",
    "pp": "p2p",
    "cp": "allgather",
    "model_parallel": "allreduce",
    "tp_dp": "allreduce",
    "tp_cp": "allgather",
    "embedding": "allreduce",
    "position_embedding": "allreduce",
    "ep": "all_to_all",
    "etp_ep": "all_to_all",
    "etp_ep_pp": "all_to_all",
    "etp": "allreduce",
    "edp": "allreduce",
    "intra_dist_opt": "allreduce",
}


def _discover_megatron_groups(ps) -> dict:
    """Return {short_name: ProcessGroup} for groups present in the current
    parallel layout. Accessors that aren't applicable raise; we skip those."""
    accessors = [
        ("tp", ps.get_tensor_model_parallel_group),
        ("dp", ps.get_data_parallel_group),
        ("pp", ps.get_pipeline_model_parallel_group),
        ("cp", ps.get_context_parallel_group),
        ("model_parallel", ps.get_model_parallel_group),
        ("tp_dp", ps.get_tensor_and_data_parallel_group),
        ("tp_cp", ps.get_tensor_and_context_parallel_group),
        ("embedding", ps.get_embedding_group),
        ("position_embedding", ps.get_position_embedding_group),
        ("ep", ps.get_expert_model_parallel_group),
        ("etp", ps.get_expert_tensor_parallel_group),
        ("edp", ps.get_expert_data_parallel_group),
        ("etp_ep", ps.get_expert_tensor_and_model_parallel_group),
        ("etp_ep_pp", ps.get_expert_tensor_model_pipeline_parallel_group),
        ("intra_dist_opt", ps.get_intra_distributed_optimizer_instance_group),
    ]
    found: dict = {}
    for name, fn in accessors:
        try:
            g = fn()
        except Exception:
            continue
        if g is not None:
            found[name] = g
    return found


def _run_warmup_op(op: str, group, world: int) -> None:
    if op == "all_to_all":
        chunk = 256 * 1024
        inp = torch.zeros(chunk * world, dtype=torch.float32, device="cuda")
        out = torch.zeros_like(inp)
        dist.all_to_all_single(out, inp, group=group)
    elif op == "allgather":
        chunk = 256 * 1024
        inp = torch.zeros(chunk, dtype=torch.float32, device="cuda")
        out = torch.zeros(chunk * world, dtype=torch.float32, device="cuda")
        dist.all_gather_into_tensor(out, inp, group=group)
    elif op == "p2p":
        # Two stages: leading broadcast warms the main PG ncclComm_t that the
        # reflection picks up; isend/irecv ring exercises the realistic PP
        # path (hidden 2-rank P2P comms are a known coverage gap).
        x = torch.zeros(256 * 1024, dtype=torch.float32, device="cuda")
        src = dist.get_global_rank(group, 0)
        dist.broadcast(x, src=src, group=group)
        local = dist.get_rank(group=group)
        nxt = dist.get_global_rank(group, (local + 1) % world)
        prv = dist.get_global_rank(group, (local - 1) % world)
        send_buf = torch.zeros(256 * 1024, dtype=torch.float32, device="cuda")
        recv_buf = torch.zeros_like(send_buf)
        ops = [
            dist.P2POp(dist.isend, send_buf, nxt, group=group),
            dist.P2POp(dist.irecv, recv_buf, prv, group=group),
        ]
        for r in dist.batch_isend_irecv(ops):
            r.wait()
    else:  # allreduce, broadcast fallback
        x = torch.zeros(256 * 1024, dtype=torch.float32, device="cuda")
        dist.all_reduce(x, group=group)


def _warm_up_groups(groups: dict) -> dict:
    """Fire each group's canonical op and return {name: (world, op)} for those warmed."""
    warmed: dict = {}
    for name, group in groups.items():
        try:
            world = dist.get_world_size(group=group)
        except Exception:
            continue
        if world <= 1:
            continue
        op = _GROUP_TO_WARMUP_OP.get(name, "allreduce")
        try:
            _run_warmup_op(op, group, world)
            torch.cuda.synchronize()
        except Exception as e:
            _log(f"warm-up {name!r} (op={op}) failed: {e}")
            continue
        warmed[name] = (world, op)
    return warmed


def test_nccl_suspend_resume_via_parallel_state() -> None:
    from verl.utils.nccl_suspend import is_supported
    from verl.workers.engine.megatron.utils import (
        _collect_megatron_comms,
        resume_via_parallel_state,
        suspend_via_parallel_state,
    )

    if not is_supported():
        _log("SKIP: libnccl.so.2 is missing ncclCommSuspend (need NCCL >= 2.29.7)")
        return

    try:
        from megatron.core import parallel_state as ps
    except ImportError as e:
        _log(f"SKIP: megatron.core not importable: {e}")
        return

    world_size = int(os.environ["WORLD_SIZE"])
    tp_size = int(os.environ.get("TP_SIZE", "2"))
    pp_size = int(os.environ.get("PP_SIZE", "2"))
    cp_size = int(os.environ.get("CP_SIZE", "1"))
    ep_size = int(os.environ.get("EP_SIZE", "1"))
    etp_size = int(os.environ.get("ETP_SIZE", str(tp_size)))
    expected_dp = world_size // (tp_size * pp_size * cp_size)

    _log(
        f"world_size={world_size}, TP={tp_size}, PP={pp_size}, "
        f"CP={cp_size}, DP={expected_dp}, EP={ep_size}, ETP={etp_size}"
    )

    init_kwargs = {
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_size": pp_size,
        "context_parallel_size": cp_size,
    }
    if ep_size > 1:
        init_kwargs["expert_model_parallel_size"] = ep_size
        init_kwargs["expert_tensor_parallel_size"] = etp_size
    ps.initialize_model_parallel(**init_kwargs)

    discovered = _discover_megatron_groups(ps)
    warmed = _warm_up_groups(discovered)
    _log(f"Warmed {len(warmed)} group(s): {sorted(warmed)}")
    assert warmed, "no Megatron groups warmed up — invalid parallel config?"

    handles = _collect_megatron_comms()
    _log(f"Discovered {len(handles)} unique NCCL comm(s)")
    assert handles, "must find at least one warm comm"
    assert len(handles) <= len(warmed), f"discovered {len(handles)} comms but only {len(warmed)} groups were warmed up"

    mem_before = _gpu_used_mb()
    sus = suspend_via_parallel_state(measure_per_comm=True)
    assert sus.success, f"suspend failed: {sus.skipped_reason}"
    assert sus.freed_mb > 100.0, f"freed only {sus.freed_mb:.0f} MB — suspect no real memory released"
    _log(f"suspend: freed {sus.freed_mb:.0f} MB across {len(sus.comms)} comm(s) in {sus.total_ms:.0f} ms")

    # Idempotency: second call returns already_suspended without re-issuing ncclCommSuspend.
    sus2 = suspend_via_parallel_state()
    assert not sus2.success and sus2.skipped_reason == "already_suspended", (
        f"second suspend should be a no-op, got {sus2}"
    )

    res = resume_via_parallel_state(measure_per_comm=True)
    assert res.success, f"resume failed: {res.skipped_reason}"
    rel_err = abs(res.reclaimed_mb - sus.freed_mb) / max(sus.freed_mb, 1.0)
    assert rel_err < 0.05, (
        f"resume reclaimed {res.reclaimed_mb:.0f} MB but suspend freed {sus.freed_mb:.0f} MB ({rel_err:.1%} drift)"
    )
    _log(f"resume: reclaimed {res.reclaimed_mb:.0f} MB across {len(res.comms)} comm(s) in {res.total_ms:.0f} ms")

    # Net memory should be within ~5% of the pre-suspend baseline.
    mem_after = _gpu_used_mb()
    net_err = abs(mem_after - mem_before) / max(mem_before, 1.0)
    assert net_err < 0.05, f"GPU memory drift after suspend/resume: {mem_before:.0f} → {mem_after:.0f} MB"

    # Post-resume sanity: every warmed group still works.
    for name, group in discovered.items():
        if name not in warmed:
            continue
        x = torch.ones(1024, dtype=torch.float32, device="cuda")
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()
    _log(f"post-resume allreduce passed on all {len(warmed)} warmed group(s)")

    # Second cycle to confirm state machine handles repeated suspend/resume.
    sus3 = suspend_via_parallel_state()
    assert sus3.success, "second suspend cycle failed"
    res3 = resume_via_parallel_state()
    assert res3.success, "second resume cycle failed"

    ps.destroy_model_parallel()
    _log("PASS")


if __name__ == "__main__":
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    try:
        test_nccl_suspend_resume_via_parallel_state()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    sys.exit(0)

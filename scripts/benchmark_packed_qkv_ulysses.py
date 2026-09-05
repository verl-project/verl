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
"""Microbenchmark packed versus three-call Ulysses QKV all-to-all.

Example:
    torchrun --standalone --nproc-per-node=2 scripts/benchmark_packed_qkv_ulysses.py \
        --local-seq-lens 256 1024 4096
"""

import argparse
import os
from collections.abc import Callable

import torch
import torch.distributed as dist

from verl.utils.ulysses import gather_seq_scatter_heads, set_ulysses_sequence_parallel_group

_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "fp8_e5m2": getattr(torch, "float8_e5m2", None),
}
_DEFAULT_DTYPES = [name for name, dtype in _DTYPES.items() if dtype is not None]


def _time_ms(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _measure_packed_parts(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, warmup: int, iterations: int
) -> tuple[float, float, float, float]:
    def pack():
        return torch.cat((q, k, v), dim=0)

    pack_ms = _time_ms(pack, warmup, iterations)
    packed = pack()

    def all_to_all():
        return gather_seq_scatter_heads(packed, seq_dim=2, head_dim=1)

    packed_a2a_ms = _time_ms(all_to_all, warmup, iterations)
    output = all_to_all()

    def unpack():
        return output.chunk(3, dim=0)

    unpack_ms = _time_ms(unpack, warmup, iterations)
    return pack_ms, packed_a2a_ms, unpack_ms, pack_ms + packed_a2a_ms + unpack_ms


def _random_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # torch.randn does not directly create FP8 tensors. FP32 initialization plus
    # casting represents the FP8 activation buffer handed to the collective.
    return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--local-seq-lens", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--dtypes", nargs="+", choices=_DTYPES, default=_DEFAULT_DTYPES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    if args.heads % world_size:
        raise ValueError(f"--heads ({args.heads}) must be divisible by world size ({world_size})")
    set_ulysses_sequence_parallel_group(dist.group.WORLD)

    if dist.get_rank() == 0:
        print(
            "| WS | dtype | S_local | heads | head_dim | 3xA2A ms | pack ms | 1xA2A ms | "
            "unpack ms | packed total ms | speedup |"
        )
        print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for dtype_name in args.dtypes:
        dtype = _DTYPES[dtype_name]
        if dtype is None:
            raise RuntimeError(f"{dtype_name} is unavailable in this PyTorch build")

        for local_seq_len in args.local_seq_lens:
            shape = (args.batch_size, args.heads, local_seq_len, args.head_dim)
            q = _random_tensor(shape, dtype, device)
            k = _random_tensor(shape, dtype, device)
            v = _random_tensor(shape, dtype, device)

            def three_all_to_all(q=q, k=k, v=v):
                return (
                    gather_seq_scatter_heads(q, seq_dim=2, head_dim=1),
                    gather_seq_scatter_heads(k, seq_dim=2, head_dim=1),
                    gather_seq_scatter_heads(v, seq_dim=2, head_dim=1),
                )

            torch.cuda.reset_peak_memory_stats(device)
            legacy_ms = _time_ms(three_all_to_all, args.warmup, args.iterations)
            legacy_peak = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            pack_ms, packed_a2a_ms, unpack_ms, packed_total_ms = _measure_packed_parts(
                q, k, v, args.warmup, args.iterations
            )
            packed_peak = torch.cuda.max_memory_allocated(device)

            if dist.get_rank() == 0:
                speedup = legacy_ms / packed_total_ms
                print(
                    f"| {world_size} | {dtype_name} | {local_seq_len} | {args.heads} | {args.head_dim} | "
                    f"{legacy_ms:.3f} | {pack_ms:.3f} | {packed_a2a_ms:.3f} | {unpack_ms:.3f} | "
                    f"{packed_total_ms:.3f} | {speedup:.3f}x |"
                )
                print(f"  peak HBM: three-A2A={legacy_peak / 2**20:.1f} MiB, packed={packed_peak / 2**20:.1f} MiB")

    set_ulysses_sequence_parallel_group(None)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

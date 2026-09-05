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
"""Opt-in NCCL checks for the AsyncCollectiveHandle CUDA stream contract."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "collective.py"
_SPEC = importlib.util.spec_from_file_location("collective_cuda_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collective = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = collective
_SPEC.loader.exec_module(collective)


def _nccl_stream_worker(rank: int, world_size: int, init_method: str, artifact_dir: str) -> None:
    torch.cuda.set_device(rank)
    verl_package = types.ModuleType("verl")
    verl_package.__path__ = []
    utils_package = types.ModuleType("verl.utils")
    utils_package.__path__ = []
    device_module = types.ModuleType("verl.utils.device")
    device_module.get_device_id = torch.cuda.current_device
    device_module.get_device_name = lambda: "cuda"
    device_module.get_torch_device = lambda: torch.cuda
    device_module.is_device_available = torch.cuda.is_available
    sys.modules["verl"] = verl_package
    sys.modules["verl.utils"] = utils_package
    sys.modules["verl.utils.device"] = device_module
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        producer_stream = torch.cuda.Stream(device=device)
        other_stream = torch.cuda.Stream(device=device)
        tensor = torch.tensor(float(rank + 1), device=device)
        producer_stream.wait_stream(torch.cuda.current_stream(device))
        finalize_calls = 0

        def finalize() -> torch.Tensor:
            nonlocal finalize_calls
            finalize_calls += 1
            return tensor.clone()

        with torch.cuda.stream(producer_stream):
            complete_event = torch.cuda.Event()
            handle = collective.AsyncCollectiveHandle(
                work=dist.all_reduce(tensor, async_op=True),
                finalize=finalize,
                comm_kind="all_reduce",
                process_group_id="world",
                sequence_id=0,
                complete_event=complete_event,
                consumer_device=device,
                owned_resources=(tensor,),
            )
            handle.wait_collective()

        with torch.cuda.stream(other_stream):
            with pytest.raises(RuntimeError, match="one CUDA consumer stream"):
                handle.finalize_result()

        with torch.cuda.stream(producer_stream):
            result = handle.finalize_result()
            repeated_result = handle.wait()
            consumed = result * 2
        producer_stream.synchronize()

        expected = world_size * (world_size + 1) / 2
        assert result is repeated_result
        assert result.item() == expected
        assert consumed.item() == expected * 2
        assert finalize_calls == 1
        assert complete_event.query()
        assert handle.collective_complete
        assert handle.finalized
        assert handle.finalization_error is None

        properties = torch.cuda.get_device_properties(device)
        report = {
            "backend": dist.get_backend(),
            "comm_kind": handle.comm_kind,
            "complete_event_observed": complete_event.query(),
            "consumer_stream_contract": "single_stream_fail_loud",
            "cuda_version": torch.version.cuda,
            "device_name": properties.name,
            "finalize_calls": finalize_calls,
            "physical_gpu_uuids": os.environ.get("PHYSICAL_GPU_UUIDS", "unknown"),
            "process_group_membership": list(range(world_size)),
            "rank": rank,
            "result": result.item(),
            "single_consumer_stream_rejected": True,
            "torch_version": torch.__version__,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "container-managed"),
            "world_size": world_size,
        }
        output = Path(artifact_dir) / f"async-collective-nccl-rank-{rank}.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("VERL_RUN_ASYNC_COLLECTIVE_NCCL_TEST") != "1",
    reason="set VERL_RUN_ASYNC_COLLECTIVE_NCCL_TEST=1 for the opt-in NCCL check",
)
def test_async_collective_handle_nccl_stream_contract(tmp_path):
    world_size = int(os.environ.get("VERL_ASYNC_COLLECTIVE_WORLD_SIZE", "2"))
    if world_size not in (2, 4):
        pytest.fail("VERL_ASYNC_COLLECTIVE_WORLD_SIZE must be 2 or 4")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"the NCCL check requires {world_size} visible CUDA devices")
    artifact_dir = os.environ.get("VERL_ASYNC_COLLECTIVE_ARTIFACT_DIR", str(tmp_path))
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    init_method = f"file://{tmp_path / 'nccl_init'}"
    mp.spawn(
        _nccl_stream_worker,
        args=(world_size, init_method, artifact_dir),
        nprocs=world_size,
        join=True,
    )

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check each Ascend operator independently through the real verl switch."""

import importlib.util
import multiprocessing
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _run_check(check, enabled):
    for name in ("VLLM_BATCH_INVARIANT", "VERL_FULL_DETERMINISM", "VERL_SEED"):
        os.environ.pop(name, None)

    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend
    import vllm_ascend.batch_invariant as bi

    from verl.workers.config import RolloutConfig
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    config = RolloutConfig(name="vllm", full_determinism=enabled, tensor_model_parallel_size=1, max_model_len=128)
    # Supply model metadata and Ray identity only. No model, Ray cluster or engine is started.
    model = SimpleNamespace(hf_config=SimpleNamespace(max_position_embeddings=128))
    context = SimpleNamespace(get_job_id=lambda: "batch-invariant-ut")
    with (
        patch("ray.get_runtime_context", return_value=context),
        patch("ray.util.get_node_ip_address", return_value="127.0.0.1"),
    ):
        server = vLLMHttpServer(
            config, model, RolloutMode.STANDALONE, [], 0, 0, 1, 1, os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")
        )
    for sock in (server._master_sock, server._dp_rpc_sock, server._dp_master_sock):
        sock.close()
    assert os.environ.get("VLLM_BATCH_INVARIANT", "0") == ("1" if enabled else "0")
    assert bi.HAS_ASCENDC_BATCH_INVARIANT, "The installed independent AscendC operator package could not be loaded."
    # Stand in for the engine's config singleton, while running the real registration code.
    with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=SimpleNamespace(weight_nz_mode=1)):
        bi.init_batch_invariance()
    torch.npu.set_device(0)
    check(enabled)


def run_on_npu(check, enabled):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU.")
    # Only absence is optional; an installed package that fails to load must fail.
    if importlib.util.find_spec("batch_invariant_ops") is None:
        pytest.skip("Requires the independent batch-invariant operator package (batch_invariant_ops).")

    # Each operator/state gets fresh global dispatch tables and its own exit status.
    process = multiprocessing.get_context("spawn").Process(target=_run_check, args=(check, enabled))
    process.start()
    try:
        process.join(timeout=180)
        assert process.exitcode == 0, (
            f"{check.__name__} failed or timed out ({enabled=}, exitcode={process.exitcode}); see subprocess output."
        )
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def profile_operator(entry, call, enabled, operator=None):
    import torch
    from torch.autograd.profiler_legacy import profile

    with profile(use_cuda=False) as captured:
        result = call()
        torch.npu.synchronize()
    events = {event.key: int(event.count) for event in captured.key_averages()}
    counts = {name: count for name, count in events.items() if name.startswith("batch_invariant_ops::")}
    if enabled and operator is not None:
        assert counts.get(f"batch_invariant_ops::{operator}", 0) > 0, (entry, counts)
    else:
        assert not counts, (entry, counts)
    print(f"{enabled=}, {entry=}, {counts=}", flush=True)
    return result, events


def _check_switch(enabled):
    import vllm_ascend.batch_invariant as bi

    assert os.environ.get("VLLM_BATCH_INVARIANT", "0") == ("1" if enabled else "0")
    assert (bi._batch_invariant_LIB is not None) is enabled
    print(f"{enabled=}, switch_checked=True", flush=True)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_batch_invariant_switch(enabled):
    run_on_npu(_check_switch, enabled)


def _check_mm(enabled):
    import torch

    x_cpu = (torch.arange(512).reshape(16, 32) % 7 - 3).to(torch.bfloat16)
    y_cpu = (torch.arange(512).reshape(32, 16) % 5 - 2).to(torch.bfloat16)
    expected = (x_cpu.float() @ y_cpu.float()).to(torch.bfloat16)
    x, y = x_cpu.npu(), y_cpu.npu()
    actual, _ = profile_operator("torch.mm", lambda: torch.mm(x, y), enabled, "npu_mm_batch_invariant")
    assert actual.device == x.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    print(f"{enabled=}, entry='torch.mm', result_checked=True", flush=True)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_mm(enabled):
    run_on_npu(_check_mm, enabled)


def _check_matmul(enabled):
    import torch

    x_cpu = (torch.arange(512).reshape(16, 32) % 7 - 3).to(torch.bfloat16)
    y_cpu = (torch.arange(512).reshape(32, 16) % 5 - 2).to(torch.bfloat16)
    expected = (x_cpu.float() @ y_cpu.float()).to(torch.bfloat16)
    x, y = x_cpu.npu(), y_cpu.npu()
    actual, _ = profile_operator("torch.matmul", lambda: torch.matmul(x, y), enabled, "npu_matmul_batch_invariant")
    assert actual.device == x.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    print(f"{enabled=}, entry='torch.matmul', result_checked=True", flush=True)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_matmul(enabled):
    run_on_npu(_check_matmul, enabled)


def _check_torch_sum(enabled):
    import torch
    import vllm_ascend.batch_invariant as bi

    assert (torch.sum is bi.reduce_sum) is enabled
    x_cpu = (torch.arange(512).reshape(16, 32) % 7 - 3).to(torch.bfloat16)
    expected = x_cpu.float().sum(dim=-1).to(torch.bfloat16)
    x = x_cpu.npu()
    actual, _ = profile_operator("torch.sum", lambda: torch.sum(x, dim=-1), enabled, "npu_reduce_sum_batch_invariant")
    assert actual.device == x.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    print(f"{enabled=}, entry='torch.sum', result_checked=True", flush=True)


def _check_tensor_sum(enabled):
    import torch
    import vllm_ascend.batch_invariant as bi

    assert (torch.Tensor.sum is bi.reduce_sum) is enabled
    x_cpu = (torch.arange(512).reshape(16, 32) % 7 - 3).to(torch.bfloat16)
    expected = x_cpu.float().sum(dim=1, keepdim=True).to(torch.bfloat16)
    x = x_cpu.npu()
    actual, _ = profile_operator(
        "Tensor.sum", lambda: x.sum(dim=1, keepdim=True), enabled, "npu_reduce_sum_batch_invariant"
    )
    assert actual.device == x.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    print(f"{enabled=}, entry='Tensor.sum', result_checked=True", flush=True)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_torch_sum(enabled):
    run_on_npu(_check_torch_sum, enabled)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_tensor_sum(enabled):
    run_on_npu(_check_tensor_sum, enabled)


def _check_add_rms_norm(enabled):
    import torch
    import torch_npu
    import vllm_ascend.batch_invariant as bi

    assert (torch_npu.npu_add_rms_norm is bi.add_rms_norm) is enabled
    x_cpu = (torch.arange(512).reshape(16, 32) % 7 - 3).to(torch.bfloat16)
    residual_cpu = torch.full_like(x_cpu, 0.5)
    weight_cpu = (torch.arange(32).float() / 32 + 0.5).to(torch.bfloat16)
    x, residual, weight = x_cpu.npu(), residual_cpu.npu(), weight_cpu.npu()
    eps = 1e-6
    added = x_cpu.float() + residual_cpu.float()
    normalized = added * torch.rsqrt(added.square().mean(dim=-1, keepdim=True) + eps) * weight_cpu.float()
    (actual, rstd, residual_out), events = profile_operator(
        "AddRMSNorm", lambda: torch_npu.npu_add_rms_norm(x, residual, weight, eps), enabled
    )
    # This vLLM Ascend version uses add + RMSNorm, not the independent fused operator.
    if enabled:
        assert events.get("aten::add", 0) > 0, events
        assert events.get("npu::npu_rms_norm", 0) > 0, events
        assert events.get("npu::npu_add_rms_norm", 0) == 0, events
        assert rstd is None
    else:
        assert events.get("npu::npu_add_rms_norm", 0) > 0, events
    assert actual.device == residual_out.device == x.device
    torch.testing.assert_close(residual_out.cpu(), added.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), normalized.to(torch.bfloat16), rtol=1e-2, atol=1e-2)
    rms_events = {
        name: count
        for name, count in events.items()
        if name in ("aten::add", "npu::npu_rms_norm", "npu::npu_add_rms_norm")
    }
    print(f"{enabled=}, entry='AddRMSNorm', {rms_events=}, result_checked=True", flush=True)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_add_rms_norm(enabled):
    run_on_npu(_check_add_rms_norm, enabled)


def _check_fused_infer_attention_score_binding(enabled):
    import torch
    import torch_npu

    attention_bound = (
        torch_npu.npu_fused_infer_attention_score
        is torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
    )
    assert attention_bound is enabled
    print(f"{enabled=}, {attention_bound=}", flush=True)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_fused_infer_attention_score_binding(enabled):
    run_on_npu(_check_fused_infer_attention_score_binding, enabled)

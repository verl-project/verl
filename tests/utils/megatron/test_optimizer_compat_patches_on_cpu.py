# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for the Megatron-Core optimizer compatibility patches (no megatron import needed)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_patch_module():
    path = _REPO_ROOT / "verl/models/mcore/patch.py"
    spec = importlib.util.spec_from_file_location("mcore_patch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PATCH = _load_patch_module()


class _Opt:
    calls = []

    @classmethod
    def _build_model_and_main_param_groups(cls, tag):
        cls.calls.append((tag, torch.is_grad_enabled()))
        return tag


def test_wrap_no_grad_keeps_classmethod_and_disables_grad():
    assert _PATCH.wrap_no_grad(_Opt, "_build_model_and_main_param_groups", "_marker")
    assert isinstance(_Opt.__dict__["_build_model_and_main_param_groups"], classmethod)
    with torch.enable_grad():
        assert _Opt._build_model_and_main_param_groups("x") == "x"
    assert _Opt.calls == [("x", False)]
    # idempotent
    assert not _PATCH.wrap_no_grad(_Opt, "_build_model_and_main_param_groups", "_marker")


def _fake_clip_grads(record):
    """A stand-in for megatron.core.optimizer.clip_grads with pure-torch kernels."""

    def multi_tensor_applier(op, _buf, tensor_lists, *args):
        return op(tensor_lists, *args)

    def l2_norm_impl(tensor_lists, _per_tensor):
        (tensors,) = tensor_lists
        return torch.stack([t.float().norm() for t in tensors]).norm(), None

    def multi_tensor_scale_impl(tensor_lists, scale):
        src, dst = tensor_lists
        for s, d in zip(src, dst, strict=True):
            d.mul_(scale)

    def orig_norm(grads, norm_type, group):
        record.append(("orig_norm", len(grads)))
        return torch.stack([g.float().norm() for g in grads]).norm()

    def orig_clip(parameters, max_norm, total_norm, use_decoupled_grad):
        record.append(("orig_clip", len(parameters)))

    return SimpleNamespace(
        get_grad_norm_fp32=orig_norm,
        clip_grad_by_total_norm_fp32=orig_clip,
        multi_tensor_applier=multi_tensor_applier,
        l2_norm_impl=l2_norm_impl,
        multi_tensor_scale_impl=multi_tensor_scale_impl,
        multi_tensor_scale_tensor_impl=None,
        to_local_if_dtensor=lambda t: t,
        get_data_parallel_group_if_dtensor=lambda t, g: g,
    )


@pytest.fixture
def single_rank_group(tmp_path):
    """One-rank gloo group, destroyed afterwards only if created here (CPU tests share one pytest process)."""
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


def test_dtype_safe_grad_norm_matches_full_norm_and_forwards_single_dtype(single_rank_group):
    record = []
    cg = _fake_clip_grads(record)
    get_norm, _ = _PATCH.make_dtype_safe_grad_fns(cg)
    fp32 = [torch.full((4,), 3.0), torch.full((2,), 4.0)]
    bf16 = [torch.full((8,), 1.0, dtype=torch.bfloat16)]
    expected = torch.stack([t.float().norm() for t in fp32 + bf16]).norm().item()
    got = get_norm(fp32 + bf16, 2, dist.group.WORLD)
    assert abs(got - expected) < 1e-5
    assert record == []  # mixed list did not fall back to the original
    got_single = get_norm(fp32, 2, dist.group.WORLD)
    assert record == [("orig_norm", 2)]  # single dtype is forwarded unchanged
    assert abs(float(got_single) - torch.stack([t.norm() for t in fp32]).norm().item()) < 1e-5


def test_dtype_safe_clip_scales_every_group(single_rank_group):
    record = []
    cg = _fake_clip_grads(record)
    _, clip = _PATCH.make_dtype_safe_grad_fns(cg)
    p32 = SimpleNamespace(grad=torch.full((4,), 2.0), decoupled_grad=None)
    p16 = SimpleNamespace(grad=torch.full((4,), 2.0, dtype=torch.bfloat16), decoupled_grad=None)
    clip([p32, p16], max_norm=1.0, total_norm=4.0)  # coeff 0.25
    assert torch.allclose(p32.grad, torch.full((4,), 0.5))
    assert torch.allclose(p16.grad.float(), torch.full((4,), 0.5))
    assert record == []
    clip([p32], max_norm=1.0, total_norm=4.0)
    assert record == [("orig_clip", 1)]  # single dtype forwarded

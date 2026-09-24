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

"""CPU tests for fused-MoE handling in ``verl/utils/vllm/vllm_quant_utils.py``.

vLLM 0.24.0 (MoE refactor, vllm-project/vllm#41184) turned ``FusedMoE`` from an
``nn.Module`` class into a factory *function* that returns a ``MoERunner``, and
moved the fused expert weights onto a ``RoutedExperts`` submodule
(``experts`` -> ``experts.routed_experts``). The old code called
``isinstance(module, FusedMoE)`` which raised
``TypeError: isinstance() arg 2 must be a type`` once ``FusedMoE`` became a
function. These tests exercise both the pre-0.24 and post-0.24 module layouts
with lightweight fakes (no real vLLM required).
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VLLM_UTILS_DIR = _REPO_ROOT / "verl/utils/vllm"
_MODULE_PATH = _VLLM_UTILS_DIR / "vllm_quant_utils.py"


def _make_module(name: str) -> types.ModuleType:
    return types.ModuleType(name)


def _load_quant_utils(fused_moe_is_function: bool):
    """Load ``vllm_quant_utils`` with vLLM/verl heavyweight deps stubbed.

    Args:
        fused_moe_is_function: when True, emulate vLLM >= 0.24 where ``FusedMoE``
            is a factory function and the expert weights live on ``RoutedExperts``
            owned by a ``MoERunner``. When False, emulate vLLM < 0.24 where
            ``FusedMoE`` is the ``nn.Module`` class holding the weights.

    Returns:
        (module, namespace) where ``namespace`` exposes the fake classes used to
        build test models so callers can assert against the exact class objects
        the loaded module resolved.
    """

    class _FakeLinearBase(torch.nn.Module):
        def __init__(self, dtype=torch.float8_e4m3fn):
            super().__init__()
            self.weight = torch.empty(2, 2, dtype=dtype)

    ns: dict = {"LinearBase": _FakeLinearBase}

    fused_moe_pkg = _make_module("vllm.model_executor.layers.fused_moe")
    fused_moe_layer = _make_module("vllm.model_executor.layers.fused_moe.layer")
    linear_mod = _make_module("vllm.model_executor.layers.linear")
    linear_mod.LinearBase = _FakeLinearBase

    if fused_moe_is_function:

        class _FakeRoutedExperts(torch.nn.Module):
            def __init__(self, dtype=torch.float8_e4m3fn):
                super().__init__()
                self.w13_weight = torch.empty(2, 2, dtype=dtype)
                self.w2_weight = torch.empty(2, 2, dtype=dtype)

        class _FakeMoERunner(torch.nn.Module):
            def __init__(self, dtype=torch.float8_e4m3fn):
                super().__init__()
                self.routed_experts = _FakeRoutedExperts(dtype)

        def _fused_moe_factory(*args, **kwargs):  # noqa: N802 - mirrors vLLM name
            return _FakeMoERunner()

        fused_moe_layer.FusedMoE = _fused_moe_factory
        fused_moe_pkg.FusedMoE = _fused_moe_factory
        fused_moe_pkg.RoutedExperts = _FakeRoutedExperts
        fused_moe_pkg.MoERunner = _FakeMoERunner
        ns.update(RoutedExperts=_FakeRoutedExperts, MoERunner=_FakeMoERunner)
    else:

        class _FakeFusedMoE(torch.nn.Module):
            def __init__(self, dtype=torch.float8_e4m3fn):
                super().__init__()
                self.w13_weight = torch.empty(2, 2, dtype=dtype)
                self.w2_weight = torch.empty(2, 2, dtype=dtype)

        fused_moe_layer.FusedMoE = _FakeFusedMoE
        fused_moe_pkg.FusedMoE = _FakeFusedMoE
        ns.update(FusedMoE=_FakeFusedMoE)

    # verl leaf modules that pull in torch/triton or vLLM at import time.
    fake_kernel = _make_module("verl.utils.kernel.fp8_kernel")
    fake_kernel.scaled_fp8_blockwise = lambda *a, **k: (None, None)

    # The vllm_fp8_utils / vllm_fp4_utils siblings only touch vLLM inside
    # function bodies, so they can be imported for real. Stub the package
    # ``__init__`` (which does pull heavyweight deps) but point its search path
    # at the real directory so the submodule imports still resolve.
    fake_vllm_pkg = _make_module("verl.utils.vllm")
    fake_vllm_pkg.__path__ = [str(_VLLM_UTILS_DIR)]

    fakes = {
        "vllm": _make_module("vllm"),
        "vllm.model_executor": _make_module("vllm.model_executor"),
        "vllm.model_executor.layers": _make_module("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": fused_moe_pkg,
        "vllm.model_executor.layers.fused_moe.layer": fused_moe_layer,
        "vllm.model_executor.layers.linear": linear_mod,
        "verl.utils.kernel": _make_module("verl.utils.kernel"),
        "verl.utils.kernel.fp8_kernel": fake_kernel,
        "verl.utils.vllm": fake_vllm_pkg,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    preexisting = set(sys.modules)
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location("verl_vllm_quant_utils_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        ns["fp4_utils"] = sys.modules["verl.utils.vllm.vllm_fp4_utils"]
        ns["fp8_utils"] = sys.modules["verl.utils.vllm.vllm_fp8_utils"]
    finally:
        # The sibling modules imported above are bound to the stubbed package,
        # so drop them rather than leave them for the next importer.
        for name in set(sys.modules) - preexisting:
            if name.startswith("verl.utils.vllm."):
                sys.modules.pop(name, None)
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module, ns


PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _build_model(ns: dict, moe_dtype=torch.float8_e4m3fn):
    """Build ``model.layers.0`` with an attention linear and a fused-MoE block."""
    linear_cls = ns["LinearBase"]
    moe_cls = ns.get("MoERunner") or ns.get("FusedMoE")

    attn = torch.nn.Module()
    attn.qkv_proj = linear_cls(torch.float8_e4m3fn)

    mlp = torch.nn.Module()
    mlp.experts = moe_cls(moe_dtype)
    mlp.gate = linear_cls(torch.bfloat16)  # router: must NOT be treated as fp8

    layer0 = torch.nn.Module()
    layer0.self_attn = attn
    layer0.mlp = mlp

    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([layer0])

    model = torch.nn.Module()
    model.model = inner
    model.packed_modules_mapping = PACKED_MODULES_MAPPING
    return model


def test_new_vllm_resolves_routed_experts_and_does_not_crash():
    """vLLM >= 0.24: per-expert names must resolve to RoutedExperts, no TypeError."""
    mod, ns = _load_quant_utils(fused_moe_is_function=True)
    model = _build_model(ns)

    routed_experts = model.model.layers[0].mlp.experts.routed_experts
    assert isinstance(routed_experts, ns["RoutedExperts"])

    # Class resolution landed on the concrete post-refactor classes.
    assert ns["RoutedExperts"] in mod._EXPERT_WEIGHT_CLASSES
    assert ns["MoERunner"] in mod._MOE_STOP_CLASSES

    # The exact name that crashed before the fix: a per-expert HF weight that
    # walks into the fused-MoE block (index + proj remapped to gate_up_proj).
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    resolved = mod.get_module_from_param_name(model, name)
    assert resolved is routed_experts

    # Already-fused expert weight name resolves to the same holder.
    resolved_fused = mod.get_module_from_param_name(model, "model.layers.0.mlp.experts.w13_weight")
    assert resolved_fused is routed_experts


def test_new_vllm_is_fp8_weight_detects_moe_and_linear():
    mod, ns = _load_quant_utils(fused_moe_is_function=True)
    model = _build_model(ns, moe_dtype=torch.float8_e4m3fn)

    mod.fp8_state.seen_params.clear()
    mod.fp8_state.fp8_param_names.clear()

    # fp8 expert weight (per-expert HF name) -> detected as fp8
    assert mod.is_fp8_weight("model.layers.0.mlp.experts.0.gate_proj.weight", model) is True
    # fp8 attention linear -> detected as fp8
    assert mod.is_fp8_weight("model.layers.0.self_attn.q_proj.weight", model) is True
    # bf16 router gate -> NOT fp8
    assert mod.is_fp8_weight("model.layers.0.mlp.gate.weight", model) is False


def test_new_vllm_bf16_experts_not_flagged_fp8():
    mod, ns = _load_quant_utils(fused_moe_is_function=True)
    model = _build_model(ns, moe_dtype=torch.bfloat16)

    mod.fp8_state.seen_params.clear()
    mod.fp8_state.fp8_param_names.clear()

    assert mod.is_fp8_weight("model.layers.0.mlp.experts.0.gate_proj.weight", model) is False


def test_old_vllm_fusedmoe_class_still_supported():
    """vLLM < 0.24: FusedMoE is an nn.Module class holding the weights."""
    mod, ns = _load_quant_utils(fused_moe_is_function=False)
    model = _build_model(ns, moe_dtype=torch.float8_e4m3fn)

    assert ns["FusedMoE"] in mod._EXPERT_WEIGHT_CLASSES

    experts = model.model.layers[0].mlp.experts
    resolved = mod.get_module_from_param_name(model, "model.layers.0.mlp.experts.0.gate_proj.weight")
    assert resolved is experts

    mod.fp8_state.seen_params.clear()
    mod.fp8_state.fp8_param_names.clear()
    assert mod.is_fp8_weight("model.layers.0.mlp.experts.0.gate_proj.weight", model) is True


def test_dsv41_engram_projection_refit_emits_fp8_weight_and_scale():
    _, ns = _load_quant_utils(fused_moe_is_function=True)
    weight = (torch.arange(64 * 64).reshape(64, 64) % 31 - 15).to(torch.bfloat16)
    for name in ("layers.1.engram.wkv.weight", "layers.14.engram.wkv.base_layer.weight"):
        exported = dict(ns["fp4_utils"].iter_deepseek_v4_weights([(name, weight)], block_size=32))
        scale_name = name.removesuffix(".weight") + ".scale"
        assert set(exported) == {name, scale_name}
        assert exported[name].dtype == torch.float8_e4m3fn
        assert exported[scale_name].dtype == torch.float8_e8m0fnu
        scale = exported[scale_name].float().repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
        torch.testing.assert_close(exported[name].float() * scale, weight.float(), rtol=0, atol=0)


def _engram_owner_refit_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    mod, ns = _load_quant_utils(fused_moe_is_function=True)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        # Uneven ownership, followed by a table with an empty owner shard.
        for rows in (17, 1):
            source = (torch.arange(rows * 64).reshape(rows, 64) % 15 - 7).bfloat16()
            table = distribute_tensor(source, mesh, [Shard(0)])
            local_before = table.to_local().clone()
            targets = []
            # Rollout partitions deliberately do not match the actor's row owners.
            for begin, end in ((0, min(rows, 5)), (min(rows, 5), rows)):
                model = torch.nn.Module()
                model.model_type = "deepseek_v41"
                model.embed_tokens = torch.nn.Module()
                embed = model.embed_tokens
                embed.weight = torch.nn.Parameter(
                    torch.zeros(end - begin, 64, dtype=torch.float8_e4m3fn), requires_grad=False
                )
                embed.weight.engram_vocab_start = begin
                embed.weight_scale_inv = torch.nn.Parameter(
                    torch.zeros(end - begin, 2, dtype=torch.uint8), requires_grad=False
                )
                model.hf_to_vllm_mapper = types.SimpleNamespace(apply_list=lambda names: ["embed_tokens.weight"])
                model.image_start = torch.nn.Parameter(torch.zeros(64), requires_grad=False)

                def load_weights(weights, model=model):
                    loaded = set()
                    for name, tensor in weights:
                        model.get_parameter(name).data.copy_(tensor)
                        loaded.add(name)
                    return loaded

                model.load_weights = load_weights
                runner = types.SimpleNamespace(
                    model=model,
                    vllm_config=types.SimpleNamespace(
                        quant_config=types.SimpleNamespace(weight_block_size=[32, 32]),
                        model_config=types.SimpleNamespace(dtype=torch.bfloat16),
                    ),
                )
                pointers = (embed.weight.data_ptr(), embed.weight_scale_inv.data_ptr())
                targets.append((runner, begin, end, pointers))

            calls = []
            with patch.object(DTensor, "full_tensor", side_effect=AssertionError("Full Engram gather is forbidden")):
                stream = ns["fp8_utils"].iter_dsv41_engram_rows(
                    "layers.1.engram.embed.weight", table, torch.device("cpu"), chunk_bytes=512
                )
                for name, chunk in stream:
                    start = int(name.rsplit("__rows_", 1)[1])
                    assert chunk.nbytes <= 512
                    torch.testing.assert_close(chunk, source[start : start + len(chunk)], rtol=0, atol=0)
                    calls.append((start, len(chunk)))
                    for runner, _, _, _ in targets:
                        # Exercise both chunk-only and mixed buckets through the production entry point.
                        weights = [(name, chunk)]
                        if start == 0:
                            weights.append(("image_start", torch.ones(64)))
                        loaded = mod.load_quanted_weights(weights, runner)
                        assert {"embed_tokens.weight", "embed_tokens.weight_scale_inv"} <= loaded
                        if start == 0:
                            assert "image_start" in loaded
            torch.testing.assert_close(table.to_local(), local_before, rtol=0, atol=0)
            all_calls = [None, None]
            dist.all_gather_object(all_calls, calls)
            assert all_calls[0] == all_calls[1]
            assert sum(count for _, count in calls) == rows
            for runner, begin, end, pointers in targets:
                embed = runner.model.embed_tokens
                assert (embed.weight.data_ptr(), embed.weight_scale_inv.data_ptr()) == pointers
                scale = embed.weight_scale_inv.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
                torch.testing.assert_close(embed.weight.float() * scale, source[begin:end].float(), rtol=0, atol=0)
                torch.testing.assert_close(runner.model.image_start, torch.ones(64), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_dsv41_engram_chunked_refit_with_uneven_owners(tmp_path):
    torch.multiprocessing.spawn(_engram_owner_refit_worker, args=(str(tmp_path / "init"),), nprocs=2, join=True)


def _mcore_engram_owner_stream_worker(rank, rendezvous):
    import torch.distributed as dist

    _, ns = _load_quant_utils(fused_moe_is_function=True)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        for rows in (17, 1):
            source = (torch.arange(rows * 64).reshape(rows, 64) % 15 - 7).bfloat16()
            base, remainder = divmod(rows, 2)
            row_start = rank * base + min(rank, remainder)
            row_end = row_start + base + int(rank < remainder)
            local = source[row_start:row_end].clone()
            module = types.SimpleNamespace(
                global_num_embeddings=rows,
                row_start=row_start,
                row_end=row_end,
            )

            calls = []
            stream = ns["fp8_utils"].iter_dsv41_engram_rows(
                "layers.1.engram.embed.weight",
                local,
                torch.device("cpu"),
                chunk_bytes=512,
                module=module,
                ep_group=dist.group.WORLD,
            )
            for name, chunk in stream:
                start = int(name.rsplit("__rows_", 1)[1])
                assert chunk.nbytes <= 512
                torch.testing.assert_close(chunk, source[start : start + len(chunk)], rtol=0, atol=0)
                calls.append((start, len(chunk)))

            all_calls = [None, None]
            dist.all_gather_object(all_calls, calls)
            assert all_calls[0] == all_calls[1]
            assert sum(count for _, count in calls) == rows
            torch.testing.assert_close(local, source[row_start:row_end], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_dsv41_mcore_engram_chunked_export_with_uneven_owners(tmp_path):
    torch.multiprocessing.spawn(
        _mcore_engram_owner_stream_worker,
        args=(str(tmp_path / "mcore_init"),),
        nprocs=2,
        join=True,
    )

# Copyright 2026 Individual Contributor
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
"""CPU regression tests for layered FSDP1 LoRA collection.

The model, LoRA injection, PEFT state-dict filtering, parameters, FlatParameter,
and tensor storage are real. Only the FSDP wrapper/collective boundary is faked:
before summon an owning unit exposes a flat parameter, and during summon it
exposes its original parameters. No process group or accelerator is needed.
These tests do not exercise distributed collectives or measure peak memory.
"""

from collections import Counter, OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from torch import nn
from torch.distributed.fsdp import FlatParameter

from verl.utils import fsdp_utils


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.outer = _Block()
        self.outer.inner = _Block()
        self.unadapted = nn.Linear(4, 4, bias=False)


class _CpuFSDP(nn.Module):
    """A wrapper seam; the PEFT model itself retains its real module names.

    ``children`` describes logical FSDP nesting without replacing the original
    PEFT modules. That lets PEFT discover its own adapter prefixes normally,
    while the wrapper exposes FSDP's internal namespace to the collector.
    """

    def __init__(self, module, *, children=None, flattened=True):
        super().__init__()
        self._fsdp_wrapped_module = module
        self._is_root = None
        self.gathered = False
        self.flattened = flattened
        self.units = OrderedDict(children or {})
        nested = tuple(self.units)
        self.owned_parameters = [
            (name, param)
            for name, param in module.named_parameters()
            if not any(name == prefix or name.startswith(prefix + ".") for prefix in nested)
        ]
        # Do not register this simulated sharded view in the real PEFT model.
        flat = None
        if self.owned_parameters:
            flat = FlatParameter(
                torch.cat([param.detach().reshape(-1) for _, param in self.owned_parameters]),
                requires_grad=False,
            )
            flat._fqns = tuple(name for name, _ in self.owned_parameters)
        object.__setattr__(self, "flat_parameter", flat)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        yield prefix, self
        for name, unit in self.units.items():
            child_prefix = ".".join(filter(None, [prefix, "_fsdp_wrapped_module", name]))
            yield from unit.named_modules(memo=memo, prefix=child_prefix, remove_duplicate=remove_duplicate)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        parameters = []
        if self.flat_parameter is not None:
            if self.flattened and not self.gathered:
                parameters.append((".".join(filter(None, [prefix, "_flat_param"])), self.flat_parameter))
            else:
                for name, param in self.owned_parameters:
                    parameters.append((".".join(filter(None, [prefix, "_fsdp_wrapped_module", name])), param))
        if recurse:
            for name, unit in self.units.items():
                child_prefix = ".".join(filter(None, [prefix, "_fsdp_wrapped_module", name]))
                parameters.extend(unit.named_parameters(prefix=child_prefix, remove_duplicate=remove_duplicate))
        for name, param in parameters:
            # Real FSDP.named_parameters removes all internal wrapper segments
            # in SUMMON_FULL_PARAMS state, including those of nested children.
            if self.gathered:
                name = name.replace("_fsdp_wrapped_module.", "")
            yield name, param

    @staticmethod
    def summon_full_params(*args, **kwargs):
        raise AssertionError("the CPU fixture must install its collective seam")


@pytest.fixture
def lora_model():
    model = get_peft_model(_TinyModel(), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    with torch.no_grad():
        for index, (name, param) in enumerate(model.named_parameters(), start=1):
            if "lora_" in name:
                param.fill_(index / 10)
    # The oracle is the real PEFT export, computed before adding any wrapper.
    expected = {name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(model).items()}
    assert len(expected) == 4
    return model, expected


@pytest.fixture
def fsdp_seam(monkeypatch):
    state = SimpleNamespace(root=None, events=[], initialized=False)

    def lazy_init(module, root):
        assert module is root is state.root
        state.events.append(("lazy_init", module))
        state.initialized = True
        for _, unit in root.named_modules():
            unit._is_root = unit is root

    @contextmanager
    def summon(module, recurse=True, writeback=True, **kwargs):
        if module is state.root:
            state.events.append(("root_summon", module))
            raise AssertionError("full-root summon must not be used by layered adapter collection")
        state.events.append(("summon", module))
        assert state.initialized, "initialize FSDP from its root before summoning a child"
        assert module._is_root is False, "a child must never be temporarily marked as an FSDP root"
        assert recurse is False, "nested units must be summoned independently, exactly once"
        assert writeback is False
        assert not module.gathered
        module.gathered = True
        try:
            yield
        finally:
            module.gathered = False

    monkeypatch.setattr(fsdp_utils, "FSDP", _CpuFSDP)
    monkeypatch.setattr(fsdp_utils, "_lazy_init", lazy_init)
    monkeypatch.setattr(_CpuFSDP, "summon_full_params", staticmethod(summon))
    monkeypatch.setattr(fsdp_utils, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    return state


def _root(model, units, seam):
    root = _CpuFSDP(model, children=units)
    seam.root = root
    return root


def _assert_adapter_equals(actual, expected):
    assert set(actual) == set(expected)
    for name, value in actual.items():
        assert value.device.type == "cpu"
        assert value.requires_grad is False
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_flat_parameter_original_names_identify_lora_owning_unit(lora_model, fsdp_seam):
    model, expected = lora_model
    unit = _CpuFSDP(model.base_model.model.outer)
    visible = list(unit.named_parameters())
    assert [name for name, _ in visible] == ["_flat_param"]
    assert isinstance(visible[0][1], FlatParameter)
    assert any("lora_" in name for name in visible[0][1]._fqns)
    root = _root(model, {"base_model.model.outer": unit}, fsdp_seam)

    actual = fsdp_utils.layered_summon_lora_params(root)

    _assert_adapter_equals(actual, expected)
    assert fsdp_seam.events == [("lazy_init", root), ("summon", unit)]


def test_lora_leaf_local_weight_gets_global_prefix_before_real_peft_filter(lora_model, fsdp_seam):
    model, expected = lora_model
    units = OrderedDict()
    for name, module in model.named_modules():
        if name.endswith(("lora_A.default", "lora_B.default")):
            assert list(dict(module.named_parameters())) == ["weight"]
            units[name] = _CpuFSDP(module)
    assert len(units) == 4
    first_name, first_unit = next(iter(units.items()))
    weight = first_unit._fsdp_wrapped_module.weight
    # Exercise the installed PEFT's real filter, with no filter monkeypatch.
    assert not get_peft_model_state_dict(model, state_dict={"weight": weight})
    globally_named = get_peft_model_state_dict(model, state_dict={first_name + ".weight": weight})
    assert set(globally_named) == {first_name.replace(".default", "") + ".weight"}
    root = _root(model, units, fsdp_seam)

    actual = fsdp_utils.layered_summon_lora_params(root)

    _assert_adapter_equals(actual, expected)
    summoned = [unit for event, unit in fsdp_seam.events if event == "summon"]
    assert Counter(map(id, summoned)) == Counter(map(id, units.values()))


def test_original_parameter_keys_remain_peft_global_names(lora_model, fsdp_seam):
    model, expected = lora_model
    unit = _CpuFSDP(model.base_model.model.outer, flattened=False)
    assert any("lora_A.default.weight" in name for name, _ in unit.named_parameters())
    root = _root(model, {"base_model.model.outer": unit}, fsdp_seam)

    actual = fsdp_utils.layered_summon_lora_params(root)

    _assert_adapter_equals(actual, expected)
    assert all("_fsdp_wrapped_module" not in name for name in actual)


def test_nested_units_initialize_root_and_gather_only_owned_adapters_once(lora_model, fsdp_seam):
    model, expected = lora_model
    body = model.base_model.model
    child = _CpuFSDP(body.outer.inner)
    parent = _CpuFSDP(body.outer, children={"inner": child})
    base_only = _CpuFSDP(body.unadapted)
    root = _root(model, {"base_model.model.outer": parent, "base_model.model.unadapted": base_only}, fsdp_seam)

    actual = fsdp_utils.layered_summon_lora_params(root)

    _assert_adapter_equals(actual, expected)
    assert fsdp_seam.events == [("lazy_init", root), ("summon", parent), ("summon", child)]
    assert root._is_root is True
    assert parent._is_root is child._is_root is base_only._is_root is False
    assert not parent.gathered and not child.gathered


def test_nested_lora_leaf_flat_parameter_is_excluded_after_summon_name_cleanup(lora_model, fsdp_seam):
    model, expected = lora_model
    body = model.base_model.model
    lora_layer = body.outer.q_proj
    frozen_base = _CpuFSDP(lora_layer.base_layer)
    nested_a = _CpuFSDP(lora_layer.lora_A.default)
    parent = _CpuFSDP(lora_layer, children={"base_layer": frozen_base, "lora_A.default": nested_a})
    other_layer = _CpuFSDP(body.outer.inner)
    root = _root(
        model,
        {"base_model.model.outer.q_proj": parent, "base_model.model.outer.inner": other_layer},
        fsdp_seam,
    )
    raw_child_paths = {name for name, unit in parent.named_modules() if unit is nested_a}
    assert raw_child_paths == {"_fsdp_wrapped_module.lora_A.default"}

    actual = fsdp_utils.layered_summon_lora_params(root)

    _assert_adapter_equals(actual, expected)
    assert not any("_flat_param" in name for name in actual)
    assert fsdp_seam.events == [("lazy_init", root), ("summon", parent), ("summon", nested_a), ("summon", other_layer)]
    assert not frozen_base.gathered


def test_collected_cpu_tensors_own_storage_after_summon_exits(lora_model, fsdp_seam):
    model, expected = lora_model
    live = get_peft_model_state_dict(model)
    unit = _CpuFSDP(model.base_model.model.outer)
    root = _root(model, {"base_model.model.outer": unit}, fsdp_seam)

    actual = fsdp_utils.layered_summon_lora_params(root)

    _assert_adapter_equals(actual, expected)
    assert not unit.gathered
    for name, value in actual.items():
        assert value.untyped_storage().data_ptr() != live[name].untyped_storage().data_ptr()
    with torch.no_grad():
        for value in live.values():
            value.add_(1000)
    _assert_adapter_equals(actual, expected)


def test_empty_layered_collection_fails_before_any_full_root_summon(lora_model, fsdp_seam):
    model, _ = lora_model
    # A real adapter exists, but this layout has no discoverable owning FSDP
    # child. Silently gathering the complete model would defeat layered mode.
    root = _root(model, {}, fsdp_seam)

    with pytest.raises(RuntimeError, match="refusing full-model fallback"):
        fsdp_utils.collect_lora_params(root, layered_summon=True, base_sync_done=True)

    assert fsdp_seam.events == [("lazy_init", root)]


def test_fsdp2_materializes_real_cpu_parameters_without_fsdp1_initialization_or_summon(
    lora_model, fsdp_seam, monkeypatch
):
    model, expected = lora_model
    calls = []

    def forbidden_fsdp1(*args, **kwargs):
        raise AssertionError("FSDP2 collection must not use FSDP1 lazy-init or summon")

    class UnusedFSDP1:
        summon_full_params = staticmethod(forbidden_fsdp1)

    # Keep the FSDP1 and FSDP2 type seams distinct: fsdp_version checks v1 first.
    monkeypatch.setattr(fsdp_utils, "FSDP", UnusedFSDP1)
    monkeypatch.setattr(fsdp_utils, "FSDPModule", _CpuFSDP)
    monkeypatch.setattr(fsdp_utils, "_lazy_init", forbidden_fsdp1)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue

        def full_tensor(param=param, name=name):
            calls.append(name.replace(".default", ""))
            return param.detach().clone()

        # A real CPU Parameter stands in for the DTensor materialization seam.
        monkeypatch.setattr(param, "full_tensor", full_tensor, raising=False)
    unit = _CpuFSDP(model.base_model.model.outer, flattened=False)
    root = _root(model, {"base_model.model.outer": unit}, fsdp_seam)
    assert fsdp_utils.fsdp_version(root) == fsdp_utils.fsdp_version(unit) == 2

    actual = fsdp_utils.layered_summon_lora_params(root)

    _assert_adapter_equals(actual, expected)
    assert Counter(calls) == Counter(expected.keys())
    assert fsdp_seam.events == []
    live = get_peft_model_state_dict(model)
    _assert_adapter_equals(live, expected)
    for name, value in actual.items():
        assert value.untyped_storage().data_ptr() != live[name].untyped_storage().data_ptr()


def test_fsdp2_root_only_layout_preserves_filtered_adapter_fallback(lora_model, fsdp_seam, monkeypatch):
    model, expected = lora_model

    def forbidden_fsdp1(*args, **kwargs):
        raise AssertionError("FSDP2 root-only export must not summon the complete model")

    class UnusedFSDP1:
        summon_full_params = staticmethod(forbidden_fsdp1)

    monkeypatch.setattr(fsdp_utils, "FSDP", UnusedFSDP1)
    monkeypatch.setattr(fsdp_utils, "FSDPModule", _CpuFSDP)
    monkeypatch.setattr(fsdp_utils, "_lazy_init", forbidden_fsdp1)
    root = _root(model, {}, fsdp_seam)
    assert fsdp_utils.fsdp_version(root) == 2

    actual = fsdp_utils.collect_lora_params(root, layered_summon=True, base_sync_done=True)

    _assert_adapter_equals(actual, expected)
    assert fsdp_seam.events == []


class _BiasOnlyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.25, -0.5, 0.75, -1.0]))


class _TinyBiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.extra = _BiasOnlyModule()
        self.unadapted = nn.Linear(4, 4, bias=False)


@pytest.mark.parametrize("flattened", [True, False])
@pytest.mark.parametrize("bias", ["all", "none"])
def test_separate_bias_unit_matches_real_peft_export(bias, flattened, fsdp_seam):
    model = get_peft_model(_TinyBiasModel(), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], bias=bias))
    expected = {name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(model).items()}
    bias_name = "base_model.model.extra.bias"
    assert (bias_name in expected) is (bias == "all")
    body = model.base_model.model
    adapter = _CpuFSDP(body.q_proj, flattened=flattened)
    bias_only = _CpuFSDP(body.extra, flattened=flattened)
    unrelated = _CpuFSDP(body.unadapted, flattened=flattened)
    assert all("lora_" not in name for name, _ in bias_only.owned_parameters)
    assert [name for name, _ in bias_only.owned_parameters] == ["bias"]
    root = _root(
        model,
        {
            "base_model.model.q_proj": adapter,
            "base_model.model.extra": bias_only,
            "base_model.model.unadapted": unrelated,
        },
        fsdp_seam,
    )

    actual = fsdp_utils.collect_lora_params(root, layered_summon=True, base_sync_done=True)

    _assert_adapter_equals(actual, expected)
    expected_events = [("lazy_init", root), ("summon", adapter)]
    if bias == "all":
        expected_events.append(("summon", bias_only))
    assert fsdp_seam.events == expected_events


def _skipped_container_layout(skip_kind, owns_adapter, fsdp_seam):
    body = nn.Module()
    body.layers = _Block()
    body.layers.inner = _Block()
    model = get_peft_model(body, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    with torch.no_grad():
        for index, (name, param) in enumerate(model.named_parameters(), start=1):
            if "lora_" in name:
                param.fill_(index / 10)
    expected = {name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(model).items()}
    assert len(expected) == 4
    child = _CpuFSDP(body.layers.inner)
    local_units = {"inner": child}
    if not owns_adapter:
        local_units["q_proj"] = _CpuFSDP(body.layers.q_proj)

    if skip_kind == "root":
        units = {f"base_model.model.layers.{name}": unit for name, unit in local_units.items()}
        root = _root(model, units, fsdp_seam)
        skipped = root
    elif skip_kind == "model":
        units = {f"layers.{name}": unit for name, unit in local_units.items()}
        skipped = _CpuFSDP(body, children=units)
        root = _root(model, {"base_model.model": skipped}, fsdp_seam)
    else:
        assert skip_kind == "layers"
        skipped = _CpuFSDP(body.layers, children=local_units)
        root = _root(model, {"base_model.model.layers": skipped}, fsdp_seam)
    return root, skipped, child, expected


@pytest.mark.parametrize("skip_kind", ["root", "model", "layers"])
def test_skipped_fsdp1_owner_rejects_nonempty_partial_adapter_export(skip_kind, fsdp_seam):
    root, skipped, child, expected = _skipped_container_layout(skip_kind, True, fsdp_seam)
    # The skipped unit exposes only metadata before summon; its LoRA keys are
    # hidden in the real FlatParameter's original-name metadata.
    assert skipped.flat_parameter is not None
    assert any("lora_" in name for name in skipped.flat_parameter._fqns)
    assert "_flat_param" in dict(skipped.named_parameters())
    child_names = {name.replace(".default", "") for name in expected if ".inner." in name}
    assert child_names and child_names < set(expected)

    with pytest.raises(RuntimeError, match="skipped FSDP1 unit"):
        fsdp_utils.collect_lora_params(root, layered_summon=True, base_sync_done=True)

    assert fsdp_seam.events == [("lazy_init", root)]
    assert not skipped.gathered and not child.gathered


@pytest.mark.parametrize("skip_kind", ["root", "model", "layers"])
def test_skipped_fsdp1_container_excludes_nested_adapter_owners(skip_kind, fsdp_seam):
    root, skipped, child, expected = _skipped_container_layout(skip_kind, False, fsdp_seam)
    if skipped.flat_parameter is not None:
        assert not any("lora_" in name for name in skipped.flat_parameter._fqns)
    # The container sees its child's flattened metadata recursively. It must
    # normalize wrapper segments and exclude that ownership before deciding
    # whether the skipped container itself has an unexportable adapter.
    assert any(
        "lora_" in original_name
        for _, param in skipped.named_parameters()
        for original_name in (getattr(param, "_fqns", None) or ())
    )

    actual = fsdp_utils.collect_lora_params(root, layered_summon=True, base_sync_done=True)

    _assert_adapter_equals(actual, expected)
    assert ("summon", child) in fsdp_seam.events
    assert all(unit is not skipped for event, unit in fsdp_seam.events if event == "summon")
    assert not skipped.gathered and not child.gathered

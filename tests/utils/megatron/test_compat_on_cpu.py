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

"""Compatibility selection must preserve native methods and per-model ownership."""

import gc
import inspect
from weakref import WeakKeyDictionary, ref

import pytest
import torch

from verl.models.mcore import compat


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(compat, "_checked_global", set())
    monkeypatch.setattr(compat, "_checked_models", WeakKeyDictionary())


def test_native_capability_leaves_callable_untouched():
    class Provider:
        def forward(self, *, output_processor=None):
            return output_processor

    original = Provider.forward

    def forbidden():
        raise AssertionError("Native capability must bypass compatibility installation")

    assert not compat.install_patch(
        "mtp_postprocess",
        forbidden,
        needed=lambda: "output_processor" not in inspect.signature(Provider.forward).parameters,
    )
    assert Provider.forward is original


def test_install_once_and_failure_can_retry():
    attempts = []

    def install():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("installation failed")

    with pytest.raises(RuntimeError, match="installation failed"):
        compat.install_patch("recompute", install, needed=lambda: True)
    assert compat.install_patch("recompute", install, needed=lambda: True)
    assert not compat.install_patch("recompute", install, needed=lambda: True)
    assert len(attempts) == 2


def test_model_scopes_do_not_retain_models():
    calls = []
    first, second = torch.nn.Module(), torch.nn.Module()
    for model in (first, first, second):
        compat.install_patch("mtp_postprocess", lambda: calls.append(1), needed=lambda: True, owner=model)
    assert len(calls) == 2
    first_ref = ref(first)
    del first
    gc.collect()
    assert first_ref() is None
    assert len(compat._checked_models) == 1


def test_recheck_native_before_installer_cache():
    checks = []
    compat.install_patch("mla", lambda: None, needed=lambda: True)
    assert not compat.install_patch("mla", lambda: None, needed=lambda: checks.append("native") or False)
    assert checks == ["native"]


def test_mtp_setup_is_idempotent_and_can_enable_detach_later(monkeypatch):
    from verl.models.mcore import mtp_patch

    calls = []
    monkeypatch.setattr(mtp_patch, "patch_postprocess", lambda model: calls.append("output"))
    monkeypatch.setattr(mtp_patch, "patch_mtp_layer_checkpointed_forward", lambda model: calls.append("checkpoint"))
    monkeypatch.setattr(mtp_patch, "patch_mtp_layer_get_embeddings", lambda model: calls.append("embeddings"))
    model = torch.nn.Module()
    compat.apply_model_patches(model, mtp_enabled=True)
    compat.apply_model_patches(model, mtp_enabled=True, detach_encoder=True)
    compat.apply_model_patches(model, mtp_enabled=True, detach_encoder=True)
    assert calls == ["output", "checkpoint", "embeddings"]


def test_zero_mtp_reference_only_installs_postprocess(monkeypatch):
    from verl.models.mcore import mtp_patch

    calls = []
    monkeypatch.setattr(mtp_patch, "patch_postprocess", lambda model: calls.append(model))
    model = torch.nn.Module()
    compat.apply_model_patches(model, forward_only=True, mtp_num_layers=0)
    compat.apply_model_patches(model, forward_only=True, mtp_num_layers=0)
    assert calls == [model]


def test_unaffected_writer_never_imports_legacy_installer(monkeypatch):
    calls = []
    monkeypatch.setattr(compat, "_legacy", lambda name: lambda: calls.append(name))
    compat.apply_bootstrap_patches()
    compat.apply_bootstrap_patches()
    assert calls == ["neutralize_broken_flash_attn_cute"]


def test_unknown_patch_fails_before_mutating():
    with pytest.raises(KeyError):
        compat.install_patch("not_registered", lambda: pytest.fail("mutated"), needed=lambda: True)


def test_forward_setup_uses_shared_compatibility_boundary(monkeypatch):
    from verl.models.mcore import model_forward_fused

    calls = []
    model, model_config, engine_config = object(), object(), object()
    monkeypatch.setattr(
        model_forward_fused, "patch_fused_forward", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    compat.configure_forward(model, model_config, engine_config=engine_config)
    assert calls == [((model, model_config), {"engine_config": engine_config})]

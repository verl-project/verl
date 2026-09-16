# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

"""Check the optional FA4 guard without importing Megatron or GPU libraries."""

import builtins
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[3] / "examples/real_nvfp4/runtime_backports/patch_megatron_fa4.py"
spec = importlib.util.spec_from_file_location("fa4_guard_backport", path)
backport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backport)


@pytest.mark.parametrize("version", [None, "4.0.0b19", "4.0.0b20", "4.0.0"])
def test_distribution_checked_before_optional_import(monkeypatch, version):
    def metadata_version(name):
        assert name == "flash-attn-4"
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(importlib.metadata, "version", metadata_version)
    original_import = builtins.__import__
    imported = []
    sentinel = object()

    def checked_import(name, *args, **kwargs):
        if name == "flash_attn.cute":
            imported.append(name)
            return SimpleNamespace(flash_attn_varlen_func=sentinel)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    namespace = {}
    exec(backport.NEW, namespace)
    enabled = version in ("4.0.0b20", "4.0.0")
    assert namespace["HAVE_FA4"] is enabled
    assert len(imported) == int(enabled)
    assert namespace["flash_attn4_varlen_func"] is (sentinel if enabled else None)


def test_broken_supported_install_not_silently_disabled(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "4.0.0b20")
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "flash_attn.cute":
            raise AttributeError("broken supported FA4 installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(AttributeError, match="broken supported FA4"):
        exec(backport.NEW, {})


def test_source_hash_guard_and_idempotence(monkeypatch):
    original = "# fixture\n" + backport.OLD + "\n# end\n"
    monkeypatch.setattr(backport, "BASE_SHA256", hashlib.sha256(original.encode()).hexdigest())
    patched = backport.patched_source(original)
    assert backport.patched_source(patched) == patched
    backport.verify_patched_source(patched)
    with pytest.raises(RuntimeError):
        backport.patched_source(original + "# unexpected change\n")
    with pytest.raises(RuntimeError):
        backport.verify_patched_source(patched + "# unexpected change\n")

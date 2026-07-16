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
"""Hermetic CPU tests for verl/models/mcore/bridge.py ImportError guidance (#7071).

Loads bridge.py by path so we never import the package tree as
``verl.models.mcore.bridge``. Isolation uses Pattern A (fake parent with empty
``__path__``) / controlled fakes so a host with real megatron-bridge installed
still exercises the three message branches.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import sys
import types
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE_PATH = _REPO_ROOT / "verl" / "models" / "mcore" / "bridge.py"
_PIN = "0.5.0"


def _load_bridge_by_path() -> None:
    """Execute bridge.py as a fresh module. Raises ImportError from its except."""
    name = f"_bridge_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, _BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Do not register in sys.modules permanently; just exec.
    spec.loader.exec_module(module)


@contextmanager
def _isolated_import_env(
    *,
    version_value: str | None,
    install_fakes: bool,
    train_utils_attrs: dict | None = None,
    force_import_error: bool = False,
) -> Iterator[None]:
    """Save/restore sys.modules + meta_path; install hermetic megatron fakes.

    version_value:
      None -> PackageNotFoundError for \"megatron-bridge\"
      str  -> that version string
    install_fakes:
      False + force_import_error: Pattern A empty parent, no bridge submodule
      True: install megatron.bridge (+ nested train_utils) with controlled attrs
    force_import_error:
      if True with install_fakes, make train_utils import raise ImportError
    """
    saved_modules = {k: sys.modules.get(k) for k in list(sys.modules) if k == "megatron" or k.startswith("megatron.")}
    saved_meta_path = list(sys.meta_path)
    real_version = md.version

    def _version(name: str) -> str:
        if name == "megatron-bridge":
            if version_value is None:
                raise md.PackageNotFoundError(name)
            return version_value
        return real_version(name)

    # Drop any real megatron* entries so fakes win.
    for key in list(sys.modules):
        if key == "megatron" or key.startswith("megatron."):
            del sys.modules[key]

    md.version = _version  # type: ignore[assignment]
    try:
        if force_import_error and not install_fakes:
            # Pattern A: parent package present with empty path -> no submodule discovery.
            fake_megatron = types.ModuleType("megatron")
            fake_megatron.__path__ = []  # type: ignore[attr-defined]
            sys.modules["megatron"] = fake_megatron
        elif install_fakes:
            fake_megatron = types.ModuleType("megatron")
            fake_megatron.__path__ = []  # type: ignore[attr-defined]
            sys.modules["megatron"] = fake_megatron

            if force_import_error:
                # pin-broken: package metadata says 0.5.0 but import fails.
                class _RaiseBridgeFinder:
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "megatron.bridge" or fullname.startswith("megatron.bridge."):
                            raise ImportError(f"simulated pin-broken import: {fullname}")
                        return None

                sys.meta_path.insert(0, _RaiseBridgeFinder())
            else:
                # too-old: AutoBridge present; train_utils missing LinearForLastLayer.
                fake_bridge = types.ModuleType("megatron.bridge")
                fake_bridge.AutoBridge = object  # type: ignore[attr-defined]
                fake_bridge.__path__ = []  # type: ignore[attr-defined]
                sys.modules["megatron.bridge"] = fake_bridge
                fake_megatron.bridge = fake_bridge  # type: ignore[attr-defined]

                training = types.ModuleType("megatron.bridge.training")
                training.__path__ = []  # type: ignore[attr-defined]
                sys.modules["megatron.bridge.training"] = training
                fake_bridge.training = training  # type: ignore[attr-defined]

                utils = types.ModuleType("megatron.bridge.training.utils")
                utils.__path__ = []  # type: ignore[attr-defined]
                sys.modules["megatron.bridge.training.utils"] = utils
                training.utils = utils  # type: ignore[attr-defined]

                train_utils = types.ModuleType("megatron.bridge.training.utils.train_utils")
                attrs = train_utils_attrs or {}
                for k, v in attrs.items():
                    setattr(train_utils, k, v)
                sys.modules["megatron.bridge.training.utils.train_utils"] = train_utils
                utils.train_utils = train_utils  # type: ignore[attr-defined]

        yield
    finally:
        md.version = real_version  # type: ignore[assignment]
        sys.meta_path[:] = saved_meta_path
        # Remove any megatron* we introduced.
        for key in list(sys.modules):
            if key == "megatron" or key.startswith("megatron."):
                del sys.modules[key]
        # Restore prior entries (including None = was absent).
        for key, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod


@pytest.mark.parametrize(
    "case,version_value,install_fakes,train_utils_attrs,force_import_error,must_have,must_not",
    [
        (
            "absent",
            None,
            False,
            None,
            True,  # Pattern A empty parent
            ["not installed", "--no-deps", f"=={_PIN}"],
            [],
        ),
        (
            "too-old",
            "0.3.1",
            True,
            {"freeze_moe_router": object, "make_value_model": object},  # no LinearForLastLayer
            False,
            ["0.3.1", "too old", "--no-deps"],
            ["not installed"],
        ),
        (
            "pin-broken",
            _PIN,
            True,
            None,
            True,
            [_PIN, "megatron-core"],
            ["too old", "not installed"],
        ),
    ],
    ids=["absent", "too-old", "pin-broken"],
)
def test_bridge_import_error_message_branches(
    case,
    version_value,
    install_fakes,
    train_utils_attrs,
    force_import_error,
    must_have,
    must_not,
):
    with _isolated_import_env(
        version_value=version_value,
        install_fakes=install_fakes,
        train_utils_attrs=train_utils_attrs,
        force_import_error=force_import_error,
    ):
        with pytest.raises(ImportError) as exc_info:
            _load_bridge_by_path()

    msg = str(exc_info.value)
    for fragment in must_have:
        assert fragment in msg, f"case={case}: expected {fragment!r} in message:\n{msg}"
    for fragment in must_not:
        assert fragment not in msg, f"case={case}: did not expect {fragment!r} in message:\n{msg}"

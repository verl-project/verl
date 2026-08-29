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
"""Tests for :func:`verl.trainer.main_ppo._resolve_remote_backend_from_hydra_choice`.

CPU-only: compose the config with ``initialize_config_dir`` + ``compose``
and patch :class:`hydra.core.hydra_config.HydraConfig` to expose the
runtime ``choices`` mapping the hook reads.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

VERL_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = VERL_ROOT / "verl" / "trainer" / "config"


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _compose(overrides: list[str]):
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


class _FakeHydraCfg:
    """Just enough of HydraConfig to satisfy the hook."""

    def __init__(self, choices: dict[str, str]):
        self.runtime = type("_R", (), {"choices": choices})()


def _run_hook_with_choice(cfg, choice: str | None):
    """Invoke the hook with ``HydraConfig.get()`` patched to expose ``choice``."""
    from verl.trainer import main_ppo

    fake = _FakeHydraCfg({"remote_backend": choice} if choice is not None else {})
    with patch("verl.trainer.main_ppo.HydraConfig") as hydra_cfg:
        hydra_cfg.get.return_value = fake
        main_ppo._resolve_remote_backend_from_hydra_choice(cfg)


def test_default_compose_has_no_remote_backend_key():
    """No Hydra choice -> no mutation."""
    cfg = _compose(overrides=[])
    assert "remote_backend" not in cfg or cfg.remote_backend is None
    assert cfg.trainer.v1.trainer_mode == "sync"

    _run_hook_with_choice(cfg, None)
    assert cfg.trainer.get("remote_backend") in (None, "")
    assert cfg.trainer.v1.trainer_mode == "sync"


def test_hydra_choice_hook_populates_trainer_fields():
    """``remote_backend=<name>`` stamps trainer.remote_backend and flips
    trainer.v1.trainer_mode from the default ``sync`` to ``remote_backend``."""
    cfg = _compose(overrides=[])
    _run_hook_with_choice(cfg, "stub_test_backend")

    assert cfg.trainer.remote_backend == "stub_test_backend"
    assert cfg.trainer.v1.trainer_mode == "remote_backend"


def test_hydra_choice_hook_respects_explicit_trainer_mode():
    """A user-set trainer_mode wins; only the name field gets stamped."""
    cfg = _compose(overrides=["trainer.v1.trainer_mode=colocate_async"])
    _run_hook_with_choice(cfg, "stub_test_backend2")

    assert cfg.trainer.remote_backend == "stub_test_backend2"
    assert cfg.trainer.v1.trainer_mode == "colocate_async"

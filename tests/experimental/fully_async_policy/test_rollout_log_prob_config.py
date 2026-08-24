# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).parents[3] / "verl" / "experimental" / "fully_async_policy" / "config_validation.py"
_SPEC = importlib.util.spec_from_file_location("fully_async_config_validation", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_rollout_log_prob_config = _MODULE.validate_rollout_log_prob_config


def _make_config(*, bypass_mode: bool, calculate_log_probs: bool):
    return SimpleNamespace(
        algorithm=SimpleNamespace(rollout_correction=SimpleNamespace(bypass_mode=bypass_mode)),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(calculate_log_probs=calculate_log_probs)),
    )


def test_trainer_recomputed_log_probs_do_not_require_rollout_log_probs():
    config = _make_config(bypass_mode=False, calculate_log_probs=False)

    validate_rollout_log_prob_config(config)


def test_bypass_mode_requires_rollout_log_probs():
    config = _make_config(bypass_mode=True, calculate_log_probs=False)

    with pytest.raises(ValueError, match="bypass_mode=True requires"):
        validate_rollout_log_prob_config(config)


def test_bypass_mode_accepts_rollout_log_probs():
    config = _make_config(bypass_mode=True, calculate_log_probs=True)

    validate_rollout_log_prob_config(config)

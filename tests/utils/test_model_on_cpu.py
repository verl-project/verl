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

from types import SimpleNamespace  # Or use a mock object library
from unittest.mock import MagicMock, patch

import pytest

from verl.utils.model import load_valuehead_model, update_model_config


# Parametrize with different override scenarios
@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"param_a": 5, "new_param": "plain_added"},
        {"param_a": 2, "nested_params": {"sub_param_x": "updated_x", "sub_param_z": True}},
    ],
)
def test_update_model_config(override_kwargs):
    """
    Tests that update_model_config correctly updates attributes,
    handling both plain and nested overrides via parametrization.
    """
    # Create a fresh mock config object for each test case
    mock_config = SimpleNamespace(
        param_a=1, nested_params=SimpleNamespace(sub_param_x="original_x", sub_param_y=100), other_param="keep_me"
    )
    # Apply the updates using the parametrized override_kwargs
    update_model_config(mock_config, override_kwargs)

    # Assertions to check if the config was updated correctly
    if "nested_params" in override_kwargs:  # Case 2: Nested override
        override_nested = override_kwargs["nested_params"]
        assert mock_config.nested_params.sub_param_x == override_nested["sub_param_x"], "Nested sub_param_x mismatch"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert hasattr(mock_config.nested_params, "sub_param_z"), "Expected nested sub_param_z to be added"
        assert mock_config.nested_params.sub_param_z == override_nested["sub_param_z"], "Value of sub_param_z mismatch"
    else:  # Case 1: Plain override (nested params untouched)
        assert mock_config.nested_params.sub_param_x == "original_x", "Nested sub_param_x should be unchanged"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert not hasattr(mock_config.nested_params, "sub_param_z"), "Nested sub_param_z should not exist"


# ---------------------------------------------------------------------------
# load_valuehead_model: attn_implementation precedence
# ---------------------------------------------------------------------------
# These tests pin down the attention-implementation selection contract so the
# documented ``critic.model.override_config.attn_implementation`` knob keeps
# working on hardware that cannot run ``flash_attention_2`` (e.g. Turing T4
# and other pre-Ampere GPUs).
#
# Resolution order inside ``load_valuehead_model``:
#   1. explicit ``attn_implementation`` kwarg
#   2. ``getattr(model_config, "_attn_implementation", None)``
#   3. ``"flash_attention_2"`` (historical default)


def _make_model_config(_attn_implementation=None):
    """Build a minimal stand-in for a ``transformers`` ``PretrainedConfig``.

    Only ``_attn_implementation`` is populated, mimicking what
    ``AutoConfig.from_pretrained(..., attn_implementation=...)`` (and verl's
    ``HFModelConfig``) would bake in.
    """
    cfg = SimpleNamespace()
    if _attn_implementation is not None:
        cfg._attn_implementation = _attn_implementation
    return cfg


@pytest.fixture
def mock_token_classification_from_pretrained():
    """Patch ``AutoModelForTokenClassification.from_pretrained`` so the test stays CPU-only."""
    with patch("transformers.AutoModelForTokenClassification.from_pretrained") as mocked:
        mocked.return_value = MagicMock(name="fake-valuehead-model")
        yield mocked


def test_load_valuehead_model_explicit_attn_implementation_wins(mock_token_classification_from_pretrained):
    """An explicit ``attn_implementation`` argument must override the config default."""
    model_config = _make_model_config(_attn_implementation="flash_attention_2")

    load_valuehead_model(
        local_path="/tmp/fake",
        torch_dtype="float32",
        model_config=model_config,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )

    assert mock_token_classification_from_pretrained.call_count == 1
    kwargs = mock_token_classification_from_pretrained.call_args.kwargs
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["pretrained_model_name_or_path"] == "/tmp/fake"
    assert kwargs["config"] is model_config
    assert kwargs["trust_remote_code"] is False


@pytest.mark.parametrize("impl", ["sdpa", "eager", "flash_attention_2"])
def test_load_valuehead_model_falls_back_to_model_config(mock_token_classification_from_pretrained, impl):
    """When no explicit value is given, honour ``model_config._attn_implementation``.

    Regression guard for the documented
    ``critic.model.override_config.attn_implementation`` override: previously
    ``load_valuehead_model`` hard-coded ``"flash_attention_2"`` and silently
    ignored whatever the caller had baked into ``model_config``.
    """
    model_config = _make_model_config(_attn_implementation=impl)

    load_valuehead_model(
        local_path="/tmp/fake",
        torch_dtype="float32",
        model_config=model_config,
        trust_remote_code=False,
    )

    assert mock_token_classification_from_pretrained.call_args.kwargs["attn_implementation"] == impl


def test_load_valuehead_model_defaults_to_flash_attention_2(mock_token_classification_from_pretrained):
    """Preserve the historical default for callers that pre-date the knob."""
    model_config = _make_model_config(_attn_implementation=None)

    load_valuehead_model(
        local_path="/tmp/fake",
        torch_dtype="float32",
        model_config=model_config,
        trust_remote_code=False,
    )

    assert mock_token_classification_from_pretrained.call_args.kwargs["attn_implementation"] == "flash_attention_2"


def test_load_valuehead_model_explicit_none_attr_falls_back(mock_token_classification_from_pretrained):
    """``_attn_implementation`` explicitly set to ``None`` must still fall back to FA2."""
    model_config = SimpleNamespace(_attn_implementation=None)

    load_valuehead_model(
        local_path="/tmp/fake",
        torch_dtype="float32",
        model_config=model_config,
        trust_remote_code=False,
    )

    assert mock_token_classification_from_pretrained.call_args.kwargs["attn_implementation"] == "flash_attention_2"

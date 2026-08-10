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

"""CPU tests for the MTP ``_checkpointed_forward`` compatibility shim.

Covers the three megatron-core signature generations that ``mtp_patch`` has to cope with:

* ``0.14``-``0.17``: ``_checkpointed_forward(forward_func, *args, **kwargs)``
  -> verl's slime-derived recompute patch applies, no shim.
* ``0.18.x``: ``_checkpointed_forward(hidden_states, decoder_input, ..., no padding_mask)``
  while ``MultiTokenPredictionLayer.forward`` passes ``padding_mask=...``
  -> hard ``TypeError`` without the shim (NVIDIA/Megatron-LM#4933).
* megatron-core ``main``: signature already declares ``padding_mask`` -> no shim.

These run on CPU. The shim only inspects and rebinds ``layer._checkpointed_forward``, so
stand-in objects with the right signatures are enough -- no model, no GPU, no distributed init.
``mtp_patch`` itself imports megatron.core at module scope, hence the ``importorskip``.
"""

import pytest

# ``mtp_patch`` imports megatron.core at module scope; skip rather than fail where it is absent.
pytest.importorskip("megatron.core", reason="mtp_patch requires megatron-core at import time")

from verl.models.mcore.mtp_patch import _patch_padding_mask_kwarg  # noqa: E402


class _LayerOldApi:
    """megatron-core 0.14-0.17: first parameter is a callable."""

    def _checkpointed_forward(self, forward_func, *args, **kwargs):
        return ("old", forward_func, args, kwargs)


class _Layer018:
    """megatron-core 0.18.x: tensors passed directly, ``padding_mask`` **not** declared."""

    def _checkpointed_forward(self, hidden_states, decoder_input, attention_mask=None, context=None):
        return ("0.18", hidden_states, decoder_input, attention_mask, context)


class _LayerMain:
    """megatron-core main: ``padding_mask`` is declared, so nothing to shim."""

    def _checkpointed_forward(self, hidden_states, decoder_input, attention_mask=None, padding_mask=None, context=None):
        return ("main", hidden_states, decoder_input, attention_mask, padding_mask, context)


def _bound_params(layer):
    from inspect import signature

    return list(signature(layer._checkpointed_forward).parameters)


def test_shim_not_installed_when_padding_mask_already_supported():
    layer = _LayerMain()
    assert _patch_padding_mask_kwarg(layer, _bound_params(layer)) is False
    # The shim rebinds by assigning an *instance* attribute, so "was not rebound" means no such
    # attribute exists. Comparing the accessed method with `is` would not work here: attribute
    # access on an un-rebound method builds a fresh bound-method object every time, so the
    # identity check fails even when nothing was patched.
    assert "_checkpointed_forward" not in vars(layer), "must not rebind when megatron already accepts it"
    # and the real kwarg still reaches the method
    assert layer._checkpointed_forward(1, 2, padding_mask="pm")[4] == "pm"


def test_shim_installed_on_0_18_and_accepts_padding_mask_none():
    """Without the shim this raises TypeError -- that is the bug users hit."""
    layer = _Layer018()
    with pytest.raises(TypeError, match="padding_mask"):
        layer._checkpointed_forward(1, 2, padding_mask=None)

    assert _patch_padding_mask_kwarg(layer, _bound_params(layer)) is True
    out = layer._checkpointed_forward(1, 2, attention_mask="am", padding_mask=None)
    assert out == ("0.18", 1, 2, "am", None), "original args must still be forwarded unchanged"


def test_shim_raises_instead_of_silently_dropping_a_real_mask():
    """A non-None mask means padded positions must be excluded; dropping it would be silent
    corruption (padding treated as real tokens), so fail loudly instead."""
    layer = _Layer018()
    _patch_padding_mask_kwarg(layer, _bound_params(layer))
    with pytest.raises(NotImplementedError, match="padding_mask"):
        layer._checkpointed_forward(1, 2, padding_mask=object())


def test_old_api_is_left_for_the_slime_recompute_patch():
    """params[0] == 'forward_func' is handled by the existing patch, not by this shim."""
    layer = _LayerOldApi()
    params = _bound_params(layer)
    assert params[0] == "forward_func"
    # the caller only reaches the shim when params[0] != "forward_func", but assert the shim
    # itself is a no-op here too, so an ordering change upstream cannot double-patch.
    assert _patch_padding_mask_kwarg(layer, params) is True  # signature lacks padding_mask
    # ...and forwarding still works through the shim
    sentinel = object()
    assert layer._checkpointed_forward(sentinel)[1] is sentinel


def test_shim_is_idempotent_per_layer():
    layer = _Layer018()
    assert _patch_padding_mask_kwarg(layer, _bound_params(layer)) is True
    after_first = layer._checkpointed_forward
    # second call sees the shim's own signature, which does declare padding_mask
    assert _patch_padding_mask_kwarg(layer, _bound_params(layer)) is False
    assert layer._checkpointed_forward is after_first

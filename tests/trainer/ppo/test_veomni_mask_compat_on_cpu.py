# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
from transformers import Qwen3_5MoeTextConfig
from transformers.masking_utils import create_causal_mask

from verl.utils.veomni.mask_compat import _uncached_causal_mask_adapter, install_qwen_uncached_mask_compat


@pytest.mark.parametrize("packed,padded", [(False, False), (True, False), (False, True)])
def test_uncached_mask_preserves_real_causal_semantics(packed, padded):
    config = Qwen3_5MoeTextConfig()
    config._attn_implementation = "eager"
    positions = torch.tensor([[0, 1, 2, 0, 1]]) if packed else torch.arange(5).unsqueeze(0)
    padding = torch.tensor([[0, 1, 1, 1, 1]]) if padded else None
    kwargs = dict(
        config=config,
        inputs_embeds=torch.randn(1, 5, 8),
        attention_mask=padding,
        past_key_values=None,
        position_ids=positions,
    )
    actual = _uncached_causal_mask_adapter(create_causal_mask)(cache_position=torch.arange(5), **kwargs)
    torch.testing.assert_close(actual, create_causal_mask(**kwargs))
    allowed = torch.ones(5, 5, dtype=torch.bool).tril()
    if packed:
        allowed[3:, :3] = False
    if padded:
        allowed[:, 0] = False
    assert torch.equal(actual[0, 0] == 0, allowed)


def test_rejects_cache_and_noncanonical_positions():
    call = _uncached_causal_mask_adapter(lambda **kwargs: kwargs)
    with pytest.raises(ValueError, match="uncached"):
        call(past_key_values=object())
    with pytest.raises(ValueError, match="span"):
        call(inputs_embeds=torch.zeros(1, 3, 4), cache_position=torch.arange(2))
    with pytest.raises(RuntimeError, match="start at zero"):
        call(inputs_embeds=torch.zeros(1, 3, 4), cache_position=torch.arange(1, 4))


def test_version_and_model_guards_do_not_import_veomni(monkeypatch):
    import verl.utils.veomni.mask_compat as compat

    monkeypatch.setattr(compat.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    assert not install_qwen_uncached_mask_compat("other")
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda name: "0.1.12")
    assert not install_qwen_uncached_mask_compat("qwen3_5_moe")
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda name: "0.1.11" if name == "veomni" else "5.13.0")
    assert not install_qwen_uncached_mask_compat("qwen3_5_moe")


def test_install_is_local_and_idempotent(monkeypatch):
    from types import SimpleNamespace

    from transformers import masking_utils

    import verl.utils.veomni.mask_compat as compat

    module = SimpleNamespace(create_causal_mask=create_causal_mask)
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda name: "0.1.11" if name == "veomni" else "5.12.1")
    assert install_qwen_uncached_mask_compat("qwen3_5_moe")
    installed = module.create_causal_mask
    assert installed is not create_causal_mask
    assert install_qwen_uncached_mask_compat("qwen3_5_moe")
    assert module.create_causal_mask is installed
    assert masking_utils.create_causal_mask is create_causal_mask


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("packed,padded", [(False, False), (True, False), (False, True), (True, True)])
def test_qwen3_installed_helpers_preserve_actual_mask_semantics(monkeypatch, sliding, packed, padded):
    from types import SimpleNamespace

    from transformers import Qwen3MoeConfig, masking_utils
    from transformers.masking_utils import create_sliding_window_causal_mask

    import verl.utils.veomni.mask_compat as compat

    module = SimpleNamespace(
        create_causal_mask=create_causal_mask, create_sliding_window_causal_mask=create_sliding_window_causal_mask
    )
    imported = []

    def import_module(name):
        imported.append(name)
        return module

    monkeypatch.setattr(compat.importlib, "import_module", import_module)
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda name: "0.1.11" if name == "veomni" else "5.12.1")
    assert compat.install_qwen_uncached_mask_compat("qwen3_moe")
    assert imported == ["veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu"]
    installed = (module.create_causal_mask, module.create_sliding_window_causal_mask)
    assert compat.install_qwen_uncached_mask_compat("qwen3_moe")
    assert installed == (module.create_causal_mask, module.create_sliding_window_causal_mask)
    assert masking_utils.create_causal_mask is create_causal_mask
    assert masking_utils.create_sliding_window_causal_mask is create_sliding_window_causal_mask

    config = Qwen3MoeConfig(use_sliding_window=sliding, sliding_window=2 if sliding else None)
    config._attn_implementation = "eager"
    positions = torch.tensor([[0, 1, 2, 0, 1]]) if packed else torch.arange(5).unsqueeze(0)
    kwargs = dict(
        config=config,
        inputs_embeds=torch.randn(1, 5, 8),
        attention_mask=torch.tensor([[0, 1, 1, 1, 1]]) if padded else None,
        past_key_values=None,
        position_ids=positions,
    )
    original = create_sliding_window_causal_mask if sliding else create_causal_mask
    adapted = module.create_sliding_window_causal_mask if sliding else module.create_causal_mask
    actual = adapted(cache_position=torch.arange(5), **kwargs)
    torch.testing.assert_close(actual, original(**kwargs))
    rows, columns = torch.arange(5)[:, None], torch.arange(5)[None, :]
    allowed = columns <= rows
    if sliding:
        allowed &= rows - columns < 2
    # Transformers infers packed blocks only when no explicit 2D padding mask is supplied.
    if packed and not padded:
        allowed[3:, :3] = False
    if padded:
        allowed[:, 0] = False
    assert torch.equal(actual[0, 0] == 0, allowed)
    with pytest.raises(ValueError, match="uncached"):
        adapted(**{**kwargs, "past_key_values": object()}, cache_position=torch.arange(5))


@pytest.mark.parametrize("veomni_version,transformers_version", [("0.1.12", "5.12.1"), ("0.1.11", "5.13.0")])
def test_qwen3_other_versions_do_not_import(monkeypatch, veomni_version, transformers_version):
    import verl.utils.veomni.mask_compat as compat

    monkeypatch.setattr(compat.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    monkeypatch.setattr(
        compat.importlib.metadata, "version", lambda name: veomni_version if name == "veomni" else transformers_version
    )
    assert not compat.install_qwen_uncached_mask_compat("qwen3_moe")

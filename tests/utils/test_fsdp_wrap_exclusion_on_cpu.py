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

"""CPU-only tests for keeping a frozen vision tower out of FSDP wrapping.

``actor.freeze_vision_tower`` freezes the vision tower and excludes it from
per-module FSDP wrapping (both FSDP1 and FSDP2), so its params join the root
FSDP unit. These tests verify the wrap-target selection on both FSDP paths
plus the ``get_vl_model_vision_tower`` helper.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.fsdp_utils import _select_fsdp2_wrap_targets, get_fsdp_wrap_policy
from verl.utils.model import get_vl_model_vision_tower


class _MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)


class _MockVisionBlock(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)


class _MockVisionTower(nn.Module):
    def __init__(self, hidden_size=32, depth=2):
        super().__init__()
        self.patch_embed = nn.Linear(12, hidden_size)
        self.blocks = nn.ModuleList([_MockVisionBlock(hidden_size) for _ in range(depth)])

    def forward(self, pixel_values, grid_thw=None):
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block.fc2(F.gelu(block.fc1(x)))
        return x


class _MockVLM(nn.Module):
    """Minimal vision-language model: vision tower + decoder layers + embed/lm_head."""

    _no_split_modules = ["_MockDecoderLayer", "_MockVisionBlock"]

    def __init__(self, vocab_size=64, hidden_size=32, num_layers=2):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.visual = _MockVisionTower(hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_MockDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)


class _VisionOnlyVLM(_MockVLM):
    """A model whose ``_no_split_modules`` lists only the vision block class."""

    _no_split_modules = ["_MockVisionBlock"]


def _wrap_classes_from_policy(policy):
    """Extract the transformer class set from the FSDP1 policy partial chain."""
    or_policy = policy.func
    assert or_policy.__name__ == "_or_policy", "expected _or_policy at the top level"
    transformer_policy = policy.keywords["policies"][-1]
    return transformer_policy.keywords["transformer_layer_cls"]


def test_fsdp1_policy_excludes_frozen_vision_tower():
    model = _MockVLM()
    policy = get_fsdp_wrap_policy(model, exclude_modules=[model.visual])
    classes = _wrap_classes_from_policy(policy)
    assert classes == {_MockDecoderLayer}, f"expected only decoder layers, got {classes}"


def test_fsdp1_policy_without_exclusion_wraps_vision_blocks():
    model = _MockVLM()
    policy = get_fsdp_wrap_policy(model)
    classes = _wrap_classes_from_policy(policy)
    assert classes == {_MockDecoderLayer, _MockVisionBlock}, f"expected both classes, got {classes}"


def test_fsdp1_policy_allows_exclusion_removing_all_classes():
    """Excluding the only wrap class wraps nothing per-module (root unit holds
    everything), matching the FSDP2 path; genuinely missing classes still raise."""
    import pytest

    model = _VisionOnlyVLM()
    assert get_fsdp_wrap_policy(model) is not None, "without exclusion the policy should build"
    assert get_fsdp_wrap_policy(model, exclude_modules=[model.visual]) is None, (
        "exclusion removing all wrap classes must build no per-module policy, not raise"
    )

    class _MissingClassVLM(_MockVLM):
        _no_split_modules = ["_NoSuchLayerClass"]

    with pytest.raises(Exception, match="Could not find any of the transformer layer classes"):
        get_fsdp_wrap_policy(_MissingClassVLM())


def _lora_leaf(where):
    """Attach a trainable LoRA-like linear leaf, as PEFT does."""
    leaf = nn.Linear(32, 32)
    where.lora_A = leaf
    return leaf


def _lambda_fn_from_policy(policy):
    """Extract the LoRA lambda_fn from the FSDP1 policy partial chain."""
    lambda_policy = policy.keywords["policies"][0]
    return lambda_policy.keywords["lambda_fn"]


def test_fsdp1_lora_policy_excludes_leaves_inside_excluded_tower():
    """Trainable LoRA leaves inside the frozen, wrap-excluded tower must not be
    wrapped individually: a rank that skips the tower would never enter those
    units and desynchronize collective order (#3491). They join the root unit."""
    model = _MockVLM()
    model.visual.requires_grad_(False)
    tower_leaf = _lora_leaf(model.visual)
    lang_leaf = _lora_leaf(model.layers[0])

    lambda_fn = _lambda_fn_from_policy(get_fsdp_wrap_policy(model, is_lora=True, exclude_modules=[model.visual]))
    assert lambda_fn(tower_leaf) is False, "tower LoRA leaf must not be individually wrapped"
    assert lambda_fn(lang_leaf) is True, "language LoRA leaf must stay individually wrapped"


def test_fsdp1_lora_policy_without_exclusion_wraps_tower_leaves():
    """Without exclusion (freeze off) the LoRA policy is unchanged."""
    model = _MockVLM()
    tower_leaf = _lora_leaf(model.visual)

    lambda_fn = _lambda_fn_from_policy(get_fsdp_wrap_policy(model, is_lora=True))
    assert lambda_fn(tower_leaf) is True, "no exclusion: LoRA leaves are wrapped as before"


def test_mark_frozen_vision_tower_for_skip_gates_on_trainable_params():
    """The engine marks only visual paths with no trainable params anywhere:
    with the path wrap-excluded into the root FSDP unit, a trainable member
    would get grads on image ranks only on mixed-modality batches, which FSDP1's
    root-unit gradient reduce does not support (verified to hang). The tower
    (first module) is the one marked."""
    import pytest

    from verl.utils.fsdp_utils import mark_frozen_vision_tower_for_skip

    tower = _MockVisionTower()
    tower.requires_grad_(False)
    projector = nn.Linear(32, 32)  # frozen sibling, as the engine freezes it
    projector.requires_grad_(False)
    assert mark_frozen_vision_tower_for_skip([tower, projector]) is True
    assert getattr(tower, "_verl_frozen_vision_tower", False) is True
    assert not hasattr(projector, "_verl_frozen_vision_tower")

    tower_with_adapter = _MockVisionTower()
    tower_with_adapter.requires_grad_(False)
    _lora_leaf(tower_with_adapter)  # trainable leaf inside the tower, as PEFT attaches
    assert mark_frozen_vision_tower_for_skip([tower_with_adapter]) is False
    assert not hasattr(tower_with_adapter, "_verl_frozen_vision_tower")

    # A trainable sibling (e.g. an adapter on multi_modal_projector) is rejected
    # outright: the dummy branches only run the tower, so nothing would align a
    # trainable sibling across ranks — better a build-time error than the
    # mixed-modality FSDP hang it would produce.
    frozen_tower = _MockVisionTower()
    frozen_tower.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable params on a visual-path sibling"):
        mark_frozen_vision_tower_for_skip([frozen_tower, nn.Linear(32, 32)])
    assert not hasattr(frozen_tower, "_verl_frozen_vision_tower")


def test_get_vlm_visual_path_modules_includes_frozen_siblings():
    """The freeze/exclusion unit is the visual path: the tower plus sibling
    modules that also run only on images (Mllama-style multi_modal_projector,
    MiniCPM-style resampler). Qwen/GLM keep the projector inside the tower, so
    nothing extra is found there."""
    from verl.utils.model import get_vlm_visual_path_modules

    tower = _MockVisionTower()
    projector = nn.Linear(32, 32)

    class _MllamaStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = tower
            self.multi_modal_projector = projector
            self.language_model = nn.Module()

    class _MiniCPMSibling(nn.Module):
        def __init__(self):
            super().__init__()
            self.vpm = tower
            self.resampler = projector
            self.llm = nn.Module()

    class _QwenStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = tower  # merger lives inside the tower

    class _TextOnly(nn.Module):
        pass

    assert get_vlm_visual_path_modules(_MllamaStyle()) == [tower, projector]
    assert get_vlm_visual_path_modules(_MiniCPMSibling()) == [tower, projector]
    assert get_vlm_visual_path_modules(_QwenStyle()) == [tower]
    assert get_vlm_visual_path_modules(_TextOnly()) == []


def test_get_vlm_visual_path_modules_finds_nested_sibling_through_value_head_wrapper():
    """A projector nested one level deeper than the tower must be found too:
    TRL value-head critic over a Gemma3-style layout keeps both tower and
    projector under ``pretrained_model.model``. A missed sibling would stay
    trainable outside the visual-path list while the tower is still marked for
    the dummy-forward skip — the asymmetric-trainable-param hang this PR
    documents."""
    from verl.utils.model import get_vlm_visual_path_modules

    tower = _MockVisionTower()
    projector = nn.Linear(32, 32)

    class _Gemma3Backbone(nn.Module):
        # transformers >= 4.52 Gemma3ForConditionalGeneration layout
        def __init__(self):
            super().__init__()
            self.vision_tower = tower
            self.multi_modal_projector = projector
            self.language_model = nn.Module()

    class _Gemma3ForConditionalGeneration(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Gemma3Backbone()

    class _ValueHeadWrapper(nn.Module):
        # TRL AutoModelForCausalLMWithValueHead layout
        def __init__(self, inner):
            super().__init__()
            self.pretrained_model = inner
            self.v_head = nn.Linear(32, 32)

    model = _ValueHeadWrapper(_Gemma3ForConditionalGeneration())
    assert get_vlm_visual_path_modules(model) == [tower, projector]


def test_get_vl_model_vision_tower_skips_non_module_attributes():
    """The generic probe names (``vision``, ``visual``) may carry non-module
    values (config strings, paths) on some models; only nn.Module values are
    tower hits — a non-module must be skipped, not returned and later crashed
    on by requires_grad_()."""
    from verl.utils.model import get_vl_model_vision_tower

    tower = _MockVisionTower()

    class _StrAttrBeforeTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = "some/config/path"  # probed before vision_tower, not a Module
            self.vision_tower = tower

    class _OnlyStrAttr(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = {"remote": "code"}

    assert get_vl_model_vision_tower(_StrAttrBeforeTower()) is tower
    assert get_vl_model_vision_tower(_OnlyStrAttr()) is None


def test_fsdp2_targets_exclude_frozen_vision_tower():
    model = _MockVLM()
    targets = _select_fsdp2_wrap_targets(
        model, ["_MockDecoderLayer", "_MockVisionBlock"], exclude_modules=[model.visual]
    )
    target_classes = {type(m) for m in targets}
    assert _MockVisionBlock not in target_classes, "vision blocks must not be wrapped"
    assert _MockDecoderLayer in target_classes, "decoder layers must still be wrapped"
    # embed_tokens / lm_head are matched by name and must survive the exclusion.
    assert any(isinstance(m, nn.Embedding) for m in targets), "embed_tokens should remain a wrap target"
    # No module inside the tower may appear.
    tower_ids = {id(sub) for sub in model.visual.modules()}
    assert not (tower_ids & {id(m) for m in targets}), "no tower submodule may be a wrap target"


def test_fsdp2_targets_without_exclusion_include_vision_blocks():
    model = _MockVLM()
    targets = _select_fsdp2_wrap_targets(model, ["_MockDecoderLayer", "_MockVisionBlock"])
    target_classes = {type(m) for m in targets}
    assert _MockVisionBlock in target_classes, "vision blocks should be wrapped without exclusion"
    assert _MockDecoderLayer in target_classes


def test_get_vl_model_vision_tower_layouts():
    tower = _MockVisionTower()

    class _TopLevel(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = tower

    class _Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _TopLevel()

    class _ValueHeadWrapper(nn.Module):
        """TRL-style wrapper: the model lives under ``pretrained_model``."""

        def __init__(self, inner):
            super().__init__()
            self.pretrained_model = inner

    class _KimiStyle(nn.Module):
        """Kimi-VL layout: the encoder is ``vision_tower``, the text model ``model``."""

        def __init__(self):
            super().__init__()
            self.vision_tower = tower
            self.model = nn.Module()  # text model, no tower inside

    class _KimiStyleNested(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _KimiStyle()

    class _MllamaStyle(nn.Module):
        """Mllama layout: the encoder is ``vision_model``."""

        def __init__(self):
            super().__init__()
            self.vision_model = tower
            self.language_model = nn.Module()

    class _MiniCPMStyle(nn.Module):
        """MiniCPM-o layout: the encoder is ``vpm``, the language model ``llm``."""

        def __init__(self):
            super().__init__()
            self.vpm = tower
            self.llm = nn.Module()

    # Every family in verl.utils.model.VISION_TOWER_ATTRS gets one direct,
    # one nested and one value-head-wrapped case — new families extend both
    # the constant and this loop.
    from verl.utils.model import VISION_TOWER_ATTRS

    class _AttrTower(nn.Module):
        def __init__(self, attr):
            super().__init__()
            setattr(self, attr, tower)
            self.language_model = nn.Module()

    class _NestedHolder(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

    for _attr in VISION_TOWER_ATTRS:
        _direct = _AttrTower(_attr)
        assert get_vl_model_vision_tower(_direct) is tower, _attr
        assert get_vl_model_vision_tower(_NestedHolder(_direct)) is tower, _attr
        assert get_vl_model_vision_tower(_ValueHeadWrapper(_direct)) is tower, _attr

    class _TextOnly(nn.Module):
        pass

    assert get_vl_model_vision_tower(_TopLevel()) is tower
    assert get_vl_model_vision_tower(_Nested()) is tower
    assert get_vl_model_vision_tower(_ValueHeadWrapper(_TopLevel())) is tower
    assert get_vl_model_vision_tower(_ValueHeadWrapper(_Nested())) is tower
    assert get_vl_model_vision_tower(_KimiStyle()) is tower
    assert get_vl_model_vision_tower(_KimiStyleNested()) is tower
    assert get_vl_model_vision_tower(_ValueHeadWrapper(_KimiStyle())) is tower
    assert get_vl_model_vision_tower(_MllamaStyle()) is tower
    assert get_vl_model_vision_tower(_ValueHeadWrapper(_MllamaStyle())) is tower
    assert get_vl_model_vision_tower(_MiniCPMStyle()) is tower
    assert get_vl_model_vision_tower(_ValueHeadWrapper(_MiniCPMStyle())) is tower
    assert get_vl_model_vision_tower(_TextOnly()) is None


def test_fsdp1_size_based_policy_with_exclusion_fails_closed():
    """min_num_params > 0 selects the size policy, which cannot carry the
    exclusion — and "keep the dummy visual forward" only protects families
    whose adapter has a dummy branch, so the combination is rejected at build
    instead of hanging the first mixed-modality batch on other families."""
    import pytest

    model = _MockVLM()
    with pytest.raises(ValueError, match="not supported with the size-based FSDP1 wrap policy"):
        get_fsdp_wrap_policy(model, config={"min_num_params": 1000}, exclude_modules=[model.visual])

    # Without exclusion the size policy builds unchanged.
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    policy = get_fsdp_wrap_policy(model, config={"min_num_params": 1000})
    inner = policy.keywords["policies"]
    assert len(inner) == 1, "only the size policy should be built"
    assert inner[0].func is size_based_auto_wrap_policy


def test_get_vlm_visual_path_modules_finds_deepseek_vl2_bare_projector():
    """DeepseekVLV2ForCausalLM keeps its image connector as the bare top-level
    ``projector`` beside the ``vision`` tower — both must join the visual path."""
    from verl.utils.model import get_vlm_visual_path_modules

    tower = _MockVisionTower()
    projector = nn.Linear(32, 32)

    class _DeepseekVL2Style(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = tower
            self.projector = projector
            self.language = nn.Module()

    assert get_vlm_visual_path_modules(_DeepseekVL2Style()) == [tower, projector]


def test_get_vlm_visual_path_modules_follows_post_peft_sibling_replacement():
    """PEFT/QAT replace the attribute of a targeted leaf, so a sibling held as
    a direct child is re-parented to an adapter wrapper. The discovery must
    follow the live tree: the engine re-runs it after adapters attach, and the
    wrap exclusion / marker gate then cover the post-adapter objects."""
    from verl.utils.model import get_vlm_visual_path_modules

    class _MllamaStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = _MockVisionTower()
            self.multi_modal_projector = nn.Linear(32, 32)
            self.language_model = nn.Module()

    model = _MllamaStyle()
    before = get_vlm_visual_path_modules(model)

    replacement = nn.Linear(32, 32)  # what PEFT's lora.Linear looks like from outside
    model.multi_modal_projector = replacement

    after = get_vlm_visual_path_modules(model)
    assert after[0] is before[0], "the tower container is never replaced"
    assert after[1] is replacement, "the sibling must be re-discovered post-replacement"


def test_get_vlm_visual_path_params_finds_llava_next_image_newline():
    """LlavaNext registers image_newline as a top-level nn.Parameter beside the
    tower: it runs only on the image path, so it must be discovered and frozen
    with the tower (a trainable one gets gradients on image ranks only)."""
    from verl.utils.model import get_vlm_visual_path_params

    class _LlavaNextStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = _MockVisionTower()
            self.multi_modal_projector = nn.Linear(32, 32)
            self.image_newline = nn.Parameter(torch.zeros(32))
            self.language_model = nn.Module()

    class _NotAParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = _MockVisionTower()
            self.image_newline = "configs/are/not/params"

    params = get_vlm_visual_path_params(_LlavaNextStyle())
    assert len(params) == 1 and isinstance(params[0], nn.Parameter)
    assert get_vlm_visual_path_params(_NotAParam()) == [], "non-Parameter values must not match"


def test_mark_rejects_trainable_bare_visual_path_param():
    """A trainable bare visual-path parameter (e.g. image_newline re-enabled
    post-freeze) is rejected like a trainable sibling: no dummy-forward
    mechanism covers it, so it would receive gradients on image ranks only."""
    import pytest

    from verl.utils.fsdp_utils import mark_frozen_vision_tower_for_skip

    tower = _MockVisionTower()
    tower.requires_grad_(False)
    frozen_newline = nn.Parameter(torch.zeros(32))
    frozen_newline.requires_grad_(False)
    trainable_newline = nn.Parameter(torch.zeros(32))

    assert mark_frozen_vision_tower_for_skip([tower], visual_path_params=[frozen_newline]) is True
    with pytest.raises(ValueError, match="trainable params on a visual-path sibling"):
        mark_frozen_vision_tower_for_skip([tower], visual_path_params=[trainable_newline])


def test_get_vlm_visual_path_params_finds_deepseek_vl2_view_seperator():
    """DeepSeek-VL2's remote code registers a second bare image-only parameter,
    ``view_seperator`` (upstream spelling), beside the tower."""
    from verl.utils.model import get_vlm_visual_path_params

    class _DeepseekVL2Style(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = _MockVisionTower()
            self.projector = nn.Linear(32, 32)
            self.view_seperator = nn.Parameter(torch.zeros(1))
            self.language = nn.Module()

    params = get_vlm_visual_path_params(_DeepseekVL2Style())
    assert len(params) == 1 and isinstance(params[0], nn.Parameter)


def test_mark_rejects_trainable_tower_without_dummy_forward_protection():
    """Trainable params inside the tower are only safe for families whose verl
    adapter runs the dummy visual forward (VL_DUMMY_VISUAL_FORWARD_MODEL_TYPES);
    a model without such an adapter has nothing aligning the excluded trainable
    tower across ranks, so the combination is rejected at build."""
    import pytest

    from verl.utils.fsdp_utils import mark_frozen_vision_tower_for_skip

    tower_with_adapter = _MockVisionTower()
    tower_with_adapter.requires_grad_(False)
    _lora_leaf(tower_with_adapter)  # trainable leaf inside the tower, as PEFT attaches

    # Protected family (default True): warning + not marked, dummy stays on.
    assert mark_frozen_vision_tower_for_skip([tower_with_adapter]) is False
    assert not hasattr(tower_with_adapter, "_verl_frozen_vision_tower")

    # Unprotected family: rejected at build.
    tower_unprotected = _MockVisionTower()
    tower_unprotected.requires_grad_(False)
    _lora_leaf(tower_unprotected)
    with pytest.raises(ValueError, match="only safe for families whose verl adapter"):
        mark_frozen_vision_tower_for_skip([tower_unprotected], tower_protected_by_dummy=False)


def test_vl_dummy_visual_forward_model_types_match_the_patch_dispatch():
    """The protection set is the single source of truth shared with the
    patch_forward_with_backends dispatch — pin the members so a new family
    added to one place trips this test until the other is updated."""
    import inspect

    from verl.models.transformers.monkey_patch import (
        VL_DUMMY_VISUAL_FORWARD_MODEL_TYPES,
        patch_forward_with_backends,
    )

    source = inspect.getsource(patch_forward_with_backends)
    for model_type in VL_DUMMY_VISUAL_FORWARD_MODEL_TYPES:
        assert f'"{model_type}"' in source, f"{model_type} missing from the patch dispatch"

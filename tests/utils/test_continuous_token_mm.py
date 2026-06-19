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
"""Unit tests for multimodal MergeResult fields and base class MM hooks."""

import pytest

from verl.utils.continuous_token import (
    ContinuousTokenBuilder,
    MergeResult,
    ct_align_response_metadata,
)


class TestMergeResultMultimodalFields:
    """Verify new MM fields have correct defaults and work with existing logic."""

    def test_default_values_text_only(self):
        """Text-only MergeResult should have empty/None MM fields."""
        result = MergeResult(token_ids=[1, 2, 3], appended_token_count=2)
        assert result.pixel_values is None
        assert result.image_grid_thw == []
        assert result.image_token_spans == []
        assert result.mm_processor_kwargs == {}

    def test_backward_compat_construction(self):
        """Old-style construction (no MM kwargs) should still work."""
        result = MergeResult(
            token_ids=[10, 20, 30],
            appended_token_count=1,
            kind="assistant",
            inserted_token_ids=[99],
            removed_prefix_token_count=0,
        )
        assert result.token_ids == [10, 20, 30]
        assert result.kind == "assistant"
        assert result.inserted_token_ids == [99]
        assert result.pixel_values is None
        assert result.image_grid_thw == []

    def test_mm_fields_populated(self):
        """MergeResult with MM fields set explicitly."""
        fake_pixels = [[0.1, 0.2, 0.3]]  # Simulate tensor as list for testing
        result = MergeResult(
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            appended_token_count=3,
            kind="non_assistant",
            pixel_values=fake_pixels,
            image_grid_thw=[(1, 4, 4)],
            image_token_spans=[(2, 6)],
            mm_processor_kwargs={"min_pixels": 3136},
        )
        assert result.pixel_values == fake_pixels
        assert result.image_grid_thw == [(1, 4, 4)]
        assert result.image_token_spans == [(2, 6)]
        assert result.mm_processor_kwargs == {"min_pixels": 3136}

    def test_frozen_immutability(self):
        """MergeResult should remain frozen (no assignment after construction)."""
        result = MergeResult(token_ids=[1], appended_token_count=0)
        with pytest.raises(AttributeError):
            result.pixel_values = "bad"  # type: ignore[misc]

    def test_align_response_metadata_ignores_mm_fields(self):
        """ct_align_response_metadata should work unchanged with MM fields present."""
        result = MergeResult(
            token_ids=[1, 2, 3, 4, 5],
            appended_token_count=2,
            kind="non_assistant",
            inserted_token_ids=[99],
            pixel_values="fake_tensor",
            image_grid_thw=[(1, 2, 2)],
            image_token_spans=[(1, 3)],
        )
        mask = [1, 1, 1]
        logprobs = [0.5, 0.6, 0.7]
        aligned_mask, aligned_logprobs = ct_align_response_metadata(
            result, mask, logprobs
        )
        # 1 inserted token (mask=0, logprob=0.0) + 2 non_assistant (mask=0, logprob=0.0)
        assert aligned_mask == [1, 1, 1, 0, 0, 0]
        assert aligned_logprobs == [0.5, 0.6, 0.7, 0.0, 0.0, 0.0]


class TestBaseClassMMHooks:
    """Verify base class MM hooks behave correctly (NotImplementedError / False)."""

    def setup_method(self):
        """Create a minimal mock tokenizer for base class instantiation."""

        class MockTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2, 3]

        self.builder = ContinuousTokenBuilder(MockTokenizer())

    def test_supports_multimodal_default_false(self):
        """Base class should return False for supports_multimodal."""
        assert ContinuousTokenBuilder.supports_multimodal() is False
        assert self.builder.supports_multimodal() is False

    def test_count_vision_tokens_raises(self):
        """Base class count_vision_tokens should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not implement count_vision_tokens"):
            self.builder.count_vision_tokens((1, 4, 4))

    def test_extract_vision_placeholders_raises(self):
        """Base class extract_vision_placeholders should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not implement extract_vision_placeholders"):
            self.builder.extract_vision_placeholders([1, 2, 3, 4])

    def test_render_tokens_with_mm_raises(self):
        """Base class render_tokens_with_mm should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not implement render_tokens_with_mm"):
            self.builder.render_tokens_with_mm(
                messages=[{"role": "user", "content": "hi"}],
                images=["fake_image.png"],
            )

    def test_supports_multimodal_classmethod(self):
        """supports_multimodal should be callable as classmethod without instance."""
        assert ContinuousTokenBuilder.supports_multimodal() is False

    def test_subclass_can_override_supports_multimodal(self):
        """A VL subclass that overrides supports_multimodal should return True."""

        class FakeVLBuilder(ContinuousTokenBuilder):
            @classmethod
            def supports_multimodal(cls) -> bool:
                return True

        assert FakeVLBuilder.supports_multimodal() is True


class TestMultimodalMergeResultWithExistingSubclasses:
    """Ensure existing text subclass _merge_token_ids still produce valid MergeResult."""

    def test_qwen_merge_still_works(self):
        """QwenContinuousTokenBuilder merge should produce MergeResult with empty MM fields."""
        from verl.utils.continuous_token import QwenContinuousTokenBuilder

        class MockQwenTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                if token == "<|im_end|>":
                    return 151645
                return 0

        builder = QwenContinuousTokenBuilder(MockQwenTokenizer())
        # Simulate: prefix ends with <|im_end|>, appended is [10, 20]
        result = builder._merge_token_ids([100, 200, 151645], [10, 20])
        assert result.token_ids == [100, 200, 151645, 198, 10, 20]
        assert result.inserted_token_ids == [198]
        assert result.appended_token_count == 2
        # MM fields should all be default
        assert result.pixel_values is None
        assert result.image_grid_thw == []
        assert result.image_token_spans == []
        assert result.mm_processor_kwargs == {}

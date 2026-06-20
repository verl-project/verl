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


# =============================================================================
# Tests for VL subclasses
# =============================================================================


class TestQwenVLContinuousTokenBuilder:
    """Test QwenVL vision token handling."""

    def setup_method(self):
        from verl.utils.continuous_token import QwenVLContinuousTokenBuilder

        class MockQwenVLTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockImageProcessor:
            merge_size = 2

        class MockProcessor:
            image_processor = MockImageProcessor()

        self.tokenizer = MockQwenVLTokenizer()
        self.processor = MockProcessor()
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_supports_multimodal(self):
        assert self.builder.supports_multimodal() is True

    def test_count_vision_tokens_single_image(self):
        """t=1, h=28, w=28, merge=2 → 1 * 14 * 14 = 196."""
        count = self.builder.count_vision_tokens((1, 28, 28))
        assert count == 196

    def test_count_vision_tokens_video(self):
        """t=4, h=14, w=14, merge=2 → 4 * 7 * 7 = 196."""
        count = self.builder.count_vision_tokens((4, 14, 14))
        assert count == 196

    def test_extract_vision_placeholders(self):
        """Should find vision spans between start and end markers."""
        # <|vision_start|>=151652, <|image_pad|>=151655, <|vision_end|>=151653
        token_ids = [1, 2, 151652, 151655, 151655, 151655, 151653, 3, 4]
        spans = self.builder.extract_vision_placeholders(token_ids)
        assert spans == [(3, 6)]  # indices of the pad tokens

    def test_extract_vision_placeholders_multiple_images(self):
        """Should find multiple vision spans."""
        token_ids = [
            151652, 151655, 151655, 151653,  # image 1: indices 1-3
            10, 20,
            151652, 151655, 151653,  # image 2: indices 7-8
        ]
        spans = self.builder.extract_vision_placeholders(token_ids)
        assert spans == [(1, 3), (7, 8)]

    def test_extract_vision_placeholders_no_images(self):
        """No vision markers → empty list."""
        token_ids = [1, 2, 3, 4, 5]
        spans = self.builder.extract_vision_placeholders(token_ids)
        assert spans == []

    def test_merge_inherits_qwen_newline_patch(self):
        """VL builder should still insert newline after im_end (from QwenBuilder)."""
        result = self.builder._merge_token_ids([100, 151645], [10, 20])
        assert result.token_ids == [100, 151645, 198, 10, 20]
        assert result.inserted_token_ids == [198]


class TestMiMoVLContinuousTokenBuilder:
    """Test MiMo-VL vision token handling."""

    def setup_method(self):
        from verl.utils.continuous_token import MiMoVLContinuousTokenBuilder

        class MockMiMoVLTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockImageProcessor:
            merge_size = 2

        class MockProcessor:
            image_processor = MockImageProcessor()

        self.tokenizer = MockMiMoVLTokenizer()
        self.processor = MockProcessor()
        self.builder = MiMoVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_supports_multimodal(self):
        assert self.builder.supports_multimodal() is True

    def test_count_vision_tokens(self):
        count = self.builder.count_vision_tokens((1, 28, 28))
        assert count == 196

    def test_extract_vision_placeholders(self):
        token_ids = [1, 151652, 151655, 151655, 151653, 2]
        spans = self.builder.extract_vision_placeholders(token_ids)
        assert spans == [(2, 4)]

    def test_merge_inherits_mimo_newline_patch(self):
        """MiMo-VL should still insert newline after im_end (from MiMoBuilder)."""
        result = self.builder._merge_token_ids([100, 151645], [10, 20])
        assert result.token_ids == [100, 151645, 198, 10, 20]
        assert result.inserted_token_ids == [198]


# =============================================================================
# Tests for wiring factory with VL families
# =============================================================================


class TestWiringVLFactory:
    """Test that create_continuous_token_builder handles VL families correctly."""

    def test_vl_family_requires_processor(self):
        """VL families should raise if processor not provided."""
        from verl.utils.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        with pytest.raises(ValueError, match="requires a processor"):
            create_continuous_token_builder(
                MockTokenizer(),
                model_family="qwen25vl",
            )

    def test_vl_family_succeeds_with_processor(self):
        """VL families should instantiate correctly with processor provided."""
        from verl.utils.continuous_token import QwenVLContinuousTokenBuilder
        from verl.utils.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockImageProcessor:
            merge_size = 2

        class MockProcessor:
            image_processor = MockImageProcessor()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            model_family="qwen25vl",
            processor=MockProcessor(),
        )
        assert isinstance(builder, QwenVLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True


# =============================================================================
# Integration tests: VL builder build_initial_tokens + merge_tokens end-to-end
# =============================================================================


class _MockQwenVLProcessor:
    """Mock processor that simulates Qwen2.5-VL image processing.

    For each image, expands to 4 image_pad tokens (simulating merge_size=2,
    image 2x4x4 -> t=1, h=4, w=4 -> 1*(4//2)*(4//2) = 4 patches).
    """

    class _ImageProcessor:
        merge_size = 2

    image_processor = _ImageProcessor()

    def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
        """Simulate processor output."""
        # Simple mock: for each image marker in text, expand to 4 pad tokens
        # The text should contain <|vision_start|><|image_pad|><|vision_end|> placeholders
        # We simulate by counting images and producing deterministic output
        num_images = len(images) if images else 0

        # Token IDs: produce a fixed sequence with vision spans expanded
        # Simplified: [BOS, vision_start, pad*4, vision_end, ..., text_tokens]
        token_ids = [151643]  # BOS
        for _ in range(num_images):
            token_ids.append(151652)  # vision_start
            token_ids.extend([151655] * 4)  # 4 image_pad tokens
            token_ids.append(151653)  # vision_end
        # Add some text tokens
        token_ids.extend([1000, 1001, 1002])  # mock text tokens

        result = {"input_ids": [token_ids]}
        if num_images > 0:
            # pixel_values dim0 = raw patches (t*h*w = 1*4*4 = 16 per image)
            import numpy as np

            result["pixel_values"] = np.zeros((num_images * 16, 3, 14, 14), dtype=np.float32)
            # image_grid_thw: each image is (1, 4, 4)
            result["image_grid_thw"] = np.array([[1, 4, 4]] * num_images, dtype=np.int64)

        return result


class _MockQwenVLTokenizer:
    """Mock tokenizer for VL integration tests."""

    name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    def encode(self, text, add_special_tokens=False):
        if text == "\n":
            return [198]
        return [1000, 1001, 1002]

    def convert_tokens_to_ids(self, token):
        mapping = {
            "<|im_end|>": 151645,
            "<|im_start|>": 151644,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
            "<|image_pad|>": 151655,
        }
        return mapping.get(token, 0)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        """Simple mock chat template."""
        tokens = [151644]  # im_start
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        tokens.extend([151652, 151655, 151653])  # vision placeholder
                    elif isinstance(block, dict) and block.get("type") == "text":
                        tokens.extend([1000, 1001])
            else:
                tokens.extend([1000, 1001, 1002])
            tokens.append(151645)  # im_end
        if add_generation_prompt:
            tokens.append(151644)  # im_start for assistant
        if not tokenize:
            return "mock_text_render"
        return tokens


class TestQwenVLBuildInitialTokens:
    """Integration test for QwenVL build_initial_tokens with images."""

    def setup_method(self):
        from verl.utils.continuous_token import QwenVLContinuousTokenBuilder

        self.tokenizer = _MockQwenVLTokenizer()
        self.processor = _MockQwenVLProcessor()
        self.builder = QwenVLContinuousTokenBuilder(
            self.tokenizer, self.processor
        )

    def test_build_initial_no_images(self):
        """Without images, should use text-only path."""
        messages = [{"role": "user", "content": "Hello"}]
        token_ids = self.builder.build_initial_tokens(messages)
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)
        assert self.builder._last_mm_extras == {}

    def test_build_initial_with_images(self):
        """With images, should use processor and store mm_extras."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "fake_image.png"},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        token_ids = self.builder.build_initial_tokens(messages)
        assert isinstance(token_ids, list)
        # Should have stored mm_extras
        mm_extras = self.builder._last_mm_extras
        assert "pixel_values" in mm_extras
        assert "image_grid_thw" in mm_extras
        assert len(mm_extras["image_grid_thw"]) == 1
        assert mm_extras["image_grid_thw"][0] == (1, 4, 4)


class TestQwenVLMergeTokens:
    """Integration test for QwenVL merge_tokens with images in appended messages."""

    def setup_method(self):
        from verl.utils.continuous_token import QwenVLContinuousTokenBuilder

        self.tokenizer = _MockQwenVLTokenizer()
        self.processor = _MockQwenVLProcessor()
        self.builder = QwenVLContinuousTokenBuilder(
            self.tokenizer, self.processor
        )

    def test_merge_no_new_images(self):
        """Without new images in appended messages, should use text-only merge."""
        previous = [{"role": "user", "content": "Hi"}]
        updated = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        result = self.builder.merge_tokens(previous, updated, runtime_ids)
        assert isinstance(result, MergeResult)
        assert result.pixel_values is None
        assert result.image_grid_thw == []

    def test_merge_with_new_images(self):
        """With new images in appended messages, should populate MM fields."""
        previous = [{"role": "user", "content": "Hi"}]
        updated = [
            {"role": "user", "content": "Hi"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "new_image.png"},
                    {"type": "text", "text": "Look at this"},
                ],
            },
        ]
        # Simulate runtime token state
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        result = self.builder.merge_tokens(previous, updated, runtime_ids)
        assert isinstance(result, MergeResult)
        # Should have MM fields populated
        assert result.pixel_values is not None
        assert len(result.image_grid_thw) == 1
        assert result.image_grid_thw[0] == (1, 4, 4)


class TestQwenVLSliceMmDelta:
    """Test _slice_mm_delta logic."""

    def setup_method(self):
        from verl.utils.continuous_token import QwenVLContinuousTokenBuilder

        self.tokenizer = _MockQwenVLTokenizer()
        self.processor = _MockQwenVLProcessor()
        self.builder = QwenVLContinuousTokenBuilder(
            self.tokenizer, self.processor
        )

    def test_all_images_new(self):
        """prev_image_count=0 should return full mm_extras."""
        import numpy as np

        full_extras = {
            "pixel_values": np.zeros((32, 3, 14, 14)),
            "image_grid_thw": [(1, 4, 4), (1, 4, 4)],
        }
        delta = self.builder._slice_mm_delta(prev_image_count=0, full_mm_extras=full_extras)
        assert delta is full_extras

    def test_partial_delta(self):
        """prev_image_count=1 with 2 total should return only second image."""
        import numpy as np

        # pixel_values dim0 = raw patches (t*h*w), not merged tokens
        # grid (1,4,4) -> raw patches = 1*4*4 = 16 per image, total = 32
        full_extras = {
            "pixel_values": np.zeros((32, 3, 14, 14)),
            "image_grid_thw": [(1, 4, 4), (1, 4, 4)],
        }
        delta = self.builder._slice_mm_delta(prev_image_count=1, full_mm_extras=full_extras)
        assert len(delta["image_grid_thw"]) == 1
        assert delta["image_grid_thw"][0] == (1, 4, 4)
        # pixel_values sliced at raw patch boundary: 16 patches for second image
        assert delta["pixel_values"].shape == (16, 3, 14, 14)

    def test_no_new_images(self):
        """prev_image_count >= total should return empty."""
        import numpy as np

        full_extras = {
            "pixel_values": np.zeros((16, 3, 14, 14)),
            "image_grid_thw": [(1, 4, 4)],
        }
        delta = self.builder._slice_mm_delta(prev_image_count=1, full_mm_extras=full_extras)
        assert delta == {}

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
"""Unit tests for multimodal continuous token builders and base class MM hooks."""

import pytest

from verl.utils.continuous_token import (
    ContinuousTokenBuilder,
    MergeResult,
)


class TestMergeResultTokenFields:
    """Verify MergeResult stays token-only and works with VL builders."""

    def test_default_values_text_only(self):
        result = MergeResult(token_ids=[1, 2, 3], appended_token_count=2)
        assert result.token_ids == [1, 2, 3]
        assert result.appended_token_count == 2
        assert result.kind == "non_assistant"

    def test_backward_compat_construction(self):
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

    def test_frozen_immutability(self):
        """MergeResult should remain frozen (no assignment after construction)."""
        result = MergeResult(token_ids=[1], appended_token_count=0)
        with pytest.raises(AttributeError):
            result.token_ids = [2]  # type: ignore[misc]


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
        """QwenContinuousTokenBuilder merge should produce token-only MergeResult."""
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
        result = builder._merge_non_assistant_token_ids([100, 200, 151645], [10, 20])
        assert result.token_ids == [100, 200, 151645, 198, 10, 20]
        assert result.inserted_token_ids == [198]
        assert result.appended_token_count == 2


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
            151652,
            151655,
            151655,
            151653,  # image 1: indices 1-3
            10,
            20,
            151652,
            151655,
            151653,  # image 2: indices 7-8
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
        result = self.builder._merge_non_assistant_token_ids([100, 151645], [10, 20])
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
        result = self.builder._merge_non_assistant_token_ids([100, 151645], [10, 20])
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

        # Token IDs: produce a fixed sequence with vision spans expanded.
        # The leading tokens simulate the synthetic prefix used by incremental
        # rendering, so prefix trimming can assert the invariant explicitly.
        token_ids = [1000, 1001, 1002]
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
            "<|observation|>": 151333,
            "<|user|>": 151336,
            "<|begin_of_image|>": 151700,
            "<|end_of_image|>": 151701,
            "<|media_start|>": 151800,
            "<|media_end|>": 151801,
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
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_build_initial_no_images(self):
        """Without images, should use text-only path."""
        messages = [{"role": "user", "content": "Hello"}]
        token_ids = self.builder.build_initial_tokens(messages)
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)

    def test_build_initial_with_images(self):
        """With images, should use processor-expanded token IDs."""
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
        assert token_ids.count(151655) == 4


class TestQwenVLMergeTokens:
    """Integration test for QwenVL merge_tokens with images in appended messages."""

    def setup_method(self):
        from verl.utils.continuous_token import QwenVLContinuousTokenBuilder

        self.tokenizer = _MockQwenVLTokenizer()
        self.processor = _MockQwenVLProcessor()
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

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
        assert result.kind == "non_assistant"

    def test_merge_with_new_images(self):
        """With new images in appended messages, should merge processor-expanded token IDs."""
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
        assert result.kind == "non_assistant"
        assert 151655 in result.token_ids

    def test_merge_with_new_images_rejects_non_prefix_processor_output(self):
        """Synthetic-prefix trimming should fail fast if processor output is not append-only."""

        class BadPrefixProcessor(_MockQwenVLProcessor):
            def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
                result = super().__call__(text=text, images=images, return_tensors=return_tensors, **kwargs)
                result["input_ids"][0][0] = 9999
                return result

        from verl.utils.continuous_token import QwenVLContinuousTokenBuilder

        builder = QwenVLContinuousTokenBuilder(self.tokenizer, BadPrefixProcessor())
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
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        with pytest.raises(ValueError, match="multimodal synthetic prefix"):
            builder.merge_tokens(previous, updated, runtime_ids)


@pytest.mark.parametrize(
    "builder_name",
    [
        "MiMoVLContinuousTokenBuilder",
        "GLM4VContinuousTokenBuilder",
        "KimiVLContinuousTokenBuilder",
    ],
)
def test_other_vl_builders_reject_non_prefix_processor_output(builder_name):
    """All VL builders should validate the synthetic-prefix token invariant."""

    class BadPrefixProcessor(_MockQwenVLProcessor):
        def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
            result = super().__call__(text=text, images=images, return_tensors=return_tensors, **kwargs)
            result["input_ids"][0][0] = 9999
            return result

    import verl.utils.continuous_token as continuous_token

    builder_cls = getattr(continuous_token, builder_name)
    builder = builder_cls(_MockQwenVLTokenizer(), BadPrefixProcessor())
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
    runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
    with pytest.raises(ValueError, match="multimodal synthetic prefix"):
        builder.merge_tokens(previous, updated, runtime_ids)

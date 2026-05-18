# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Test that apply_fsdp2's module selection handles peft-wrapped models.

peft wraps embed_tokens in a ModulesToSaveWrapper, so isinstance(module, nn.Embedding)
fails. Without name-based matching, embed_tokens + lm_head land in the root FSDP unit,
causing OOM from oversized allgather. These tests verify the module selection logic
works for: (1) vanilla models, (2) peft-wrapped models, (3) tied embeddings.
"""

import unittest
from types import SimpleNamespace

import torch.nn as nn

from verl.utils.fsdp_utils import _select_fsdp2_wrap_targets


class MockDecoderLayer(nn.Module):
    """Simulates a transformer decoder layer (e.g. Qwen3DecoderLayer)."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)


class MockModulesToSaveWrapper(nn.Module):
    """Simulates peft's ModulesToSaveWrapper around nn.Embedding.

    peft wraps modules listed in modules_to_save (like embed_tokens) in this wrapper,
    which breaks isinstance(module, nn.Embedding) checks.
    """

    def __init__(self, original_module):
        super().__init__()
        self.original_module = original_module
        self.weight = original_module.weight  # peft exposes weight


class MockCausalLM(nn.Module):
    """Simulates a causal LM with embed_tokens, decoder layers, and lm_head."""

    _no_split_modules = ["MockDecoderLayer"]

    def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2, tie_word_embeddings=False):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([MockDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


class TestFSDP2PeftWrapping(unittest.TestCase):
    """Test module selection in apply_fsdp2 for vanilla and peft-wrapped models."""

    def _get_wrapped_names(self, model, cls_names):
        """Return names of modules selected for wrapping."""
        selected = _select_fsdp2_wrap_targets(model, cls_names)
        # _select_fsdp2_wrap_targets returns module objects; map back to names
        module_to_name = {id(m): n for n, m in model.named_modules()}
        return [module_to_name[id(m)] for m in selected]

    def test_vanilla_model_wraps_layers_and_embedding(self):
        """Vanilla model (no peft): embed_tokens matched by isinstance, layers by class name."""
        model = MockCausalLM(tie_word_embeddings=False)
        names = self._get_wrapped_names(model, ["MockDecoderLayer"])

        self.assertIn("model.embed_tokens", names)
        self.assertIn("lm_head", names)
        self.assertTrue(any("layers.0" in n for n in names))
        self.assertTrue(any("layers.1" in n for n in names))

    def test_peft_wrapped_model_wraps_embed_tokens_by_name(self):
        """peft-wrapped model: embed_tokens fails isinstance but is matched by name."""
        model = MockCausalLM(tie_word_embeddings=False)
        original_embed = model.model.embed_tokens
        model.model.embed_tokens = MockModulesToSaveWrapper(original_embed)

        names = self._get_wrapped_names(model, ["MockDecoderLayer"])

        self.assertIn("model.embed_tokens", names)
        self.assertIn("lm_head", names)
        self.assertTrue(any("layers.0" in n for n in names))

    def test_tied_embeddings_skips_name_based_wrapping(self):
        """With tie_word_embeddings=True, embed_tokens/lm_head are NOT wrapped separately."""
        model = MockCausalLM(tie_word_embeddings=True)
        names = self._get_wrapped_names(model, ["MockDecoderLayer"])

        self.assertNotIn("model.embed_tokens", names)
        self.assertNotIn("lm_head", names)
        self.assertTrue(any("layers.0" in n for n in names))

    def test_peft_wrapped_tied_embeddings_skips_wrapping(self):
        """peft + tied embeddings: name-based matching is disabled, no wrapping."""
        model = MockCausalLM(tie_word_embeddings=True)
        original_embed = model.model.embed_tokens
        model.model.embed_tokens = MockModulesToSaveWrapper(original_embed)

        names = self._get_wrapped_names(model, ["MockDecoderLayer"])

        self.assertNotIn("model.embed_tokens", names)
        self.assertNotIn("lm_head", names)

    def test_no_duplicate_wrapping_for_vanilla_embedding(self):
        """Vanilla nn.Embedding should not be wrapped twice (by isinstance AND by name)."""
        model = MockCausalLM(tie_word_embeddings=False)
        names = self._get_wrapped_names(model, ["MockDecoderLayer"])

        embed_count = sum(1 for n in names if n == "model.embed_tokens")
        self.assertEqual(embed_count, 1, f"embed_tokens wrapped {embed_count} times, expected 1")


if __name__ == "__main__":
    unittest.main()

# Copyright 2025 Individual Contributor: Wu Zehuan
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

import unittest

import torch.nn as nn

from verl.utils.freeze_utils import apply_freeze_to_module


class DummyVLModel(nn.Module):
    """Minimal model with a vision-like subtree and a language head."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.encoder = nn.Linear(16, 16)
        self.model.visual.merger = nn.Linear(16, 32)
        self.model.language = nn.Linear(32, 64)
        self.lm_head = nn.Linear(64, 10)


class TestApplyFreezeToModule(unittest.TestCase):
    def test_no_pattern_does_nothing(self):
        model = DummyVLModel()
        apply_freeze_to_module(model, None)
        for p in model.parameters():
            self.assertTrue(p.requires_grad)

    def test_freeze_visual_subtree(self):
        model = DummyVLModel()
        apply_freeze_to_module(model, r"model\.visual\.")
        self.assertFalse(model.model.visual.encoder.weight.requires_grad)
        self.assertFalse(model.model.visual.merger.weight.requires_grad)
        self.assertTrue(model.model.language.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)

    def test_freeze_single_module(self):
        model = DummyVLModel()
        apply_freeze_to_module(model, r"model\.visual\.merger")
        self.assertTrue(model.model.visual.encoder.weight.requires_grad)  # not matched
        self.assertFalse(model.model.visual.merger.weight.requires_grad)

    def test_invalid_regex_raises(self):
        model = DummyVLModel()
        with self.assertRaisesRegex(ValueError, "Invalid freeze_module_pattern"):
            apply_freeze_to_module(model, r"[invalid")

    def test_megatron_works(self):
        model = DummyVLModel()
        apply_freeze_to_module(model, r"visual")

    def test_returns_frozen_count(self):
        model = DummyVLModel()
        count = apply_freeze_to_module(model, r"model\.visual\.")
        # encoder.weight, encoder.bias, merger.weight, merger.bias = 4 params
        self.assertEqual(count, 4)

    def test_anchor_based_exact_parameter_match(self):
        """Regex with ^...$ should match only the exact parameter."""
        model = DummyVLModel()
        apply_freeze_to_module(model, r"^model\.visual\.encoder\.weight$")
        self.assertFalse(model.model.visual.encoder.weight.requires_grad)
        self.assertTrue(model.model.visual.encoder.bias.requires_grad)  # not matched


if __name__ == "__main__":
    unittest.main()

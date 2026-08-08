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

"""Unit tests for translating verl's LoRA ``target_modules`` into SGLang's spelling.

``model.target_modules`` defaults to PEFT's ``"all-linear"`` shorthand, which PEFT resolves
against the loaded model. SGLang has no such shorthand and coerces the value with
``set(...)``, so forwarding the bare string tears it into characters and the LoRA memory
pool dies on one of them (``get_hidden_dim not implemented for i``). The regression these
tests guard is precisely that: the result must never be a set of single characters.

They also pin the deliberate refusal to guess. PEFT reads any other bare string as a
regex over the whole parameter key, and SGLang cannot express that, so it must fail
loudly rather than quietly adapt a different set of modules than training did.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.sglang_rollout.utils import sglang_lora_target_modules

SEVEN_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TestSGLangLoraTargetModules:
    def test_all_linear_becomes_sglang_all_sentinel(self):
        assert sglang_lora_target_modules("all-linear") == ["all"]

    def test_explicit_list_passes_through(self):
        assert sglang_lora_target_modules(SEVEN_PROJECTIONS) == SEVEN_PROJECTIONS

    def test_set_is_materialized_as_list(self):
        result = sglang_lora_target_modules({"q_proj", "v_proj"})
        assert isinstance(result, list)
        assert sorted(result) == ["q_proj", "v_proj"]

    @pytest.mark.parametrize("regex", [r".*\.q_proj", "q_proj", r"model\.layers\.\d+\.mlp\..*"])
    def test_regex_string_is_rejected_not_guessed(self, regex):
        # PEFT would re.fullmatch these against the full key. Wrapping them into a
        # one-element list would look like it worked while adapting the wrong modules.
        with pytest.raises(ValueError, match="regex"):
            sglang_lora_target_modules(regex)

    def test_rejection_message_names_the_alternatives(self):
        with pytest.raises(ValueError) as excinfo:
            sglang_lora_target_modules(r".*\.q_proj")
        message = str(excinfo.value)
        assert "all-linear" in message
        assert "q_proj" in message

    def test_no_accepted_input_is_ever_split_into_characters(self):
        # ServerArgs applies set() to whatever it receives; any single-character entry
        # means the string was iterated instead of listed.
        for value in ("all-linear", SEVEN_PROJECTIONS, {"q_proj"}):
            result = sglang_lora_target_modules(value)
            assert all(len(item) > 1 for item in result), f"{value!r} was split into {result!r}"

# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""SGLang follow-up to #7038: ban sampled vision placeholders via custom logits."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("sglang")

from sglang.srt.sampling.custom_logit_processor import (
    CustomLogitProcessor,
    DisallowedTokensLogitsProcessor,
)

from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    _merge_banned_token_params,
    _resolve_enable_custom_logit_processor,
)

IMAGE_PAD = 151655
VIDEO_PAD = 151656


class TestMergeBannedTokenParams:
    def test_preserves_unrelated_custom_params(self):
        params = {"custom_params": {"processor_mode": "strict"}}
        _merge_banned_token_params(params, [7, 9])
        assert params["custom_params"] == {"processor_mode": "strict", "token_ids": [7, 9]}

    def test_accepts_identical_caller_token_ids(self):
        params = {"custom_params": {"token_ids": [7, 9]}}
        _merge_banned_token_params(params, [7, 9])
        assert params["custom_params"]["token_ids"] == [7, 9]

    def test_rejects_conflicting_caller_token_ids(self):
        params = {"custom_params": {"token_ids": [3]}}
        with pytest.raises(ValueError, match="conflicts with the vision placeholder ban"):
            _merge_banned_token_params(params, [7, 9])

    def test_rejects_non_mapping_custom_params(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            _merge_banned_token_params({"custom_params": [1, 2]}, [7])


class TestLaunchFlagAgreesWithRequest:
    def test_bans_force_the_flag_on(self):
        assert _resolve_enable_custom_logit_processor([IMAGE_PAD], {}) is True

    def test_text_only_leaves_user_flag_on(self):
        assert _resolve_enable_custom_logit_processor([], {"enable_custom_logit_processor": True}) is True

    def test_text_only_without_user_flag_stays_off(self):
        assert _resolve_enable_custom_logit_processor([], {}) is False

    def test_explicit_false_conflicts_when_ids_are_banned(self):
        with pytest.raises(ValueError, match="enable_custom_logit_processor=False"):
            _resolve_enable_custom_logit_processor([IMAGE_PAD, VIDEO_PAD], {"enable_custom_logit_processor": False})


class TestDisallowedTokensLogitsProcessor:
    def _apply(self, logits, banned_token_ids):
        processor = CustomLogitProcessor.from_str(DisallowedTokensLogitsProcessor.to_str())
        custom_params = {"token_ids": banned_token_ids}
        return processor(logits, [custom_params] * logits.shape[0])

    def test_banned_columns_are_masked_and_the_rest_survive(self):
        vocab_size = 151680
        logits = torch.randn(4, vocab_size)
        original = logits.clone()
        masked = self._apply(logits, [IMAGE_PAD, VIDEO_PAD])
        assert torch.isneginf(masked[:, IMAGE_PAD]).all()
        assert torch.isneginf(masked[:, VIDEO_PAD]).all()
        keep = torch.ones(vocab_size, dtype=torch.bool)
        keep[[IMAGE_PAD, VIDEO_PAD]] = False
        torch.testing.assert_close(masked[:, keep], original[:, keep])

    def test_a_banned_id_can_never_win_the_argmax(self):
        logits = torch.full((2, 32), -10.0)
        logits[:, 7] = 100.0
        logits[:, 3] = 1.0
        masked = self._apply(logits, [7])
        assert masked.argmax(dim=-1).tolist() == [3, 3]


class TestGenerateReqInputCarriesProcessor:
    def test_request_keeps_processor_string_and_token_ids(self):
        from sglang.srt.managers.io_struct import GenerateReqInput

        banned = [IMAGE_PAD, VIDEO_PAD]
        sampling_params = {"max_new_tokens": 8, "custom_params": {"token_ids": banned}}
        request = GenerateReqInput(
            rid="req-0",
            input_ids=[1, 2, 3],
            sampling_params=sampling_params,
            custom_logit_processor=DisallowedTokensLogitsProcessor.to_str(),
        )
        request.normalize_batch_and_arguments()
        processor_str = request.custom_logit_processor
        if isinstance(processor_str, list):
            assert len(processor_str) == 1
            processor_str = processor_str[0]
        assert processor_str == DisallowedTokensLogitsProcessor.to_str()
        assert request.sampling_params["custom_params"] == {"token_ids": banned}

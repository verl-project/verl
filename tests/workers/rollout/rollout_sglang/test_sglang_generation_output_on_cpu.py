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

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from verl.workers.rollout.sglang_rollout.utils import extract_sglang_generation_output


@pytest.fixture
def parse_generation_output(monkeypatch):
    def extract_captured_data(output):
        return output["meta_info"]["routed_experts"]

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.state_capturer.routed_experts",
        SimpleNamespace(extract_routed_experts_from_meta_info=extract_captured_data),
    )
    config = SimpleNamespace(model_type="deepseek_v41", num_hidden_layers=2, num_experts_per_tok=1)

    def parse(*, num_route_rows, log_probs):
        output = {
            "output_ids": [11, 12],
            "meta_info": {
                "prompt_tokens": 6,
                "output_token_logprobs": [
                    (log_prob, token, None) for log_prob, token in zip(log_probs, [11, 12], strict=False)
                ],
                "routed_experts": np.arange(num_route_rows * 2, dtype=np.int32),
            },
        }
        return extract_sglang_generation_output(
            output,
            return_logprob=True,
            enable_routing_replay=True,
            hf_config=config,
            has_images=True,
            request_id="test",
        )

    return parse


@pytest.mark.parametrize("log_probs", [[], [-0.2]])
def test_partial_output_validates_routes_before_dropping_unscored_tokens(parse_generation_output, log_probs):
    assert parse_generation_output(num_route_rows=7, log_probs=log_probs) == ([], [], None)


def test_completed_output_keeps_validated_routes(parse_generation_output):
    tokens, log_probs, routes = parse_generation_output(num_route_rows=7, log_probs=[-0.2, -0.3])
    assert tokens == [11, 12]
    assert log_probs == [-0.2, -0.3]
    np.testing.assert_array_equal(routes, np.arange(14).reshape(7, 2, 1))


def test_completed_output_rejects_invalid_route_count(parse_generation_output):
    with pytest.raises(ValueError, match="5 route rows for 8 tokens"):
        parse_generation_output(num_route_rows=5, log_probs=[-0.2, -0.3])

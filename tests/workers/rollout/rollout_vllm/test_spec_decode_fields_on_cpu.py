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

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from vllm.v1.metrics.stats import RequestSpecDecodeMetrics

from verl.trainer.ppo.trainer_utils import compute_spec_decode_metrics
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def test_spec_decode_fields_count_the_drafts_vllm_accepted_per_request():
    """Three verify steps of three drafts each, accepting 3, 1 and 0: the rollout-common counts and their rates."""
    metrics = RequestSpecDecodeMetrics.new(num_spec_tokens=3)
    for accepted in (3, 1, 0):
        metrics.observe(num_draft_tokens=3, num_accepted=accepted)

    fields = vLLMHttpServer.spec_decode_fields(SimpleNamespace(spec_decode_metrics=metrics))

    assert fields == {"spec_num_draft_tokens": 9, "spec_num_accepted_tokens": 4, "spec_num_verify_steps": 3}
    rates = compute_spec_decode_metrics(
        spec_drafts=np.array([fields["spec_num_draft_tokens"]]),
        spec_accepts=np.array([fields["spec_num_accepted_tokens"]]),
        spec_verifies=np.array([fields["spec_num_verify_steps"]]),
    )
    assert rates["rollout/spec_accept_rate"] == pytest.approx(4 / 9)
    assert rates["rollout/spec_accept_length"] == pytest.approx(1 + 4 / 3)


def test_spec_decode_fields_of_an_aborted_request_are_zero():
    """An aborted request's final output carries no counts from vLLM: no drafts, no acceptances, no verify steps."""
    fields = vLLMHttpServer.spec_decode_fields(SimpleNamespace(spec_decode_metrics=None))

    assert fields == {"spec_num_draft_tokens": 0, "spec_num_accepted_tokens": 0, "spec_num_verify_steps": 0}

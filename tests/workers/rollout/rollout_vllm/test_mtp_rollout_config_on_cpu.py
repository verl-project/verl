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

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.workers.config.model import MtpConfig
from verl.workers.rollout.vllm_rollout.utils import VLLM_MTP_DRAFTER_WEIGHT_SYNC_ENV
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def _engine_args(mtp, engine_args):
    """The engine arguments the server makes of a rollout config's mtp and the engine's own arguments."""
    server = object.__new__(vLLMHttpServer)
    server.config = SimpleNamespace(mtp=mtp)
    server._apply_mtp_rollout_config(engine_args)
    return engine_args


@pytest.mark.parametrize(("enable", "weight_sync"), [(True, "1"), (False, "0")])
def test_mtp_rollout_drafts_whether_or_not_the_actor_carries_mtp(monkeypatch, enable, weight_sync):
    """enable_rollout alone gives the engine its MTP drafter; enable decides whether the weight sync reaches it."""
    monkeypatch.setenv(VLLM_MTP_DRAFTER_WEIGHT_SYNC_ENV, "")
    mtp = MtpConfig(enable=enable, enable_rollout=True, method="mtp", num_speculative_tokens=3)

    args = _engine_args(mtp, {"speculative_config": {"model": "/draft"}})

    assert args["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 3, "model": "/draft"}
    assert args["per_request_spec_decode_metrics"] == "summary"
    assert os.environ[VLLM_MTP_DRAFTER_WEIGHT_SYNC_ENV] == weight_sync


@pytest.mark.parametrize("mtp", [None, MtpConfig(), MtpConfig(enable=True)])
def test_no_mtp_drafter_without_enable_rollout(monkeypatch, mtp):
    monkeypatch.setenv(VLLM_MTP_DRAFTER_WEIGHT_SYNC_ENV, "")

    assert _engine_args(mtp, {}) == {}
    assert os.environ[VLLM_MTP_DRAFTER_WEIGHT_SYNC_ENV] == ""

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
"""Worker-extension mixin: config hop via env, patch gating on the vLLM spec method, delta RPCs."""

import sys
import types
from types import SimpleNamespace

import pytest

from tests.experimental.das.fakes import PySuffixTree
from verl.experimental.das import proposer_patch, worker_extension
from verl.experimental.das.contracts import DeltaBatch, TrajectoryDelta, serialize_delta_batch
from verl.experimental.das.drafter_state import DASDrafterState, register_active_state
from verl.experimental.das.worker_extension import (
    DAS_CONFIG_ENV,
    DASWorkerExtensionMixin,
    load_worker_config,
    maybe_install_proposer_patch,
)
from verl.workers.config.das import DASConfig


def _plant_fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    v1 = types.ModuleType("vllm.v1")
    spec = types.ModuleType("vllm.v1.spec_decode")
    suffix = types.ModuleType("vllm.v1.spec_decode.suffix_decoding")

    class SuffixDecodingProposer:
        def __init__(self, vllm_config):
            self.max_tree_depth = 24

        def propose(self, input_batch, sampled_token_ids, slot_mappings=None):
            return [[] for _ in sampled_token_ids]

    suffix.SuffixDecodingProposer = SuffixDecodingProposer
    for name, mod in (
        ("vllm", vllm),
        ("vllm.v1", v1),
        ("vllm.v1.spec_decode", spec),
        ("vllm.v1.spec_decode.suffix_decoding", suffix),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return SuffixDecodingProposer


@pytest.fixture
def _clean():
    yield
    register_active_state(None)
    proposer_patch._CONFIG = None
    worker_extension._warned_no_suffix = False


def test_load_worker_config_from_env():
    assert load_worker_config({}) is None
    assert load_worker_config({DAS_CONFIG_ENV: "not json"}) is None
    assert load_worker_config({DAS_CONFIG_ENV: DASConfig(enable=False).to_json()}) is None
    cfg = load_worker_config({DAS_CONFIG_ENV: DASConfig(enable=True, budget_long=5).to_json()})
    assert cfg is not None and cfg.budget_long == 5


@pytest.mark.usefixtures("_clean")
def test_patch_requires_enable_and_suffix_method(monkeypatch):
    cls = _plant_fake_vllm(monkeypatch)
    env = {DAS_CONFIG_ENV: DASConfig(enable=True).to_json()}
    vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(method="ngram"))
    assert maybe_install_proposer_patch(vllm_config, environ=env) is False
    assert maybe_install_proposer_patch(SimpleNamespace(speculative_config=None), environ=env) is False
    assert (
        maybe_install_proposer_patch(SimpleNamespace(speculative_config=SimpleNamespace(method="suffix")), {}) is False
    )
    assert not getattr(cls, "_verl_das_patched", False)

    vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(method="suffix"))
    assert maybe_install_proposer_patch(vllm_config, environ=env) is True
    assert getattr(cls, "_verl_das_patched", False)


@pytest.mark.usefixtures("_clean")
def test_mixin_new_installs_patch_and_tolerates_object_base(monkeypatch):
    cls = _plant_fake_vllm(monkeypatch)
    monkeypatch.setenv(DAS_CONFIG_ENV, DASConfig(enable=True).to_json())

    class Worker(DASWorkerExtensionMixin):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    worker = Worker(vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method="suffix")), rank=0)
    assert worker.kwargs["rank"] == 0
    assert getattr(cls, "_verl_das_patched", False)


@pytest.mark.usefixtures("_clean")
def test_rpcs_without_and_with_active_state():
    class Worker(DASWorkerExtensionMixin):
        pass

    worker = Worker()
    assert worker.das_apply_deltas(b"\x01") == -1

    state = DASDrafterState(DASConfig(), max_tree_depth=24, tree_factory=PySuffixTree)
    register_active_state(state)
    payload = serialize_delta_batch(DeltaBatch(to_version=3, iteration=1, adds=[TrajectoryDelta("p", 0, (1, 2, 3))]))
    assert worker.das_apply_deltas(payload) == 3
    assert state.tree_stats()["problem_trees"] == 1

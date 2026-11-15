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
"""The DAS propose implementation and the in-place proposer patch, driven duck-typed.

``_propose_impl`` is exercised against a fake ``suffix_cache`` standing in for
arctic's SuffixDecodingCache and a SimpleNamespace ``input_batch`` shaped like
vLLM's. The patch installer is exercised by planting a fake ``vllm`` package
in ``sys.modules`` (vLLM is not required on CPU CI).
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from tests.experimental.das.fakes import PySuffixTree
from verl.experimental.das import proposer_patch
from verl.experimental.das.contracts import DeltaBatch, TrajectoryDelta, serialize_delta_batch
from verl.experimental.das.drafter_state import (
    DASDrafterState,
    DraftResult,
    get_active_state,
    register_active_state,
)
from verl.experimental.das.problem_id import encode_request_id, hash_problem_id
from verl.experimental.das.proposer_patch import _propose_args, _propose_impl, install_proposer_patch
from verl.workers.config.das import DASConfig

MOTIF = [100, 101, 102, 103, 104, 105]
PROMPT = [5, 6, 7, 8]


class FakeSuffixCache:
    """Minimal stand-in for arctic's request-scoped SuffixDecodingCache."""

    def __init__(self, max_tree_depth=24, max_cached_requests=1000, draft=None):
        self.max_tree_depth = max_tree_depth
        self.max_cached_requests = max_cached_requests
        self.active_requests = set()
        self.cached_requests = set()
        self.responses = {}
        self.stopped = []
        self.draft = draft
        self.prompt_dtypes = {}
        self.pattern_dtypes = {}

    def start_request(self, req_id, prompt_ids):
        self.active_requests.add(req_id)
        self.prompt_dtypes[req_id] = getattr(prompt_ids, "dtype", None)
        self.responses[req_id] = [int(t) for t in prompt_ids]

    def add_active_response(self, req_id, token_ids):
        self.responses[req_id].extend(int(t) for t in token_ids)

    def evict_cached_response(self, req_id):
        self.cached_requests.discard(req_id)

    def speculate(self, req_id, pattern, max_spec_tokens, max_spec_factor, min_token_prob):
        self.pattern_dtypes[req_id] = getattr(pattern, "dtype", None)
        return self.draft

    def stop_request(self, req_id):
        self.active_requests.discard(req_id)
        self.cached_requests.add(req_id)
        self.stopped.append(req_id)


def make_state(**cfg_overrides):
    cfg_overrides.setdefault("enable", True)
    return DASDrafterState(DASConfig(**cfg_overrides), max_tree_depth=24, tree_factory=PySuffixTree)


def make_host(state, k=8, max_model_len=4096, cache=None):
    return SimpleNamespace(
        num_speculative_tokens=k,
        max_model_len=max_model_len,
        max_tree_depth=24,
        max_spec_factor=2.0,
        min_token_prob=0.1,
        suffix_cache=cache if cache is not None else FakeSuffixCache(),
        _das_state=state,
    )


def seed_problem(state, phash, cls=None):
    batch = DeltaBatch(
        to_version=1,
        iteration=1,
        adds=[TrajectoryDelta(phash, 0, tuple(MOTIF))],
        problem_cls={} if cls is None else {phash: cls},
    )
    state.apply_delta_batch(serialize_delta_batch(batch))


def make_batch(req_ids, generated=3):
    """vLLM-shaped InputBatch: int32 token_ids_cpu rows, req_id_to_index, counts."""
    n = len(req_ids)
    row = np.zeros(64, dtype=np.int32)
    row[: len(PROMPT)] = PROMPT
    row[len(PROMPT) : len(PROMPT) + generated] = MOTIF[:generated]
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_prompt_tokens=np.array([len(PROMPT)] * n),
        num_tokens_no_spec=np.array([len(PROMPT) + generated] * n),
        token_ids_cpu=np.stack([row] * n),
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
    )


def propose(host, state, batch, sampled):
    return _propose_impl(host, state, batch, sampled)


# ----------------------------------------------------------- propose impl


def test_das_request_drafts_from_problem_tree():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash)
    host = make_host(state)
    req_ids = [encode_request_id(phash), "plain-request"]
    batch = make_batch(req_ids)

    drafts = propose(host, state, batch, [[MOTIF[2]], [MOTIF[2]]])
    assert len(drafts) == 2
    assert drafts[0][:3] == MOTIF[3:6]
    # Stock cache returns nothing and there is no problem id: no draft.
    assert drafts[1] == []
    # Both requests were fed to the stock cache (stock bookkeeping unchanged).
    assert host.suffix_cache.active_requests == set(req_ids)
    assert host.suffix_cache.responses["plain-request"] == [*PROMPT, MOTIF[2]]


def test_partial_prefill_rows_skip_everything():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash)
    host = make_host(state)
    das_req = encode_request_id(phash)
    batch = make_batch([das_req, "other"])

    drafts = propose(host, state, batch, [[], [MOTIF[2]]])
    assert drafts[0] == []
    assert das_req not in host.suffix_cache.active_requests
    assert "other" in host.suffix_cache.active_requests


def test_short_class_skips_drafting_but_collects():
    state = make_state(budget_long=24, budget_medium=8, budget_short=0)
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash, cls=0)  # Short
    host = make_host(state)
    batch = make_batch([encode_request_id(phash)])

    drafts = propose(host, state, batch, [[MOTIF[2]]])
    assert drafts == [[]]
    assert state.tree_stats()["transient_requests"] == 1  # data still collected


def test_departed_requests_are_dropped_and_stock_cache_stopped():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    host = make_host(state)
    req = encode_request_id(phash)
    propose(host, state, make_batch([req]), [[MOTIF[2]]])
    assert state.tree_stats()["transient_requests"] == 1

    # Next step: the request is gone from the batch.
    other = encode_request_id(hash_problem_id([1, 2, 3]))
    propose(host, state, make_batch([other]), [[MOTIF[2]]])
    assert state.tree_stats()["transient_requests"] == 1  # only the new one
    assert host.suffix_cache.stopped == [req]
    assert host.suffix_cache.active_requests == {other}


def test_budget_capped_by_host_k_and_model_len():
    state = make_state(budget_long=24, budget_medium=8, budget_short=0)
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash, cls=2)  # Long: budget 24
    req_ids = [encode_request_id(phash)]
    batch = make_batch(req_ids)

    host = make_host(state, k=2)  # vLLM's num_speculative_tokens wins
    drafts = propose(host, state, batch, [[MOTIF[2]]])
    assert 0 < len(drafts[0]) <= 2

    # At max_model_len: no draft at all.
    host = make_host(state, max_model_len=int(batch.num_tokens_no_spec[0]))
    drafts = propose(host, state, batch, [[MOTIF[2]]])
    assert drafts == [[]]


def test_problem_draft_wins_ties_against_stock_cache():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash)
    stock = DraftResult(token_ids=[1, 2, 3], score=0.0, match_len=1)
    host = make_host(state, cache=FakeSuffixCache(draft=stock))
    batch = make_batch([encode_request_id(phash)])

    drafts = propose(host, state, batch, [[MOTIF[2]]])
    assert drafts[0][:3] == MOTIF[3:6]


def test_stock_cache_draft_used_when_it_scores_higher():
    state = make_state()
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash)
    stock = DraftResult(token_ids=[1, 2, 3], score=1e9, match_len=3)
    host = make_host(state, cache=FakeSuffixCache(draft=stock))
    batch = make_batch([encode_request_id(phash)])

    drafts = propose(host, state, batch, [[MOTIF[2]]])
    assert drafts[0] == [1, 2, 3]


def test_plain_request_uses_full_vllm_budget_and_stock_draft():
    state = make_state(budget_long=24, budget_medium=8, budget_short=0)
    stock = DraftResult(token_ids=[7, 8], score=0.5, match_len=2)
    host = make_host(state, cache=FakeSuffixCache(draft=stock))
    batch = make_batch(["plain-request"])

    drafts = propose(host, state, batch, [[MOTIF[2]]])
    assert drafts == [[7, 8]]
    assert state.tree_stats()["transient_requests"] == 0


def test_token_arrays_reach_the_stock_cache_as_int32():
    # vLLM hands out int32 token_ids_cpu; arctic >= 0.3 rejects anything else.
    # Use an int64 batch on purpose so the conversion is exercised.
    state = make_state()
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash)
    host = make_host(state)
    req = encode_request_id(phash)
    batch = make_batch([req])
    batch.token_ids_cpu = batch.token_ids_cpu.astype(np.int64)

    propose(host, state, batch, [[MOTIF[2]]])
    assert str(host.suffix_cache.prompt_dtypes[req]) == "int32"
    assert str(host.suffix_cache.pattern_dtypes[req]) == "int32"


def test_batch_row_resolved_through_req_id_to_index():
    # Stock vLLM looks the row up by req id; the sampled position may differ.
    state = make_state()
    phash = hash_problem_id(PROMPT)
    seed_problem(state, phash)
    host = make_host(state)
    das_req = encode_request_id(phash)
    batch = make_batch(["other", das_req])  # DAS content lives in row 1
    batch.token_ids_cpu[0, :] = 0  # row 0 holds nothing useful
    batch.req_ids = [das_req, "other"]  # sampled order is swapped
    batch.req_id_to_index = {das_req: 1, "other": 0}

    drafts = propose(host, state, batch, [[MOTIF[2]], [MOTIF[2]]])
    assert drafts[0][:3] == MOTIF[3:6]
    assert host.suffix_cache.responses[das_req][: len(PROMPT)] == PROMPT


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("batch", "sampled"), {}),
        (("batch", "sampled", None), {}),  # positional slot_mappings
        (("batch", "sampled"), {"slot_mappings": None}),  # keyword slot_mappings
        ((8, "batch", "sampled"), {}),  # num_speculative_tokens-prefixed builds
        ((), {"input_batch": "batch", "sampled_token_ids": "sampled"}),
        (("batch",), {"sampled_token_ids": "sampled"}),
    ],
)
def test_propose_args_forms(args, kwargs):
    assert _propose_args(args, kwargs) == ("batch", "sampled")


# ------------------------------------------------------- patch installer


def _plant_fake_vllm(monkeypatch, *, with_suffix=True):
    """Install a fake ``vllm`` tree in sys.modules; returns the proposer class."""
    vllm = types.ModuleType("vllm")
    v1 = types.ModuleType("vllm.v1")
    spec = types.ModuleType("vllm.v1.spec_decode")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.spec_decode", spec)
    if not with_suffix:
        monkeypatch.delitem(sys.modules, "vllm.v1.spec_decode.suffix_decoding", raising=False)
        return None

    class SuffixDecodingProposer:
        def __init__(self, vllm_config):
            sc = vllm_config.speculative_config
            self.num_speculative_tokens = sc.num_speculative_tokens
            self.max_tree_depth = sc.suffix_decoding_max_tree_depth
            self.max_spec_factor = sc.suffix_decoding_max_spec_factor
            self.min_token_prob = sc.suffix_decoding_min_token_prob
            self.max_model_len = vllm_config.model_config.max_model_len
            self.suffix_cache = FakeSuffixCache(max_tree_depth=self.max_tree_depth)

        def propose(self, input_batch, sampled_token_ids, slot_mappings=None):
            return [["stock"] for _ in sampled_token_ids]

    suffix = types.ModuleType("vllm.v1.spec_decode.suffix_decoding")
    suffix.SuffixDecodingProposer = SuffixDecodingProposer
    monkeypatch.setitem(sys.modules, "vllm.v1.spec_decode.suffix_decoding", suffix)
    return SuffixDecodingProposer


def _stub_vllm_config(k=8):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=k,
            suffix_decoding_max_tree_depth=24,
            suffix_decoding_max_spec_factor=2.0,
            suffix_decoding_min_token_prob=0.1,
        ),
        model_config=SimpleNamespace(max_model_len=4096),
    )


@pytest.fixture
def _clear_active_state():
    yield
    register_active_state(None)
    proposer_patch._CONFIG = None


def test_missing_suffix_proposer_is_refused(monkeypatch):
    _plant_fake_vllm(monkeypatch, with_suffix=False)
    assert install_proposer_patch(DASConfig(enable=True)) is False


@pytest.mark.usefixtures("_clear_active_state")
def test_patch_installs_once_and_drafts(monkeypatch):
    cls = _plant_fake_vllm(monkeypatch)
    assert install_proposer_patch(DASConfig(enable=True)) is True
    assert getattr(cls, "_verl_das_patched", False)
    patched = cls.propose
    assert install_proposer_patch(DASConfig(enable=True)) is True  # idempotent
    assert cls.propose is patched

    proposer = cls(_stub_vllm_config())
    assert proposer._das_state is not None
    assert get_active_state() is proposer._das_state
    assert proposer._das_state.max_tree_depth == 24  # taken from vLLM's speculative_config

    phash = hash_problem_id(PROMPT)
    proposer._das_state._tree_factory = PySuffixTree
    seed_problem(proposer._das_state, phash)
    das_req = encode_request_id(phash)
    batch = make_batch([das_req, "plain-request"])
    drafts = proposer.propose(batch, [[MOTIF[2]], [MOTIF[2]]], slot_mappings=None)
    assert drafts[0][:3] == MOTIF[3:6]
    assert drafts[1] == []
    assert proposer._das_state is not None  # DAS survived propose()


@pytest.mark.usefixtures("_clear_active_state")
def test_propose_failure_reverts_to_stock_with_original_args(monkeypatch):
    cls = _plant_fake_vllm(monkeypatch)
    install_proposer_patch(DASConfig(enable=True))
    proposer = cls(_stub_vllm_config())
    assert proposer._das_state is not None

    broken_batch = SimpleNamespace(req_ids=None)  # structural drift
    drafts = proposer.propose(broken_batch, [[1]], slot_mappings=None)
    assert drafts == [["stock"]]
    assert proposer._das_state is None
    assert proposer._das_failed is True
    assert get_active_state() is None
    # Stays on stock afterwards without re-logging.
    assert proposer.propose(make_batch(["r"]), [[1]]) == [["stock"]]


@pytest.mark.usefixtures("_clear_active_state")
def test_disabled_config_leaves_stock_propose(monkeypatch):
    cls = _plant_fake_vllm(monkeypatch)
    install_proposer_patch(DASConfig(enable=False))
    proposer = cls(_stub_vllm_config())
    assert proposer._das_state is None
    assert proposer.propose(make_batch(["r"]), [[1]]) == [["stock"]]

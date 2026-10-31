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
"""Ray-free stand-ins shared by the DAS CPU tests."""

from __future__ import annotations

from verl.experimental.das.drafter_state import DraftResult
from verl.workers.rollout.replica import TokenOutput


class Ref:
    """Awaitable holding a ready value, like a resolved Ray ObjectRef."""

    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def _ready():
            return self.value

        return _ready().__await__()


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return Ref(self._fn(*args, **kwargs))


class FakeActor:
    """Wraps a plain object so ``handle.method.remote(...)`` calls it synchronously."""

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return _RemoteMethod(getattr(self._obj, name))


class FakeServer:
    """Records engine calls and replays scripted ``TokenOutput`` results."""

    def __init__(self, outputs=None):
        self.calls: list[dict] = []
        self.outputs = list(outputs or [])

    def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids), **kwargs})
        if self.outputs:
            return self.outputs.pop(0)
        return TokenOutput(token_ids=[7, 8, 9], stop_reason="completed", extra_fields={"global_steps": 1})


def fake_ray_get(ref, timeout=None):
    """Drop-in for ``ray.get`` on :class:`Ref` values."""
    return ref.value if isinstance(ref, Ref) else ref


class PySuffixTree:
    """Pure-Python frequency-greedy suffix drafter for tests (arctic-inference is not importable on CPU CI).

    Mirrors arctic's SuffixTree semantics: longest-suffix match, greedy most-frequent continuation,
    frequency-ratio probabilities, and the ``max_spec_factor * match_len`` cap, with O(corpus) scans.
    """

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self._seqs: dict[int, list[int]] = {}

    def extend(self, seq_id: int, token_ids) -> None:
        self._seqs.setdefault(seq_id, []).extend(int(t) for t in token_ids)

    def remove(self, seq_id: int) -> None:
        self._seqs.pop(seq_id, None)

    def speculate(
        self,
        pattern,
        max_spec_tokens: int,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
    ) -> DraftResult | None:
        pattern = [int(t) for t in pattern][-self.max_depth :]
        for match_len in range(len(pattern), 0, -1):
            suffix = pattern[-match_len:]
            cursors = self._match_positions(suffix)
            if not cursors:
                continue
            cap = min(max_spec_tokens, int(match_len * max_spec_factor + max_spec_offset + 1e-6))
            draft = self._greedy_draft(cursors, cap, min_token_prob)
            if draft.token_ids:
                draft.match_len = match_len
                return draft
        return None

    def _match_positions(self, suffix: list[int]) -> list[tuple[int, int]]:
        n = len(suffix)
        out: list[tuple[int, int]] = []
        for seq_id, seq in self._seqs.items():
            out.extend((seq_id, pos + n) for pos in range(len(seq) - n + 1) if seq[pos : pos + n] == suffix)
        return out

    def _greedy_draft(self, cursors: list[tuple[int, int]], cap: int, min_token_prob: float) -> DraftResult:
        tokens: list[int] = []
        prob = 1.0
        score = 0.0
        while len(tokens) < cap:
            counts: dict[int, int] = {}
            total = 0
            for seq_id, pos in cursors:
                seq = self._seqs[seq_id]
                if pos < len(seq):
                    counts[seq[pos]] = counts.get(seq[pos], 0) + 1
                    total += 1
            if not counts:
                break
            best_tok = max(counts, key=lambda t: counts[t])
            prob *= counts[best_tok] / total
            if prob < min_token_prob:
                break
            tokens.append(best_tok)
            score += prob
            cursors = [(s, p + 1) for s, p in cursors if self._seqs[s][p : p + 1] == [best_tok]]
        return DraftResult(tokens, score, 0)

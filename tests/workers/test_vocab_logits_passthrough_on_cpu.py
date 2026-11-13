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
"""CPU tests for the resident top-k passthrough into the megatron logits processor.

Two batch flags can arm the logits processor of ``MegatronEngineWithLMHead``:

* ``distillation_use_topk`` — the batch must carry the teacher's top-k;
* ``needs_vocab_logits`` — the batch carries whatever distribution the loss prices
  against (a sampler's top-k, for score centering), or nothing when the loss entry
  takes no payload at all.

Whoever is armed, the payload is put into ``logits_processor_args`` so the forward shards it
with the same ``cp_layout`` / ``use_fp8_padding`` / ``local_cp_size`` as ``label``, and the
processor function receives it back as ``teacher_topk_ids`` / ``teacher_topk_logprobs`` (one
slot, one meaning per micro-batch). The tests below pin that contract, not the numerics:

* the teacher pair is only handed to the loss under the distillation flag;
* ``needs_vocab_logits`` arms the processor on its own, including for a loss entry that takes
  no payload at all;
* both flags together call the loss exactly once.

``vocab_parallel_log_probs_from_logits`` is patched out: it needs Megatron TP groups and cannot
run on CPU. ``_resident_logits_payload`` is the batch-side half of the same contract, so its
key spellings and fail-loud paths are covered here too.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.workers.engine.megatron.transformer_impl import (
    MegatronEngineWithLMHead,
    _resident_logits_payload,
)

_VOCAB_SIZE = 8
_ROWS = 4  # positions in this rank's shard
_TOPK = 3
_DISTILLATION_KEYS = ("distillation_losses", "student_mass", "teacher_mass")
_SCORE_CENTERING_KEYS = ("score_centering_per_position",)


def _payload(rows: int = _ROWS):
    """A ``(1, rows, topk)`` top-k pair the forward would have sharded already."""
    ids = torch.arange(rows * _TOPK, dtype=torch.int64).reshape(1, rows, _TOPK)
    logprobs = torch.full((1, rows, _TOPK), -1.0)
    return ids, logprobs


def _make_engine_stub():
    eng = object.__new__(MegatronEngineWithLMHead)
    eng.engine_config = SimpleNamespace(
        entropy_from_logits_with_chunking=False,
        entropy_from_logits_chunk_size=1024,
    )
    return eng


class _Recorder:
    """Logits processor that records how the forward called it.

    ``accept_payload=False`` gives the three-argument signature a loss entry with no payload
    has; any teacher kwarg then raises ``TypeError`` inside the call, which is the property
    under test rather than an accident of zipping kwargs.
    """

    def __init__(self, keys=_DISTILLATION_KEYS, accept_payload=True):
        self.calls = []
        self._keys = keys
        self._accept_payload = accept_payload

    def __call__(self, student_logits, data, data_format, **kwargs):
        if not self._accept_payload:
            assert not kwargs, f"loss entry takes no payload, got {sorted(kwargs)}"
        self.calls.append(kwargs)
        rows = student_logits.shape[1]
        return {key: torch.zeros(1, rows) for key in self._keys}

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _run_processor(
    eng,
    *,
    distillation_use_topk,
    needs_vocab_logits,
    distillation_only=False,
    payload=None,
    processor=None,
):
    logits = torch.randn(1, _ROWS, _VOCAB_SIZE)
    label = torch.randint(0, _VOCAB_SIZE, (1, _ROWS))
    temperature = torch.ones(1, _ROWS)
    batch = TensorDict({}, batch_size=[])
    payload_kwargs = {}
    if payload is not None:
        payload_kwargs["teacher_topk_ids"], payload_kwargs["teacher_topk_logprobs"] = payload

    with patch(
        "verl.workers.engine.megatron.transformer_impl.vocab_parallel_log_probs_from_logits",
        return_value=torch.zeros(1, _ROWS),
    ):
        return eng._lm_head_logits_processor(
            logits.clone(),
            label,
            temperature.clone(),
            calculate_sum_pi_squared=False,
            calculate_entropy=False,
            distillation_use_topk=distillation_use_topk,
            distillation_only=distillation_only,
            needs_vocab_logits=needs_vocab_logits,
            logits_processor_func=processor if processor is not None else _Recorder(),
            batch=batch,
            data_format="thd",
            **payload_kwargs,
        )


def test_teacher_payload_reaches_the_loss_only_under_the_distillation_flag():
    """The teacher pair is forwarded when (and only when) distillation armed the batch."""
    eng = _make_engine_stub()
    ids, logprobs = _payload()

    armed = _Recorder()
    _run_processor(
        eng,
        distillation_use_topk=True,
        needs_vocab_logits=False,
        payload=(ids, logprobs),
        processor=armed,
    )
    assert armed.call_count == 1
    assert armed.calls[0]["teacher_topk_ids"] is ids
    assert armed.calls[0]["teacher_topk_logprobs"] is logprobs

    # A teacher payload only exists for the distillation flag: the same batch keys are not a
    # payload for the capability flag, which looks for the sampler's keys instead.
    teacher_batch = TensorDict({"teacher_topk_token_ids": ids, "teacher_topk_logprobs": logprobs}, batch_size=[])
    assert _resident_logits_payload(teacher_batch, distillation_use_topk=True, needs_vocab_logits=False) is not None
    with pytest.raises(RuntimeError) as excinfo:
        _resident_logits_payload(teacher_batch, distillation_use_topk=False, needs_vocab_logits=True)
    assert "needs_vocab_logits" in str(excinfo.value)


def test_needs_vocab_logits_arms_the_processor_on_its_own():
    """A score-centering batch needs vocab logits without pretending to be distillation."""
    eng = _make_engine_stub()
    processor = _Recorder(keys=_SCORE_CENTERING_KEYS, accept_payload=False)

    ret = _run_processor(
        eng,
        distillation_use_topk=False,
        needs_vocab_logits=True,
        processor=processor,
    )

    assert processor.call_count == 1
    assert set(_SCORE_CENTERING_KEYS) <= set(ret)
    assert not any(key in ret for key in _DISTILLATION_KEYS)
    # ``needs_vocab_logits`` is not ``distillation_only``: the policy gradient still gets its
    # log-probs, so the only thing the new flag buys is the extra processor call.
    assert "log_probs" in ret


def test_sampler_payload_rides_the_same_slot():
    """A sampler top-k reaches the loss through the one payload slot the forward shards."""
    eng = _make_engine_stub()
    ids, logprobs = _payload()
    processor = _Recorder(keys=_SCORE_CENTERING_KEYS)

    ret = _run_processor(
        eng,
        distillation_use_topk=False,
        needs_vocab_logits=True,
        payload=(ids, logprobs),
        processor=processor,
    )

    assert processor.call_count == 1
    assert processor.calls[0]["teacher_topk_ids"] is ids
    assert processor.calls[0]["teacher_topk_logprobs"] is logprobs
    assert set(_SCORE_CENTERING_KEYS) <= set(ret)


def test_both_flags_call_the_loss_exactly_once_with_the_teacher_payload():
    """Both flags true collapses to one call: two calls would double-count per-forward tallies."""
    eng = _make_engine_stub()
    ids, logprobs = _payload()
    processor = _Recorder(keys=_SCORE_CENTERING_KEYS + _DISTILLATION_KEYS)

    _run_processor(
        eng,
        distillation_use_topk=True,
        needs_vocab_logits=True,
        payload=(ids, logprobs),
        processor=processor,
    )

    assert processor.call_count == 1, f"expected a single call, got {processor.call_count}"
    assert processor.calls[0]["teacher_topk_ids"] is ids
    assert processor.calls[0]["teacher_topk_logprobs"] is logprobs


def test_neither_flag_leaves_the_processor_disarmed():
    eng = _make_engine_stub()
    processor = _Recorder()
    ret = _run_processor(
        eng,
        distillation_use_topk=False,
        needs_vocab_logits=False,
        payload=_payload(),
        processor=processor,
    )

    assert processor.call_count == 0
    assert list(ret) == ["log_probs"]


def test_distillation_flag_keeps_the_three_argument_consumer_working():
    """Existing loss entries that take no payload are unaffected (ABI stays backward compatible)."""
    eng = _make_engine_stub()
    processor = _Recorder(accept_payload=False)

    ret = _run_processor(
        eng,
        distillation_use_topk=True,
        needs_vocab_logits=False,
        processor=processor,
    )

    assert processor.call_count == 1
    assert set(_DISTILLATION_KEYS) <= set(ret)


def test_distillation_only_still_skips_the_policy_log_probs():
    eng = _make_engine_stub()
    ret = _run_processor(
        eng,
        distillation_use_topk=True,
        needs_vocab_logits=False,
        distillation_only=True,
        processor=_Recorder(),
    )
    assert "log_probs" not in ret


@pytest.mark.parametrize(
    "batch_keys,expected",
    [
        (("teacher_topk_token_ids", "teacher_topk_logprobs"), ("teacher_topk_token_ids", "teacher_topk_logprobs")),
        (("teacher_ids", "teacher_logprobs"), ("teacher_ids", "teacher_logprobs")),
    ],
)
def test_teacher_payload_accepts_both_batch_spellings(batch_keys, expected):
    ids, logprobs = _payload()
    batch = TensorDict({batch_keys[0]: ids, batch_keys[1]: logprobs}, batch_size=[])

    payload = _resident_logits_payload(batch, distillation_use_topk=True, needs_vocab_logits=False)

    assert payload is not None
    assert payload[0] is batch[expected[0]]
    assert payload[1] is batch[expected[1]]


def test_sampler_payload_uses_its_own_batch_keys():
    ids, logprobs = _payload()
    batch = TensorDict({"sampler_topk_token_ids": ids, "sampler_topk_logprobs": logprobs}, batch_size=[])

    payload = _resident_logits_payload(batch, distillation_use_topk=False, needs_vocab_logits=True)

    assert payload is not None
    assert payload[0] is batch["sampler_topk_token_ids"]
    assert payload[1] is batch["sampler_topk_logprobs"]


def test_no_flag_means_no_payload_even_when_the_batch_carries_both():
    teacher_ids, teacher_logprobs = _payload()
    sampler_ids, sampler_logprobs = _payload()
    batch = TensorDict(
        {
            "teacher_topk_token_ids": teacher_ids,
            "teacher_topk_logprobs": teacher_logprobs,
            "sampler_topk_token_ids": sampler_ids,
            "sampler_topk_logprobs": sampler_logprobs,
        },
        batch_size=[],
    )

    assert _resident_logits_payload(batch, distillation_use_topk=False, needs_vocab_logits=False) is None

    # With a flag armed and BOTH payloads physically in the batch, the engine refuses: the
    # two are different distributions (teacher = target, sampler = the one the loss prices
    # against) and the processor receives them under the same names, so choosing one
    # silently would hand a consumer the wrong KIND under a name it trusts. The previous
    # contract let the distillation flag win here; a review pointed out that turn, and a
    # refusal is the cheaper failure to debug.
    with pytest.raises(RuntimeError, match="BOTH a teacher top-k"):
        _resident_logits_payload(batch, distillation_use_topk=True, needs_vocab_logits=True)

    # One payload, both flags: the distillation contract wins, which stays as it was.
    teacher_only = TensorDict(
        {
            "teacher_topk_token_ids": teacher_ids,
            "teacher_topk_logprobs": teacher_logprobs,
        },
        batch_size=[],
    )
    payload = _resident_logits_payload(teacher_only, distillation_use_topk=True, needs_vocab_logits=True)
    assert payload is not None
    assert payload[0] is teacher_only["teacher_topk_token_ids"]


@pytest.mark.parametrize(
    "flags,in_batch,expected_flag,expected_keys",
    [
        (
            {"distillation_use_topk": True, "needs_vocab_logits": False},
            {},
            "distillation_use_topk",
            "teacher_topk_token_ids/teacher_topk_logprobs or teacher_ids/teacher_logprobs",
        ),
        (
            {"distillation_use_topk": True, "needs_vocab_logits": False},
            {"teacher_topk_token_ids": torch.zeros(1, _ROWS, _TOPK, dtype=torch.int64)},
            "distillation_use_topk",
            "without teacher_topk_logprobs",
        ),
    ],
)
def test_an_armed_flag_without_its_payload_fails_loud(flags, in_batch, expected_flag, expected_keys):
    """Fail loud instead of silently leaving the loss without its payload."""
    batch = TensorDict(in_batch, batch_size=[])

    with pytest.raises(RuntimeError) as excinfo:
        _resident_logits_payload(batch, **flags)

    message = str(excinfo.value)
    assert expected_flag in message
    assert expected_keys in message


def test_logits_only_consumer_needs_no_incoming_payload():
    assert _resident_logits_payload({}, distillation_use_topk=False, needs_vocab_logits=True) is None


def test_logits_only_consumer_rejects_half_a_sampler_payload():
    with pytest.raises(RuntimeError, match="without sampler_topk_token_ids"):
        _resident_logits_payload(
            {"sampler_topk_logprobs": torch.zeros(1, _ROWS, _TOPK)},
            distillation_use_topk=False,
            needs_vocab_logits=True,
        )


def test_a_batch_carrying_both_payloads_is_refused():
    """One slot, two meanings: the engine will not guess which distribution it was handed.

    A teacher top-k is a TARGET distribution; a sampler top-k is the one the loss prices
    against. Both arrive under the same processor parameter names, so preferring one
    silently would hand a consumer the wrong kind under a name it trusts.
    """
    from verl.workers.engine.megatron.transformer_impl import _resident_logits_payload

    # A plain mapping is enough: the helper only reads payload keys off the batch, and this
    # suite runs without the `tensordict` package.
    batch = {
        "teacher_topk_token_ids": torch.zeros(1, 4, dtype=torch.long),
        "teacher_topk_logprobs": torch.zeros(1, 4),
        "sampler_topk_token_ids": torch.zeros(1, 4, dtype=torch.long),
        "sampler_topk_logprobs": torch.zeros(1, 4),
    }
    with pytest.raises(RuntimeError, match="BOTH a teacher top-k"):
        _resident_logits_payload(batch, distillation_use_topk=True, needs_vocab_logits=True)


def test_target_ids_is_only_passed_to_a_consumer_that_declares_it():
    """The loss callable is an external boundary: an undeclared argument is a TypeError
    after the forward has run, and a `**kwargs` consumer is not a semantic opt-in."""
    from verl.workers.engine.megatron.transformer_impl import _declares_keyword

    def strict(student_logits, data, data_format, teacher_topk_ids=None, teacher_topk_logprobs=None):
        return {}

    def permissive(**kwargs):
        return {}

    def typed(student_logits, target_ids=None):
        return {}

    assert _declares_keyword(strict, "target_ids") is False
    # A consumer that only swallows **kwargs has not opted IN: it could forward the value
    # somewhere that treats it as supervision, or drop it silently, and neither is an error.
    assert _declares_keyword(permissive, "target_ids") is False
    assert _declares_keyword(typed, "target_ids") is True


def test_the_ambiguity_check_sees_the_teacher_alias_too():
    """`teacher_ids` is a legal spelling of the teacher payload, so a batch carrying it
    next to a sampler payload is the same ambiguity under a different name."""
    from verl.workers.engine.megatron.transformer_impl import _resident_logits_payload

    batch = {
        "teacher_ids": torch.zeros(1, 4, dtype=torch.long),
        "teacher_logprobs": torch.zeros(1, 4),
        "sampler_topk_token_ids": torch.zeros(1, 4, dtype=torch.long),
        "sampler_topk_logprobs": torch.zeros(1, 4),
    }
    with pytest.raises(RuntimeError, match="BOTH a teacher top-k"):
        _resident_logits_payload(batch, distillation_use_topk=False, needs_vocab_logits=True)

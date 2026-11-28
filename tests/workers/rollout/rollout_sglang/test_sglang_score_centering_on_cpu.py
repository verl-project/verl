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

import base64
import importlib.util
import logging
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.score_centering import score_centering_auto_correction, score_centering_weight_fn
from verl.workers.rollout.sglang_rollout.utils import (
    extract_sglang_sampling_mask_with_coverage,
    extract_sglang_topk_logprobs_with_coverage,
    sglang_vocab_size,
    validate_sglang_topk_request,
)


def _wire_arrays(wire, *fields):
    result = {}
    for name, values, dtype in fields:
        values = np.asarray(values, dtype=dtype)
        if wire == "b64":
            result[f"{name}_b64"] = base64.b64encode(values.tobytes()).decode("ascii")
            result[f"{name}_b64_dtype"] = values.dtype.name
        else:
            result[name] = values.tolist()
    return result


def _token_logprobs(ids, values):
    return [[logprob, token, None] for token, logprob in zip(ids, values, strict=True)]


def _top_logprobs(wire, ids, logprobs, sampled_ids, sampled_logprobs):
    meta = {"output_token_logprobs": _token_logprobs(sampled_ids, sampled_logprobs)}
    if wire == "nested":
        meta["output_top_logprobs"] = [
            _token_logprobs(row_ids, row_logprobs) for row_ids, row_logprobs in zip(ids, logprobs, strict=True)
        ]
    else:
        meta.update(
            output_top_logprobs_shape=[len(ids), len(ids[0])],
            output_top_logprobs_null_prefix=0,
            **_wire_arrays(
                wire,
                ("output_top_logprobs_idx_flat", list(chain.from_iterable(ids)), "<i4"),
                ("output_top_logprobs_val_flat", list(chain.from_iterable(logprobs)), "<f4"),
            ),
        )
    return meta


def _sampling_mask(wire, ids, logprobs):
    meta = {"output_token_sampling_mask_length": len(ids)}
    if wire == "nested":
        meta.update(output_token_sampling_mask=ids, output_token_sampling_logprobs=logprobs)
    else:
        meta.update(
            _wire_arrays(
                wire,
                ("output_token_sampling_mask_flat", list(chain.from_iterable(ids)), "<i4"),
                ("output_token_sampling_mask_row_lengths", [len(row) for row in ids], "<i4"),
                ("output_token_sampling_logprobs_flat", list(chain.from_iterable(logprobs)), "<f4"),
            )
        )
    return meta


def _assert_rollout_arrays(ids, lps, complete, rows, k):
    assert ids.dtype == np.int32 and ids.shape == (rows, k)
    assert lps.dtype == np.float32 and lps.shape == (rows, k)
    assert complete.dtype == np.bool_ and complete.shape == (rows,)


@pytest.mark.parametrize("wire", ["nested", "flat", "b64"])
def test_untruncated_top_logprobs_preserve_sampled_token_inside_and_outside_head(wire):
    head_ids = [[3, 1], [5, 0]]
    head_lps = np.log([[0.60, 0.25], [0.58, 0.25]]).tolist()
    sampled_ids = [1, 7]
    sampled_lps = np.log([0.25, 0.08]).tolist()
    meta = _top_logprobs(wire, head_ids, head_lps, sampled_ids, sampled_lps)

    sampled, ids, lps, complete = extract_sglang_topk_logprobs_with_coverage(meta, sampled_ids, k=2, vocab_size=8)

    _assert_rollout_arrays(ids, lps, complete, rows=2, k=2)
    assert sampled == pytest.approx(sampled_lps)
    np.testing.assert_array_equal(ids, head_ids)
    np.testing.assert_allclose(lps, head_lps)
    assert complete.tolist() == [False, False]


@pytest.mark.parametrize("vocab_size, expected", [(3, True), (4, False), (None, False)])
@pytest.mark.parametrize("logprobs", [np.log([0.6, 0.3, 0.1]).tolist(), [np.log(0.6), np.log(0.4), -np.inf]])
def test_untruncated_head_coverage_requires_known_full_model_vocabulary(vocab_size, expected, logprobs):
    # This head has unit mass to float32 precision even if the vocabulary is wider.
    meta = _top_logprobs("nested", [[1, 2, 0]], [logprobs], [1], [logprobs[0]])

    _, _, lps, complete = extract_sglang_topk_logprobs_with_coverage(meta, [1], k=3, vocab_size=vocab_size)

    assert np.exp(lps).sum() == pytest.approx(1.0)
    assert complete.tolist() == [expected]


def test_vocabulary_size_uses_sampler_width_or_model_config():
    tokenizer = SimpleNamespace(vocab_size=5)
    config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=8, text_config=SimpleNamespace(vocab_size=9)))
    manager = SimpleNamespace(model_config=SimpleNamespace(vocab_size=12), tokenizer=tokenizer)

    assert sglang_vocab_size(config, manager) == 12
    assert sglang_vocab_size(config) == 9
    assert sglang_vocab_size(SimpleNamespace(tokenizer=tokenizer)) is None


@pytest.mark.parametrize(
    "wire, changes",
    [
        ("nested", {"output_token_logprobs": [[-0.5, 2, None]]}),
        ("nested", {"output_token_logprobs": [[-0.2, 1, None]]}),
        ("nested", {"output_top_logprobs": [[[-0.5, 1, None], [-1.5, 1, None]]]}),
        ("nested", {"output_top_logprobs": [[[-0.5, 1, None], [-1.5, 4, None]]]}),
        ("nested", {"output_top_logprobs": [[[-0.5, 1, None], [-1.5, 2.5, None]]]}),
        ("nested", {"output_top_logprobs": [[[-0.5, 1, None], [float("nan"), 2, None]]]}),
        ("nested", {"output_top_logprobs": [[[-0.5, 1, None], [-0.6, 2, None]]]}),
        ("nested", {"output_top_logprobs": [[[-1.5, 2, None], [-0.5, 1, None]]]}),
        ("flat", {"output_top_logprobs_shape": [2, 2]}),
        ("flat", {"output_top_logprobs_null_prefix": 1}),
        ("b64", {"output_top_logprobs_idx_flat_b64_dtype": "int64"}),
        ("b64", {"output_top_logprobs_val_flat_b64": "not base64"}),
    ],
)
def test_untruncated_top_logprobs_reject_malformed_provider_records(wire, changes):
    meta = _top_logprobs(wire, [[1, 2]], [[-0.5, -1.5]], [1], [-0.5])
    meta.update(changes)

    with pytest.raises(ValueError):
        extract_sglang_topk_logprobs_with_coverage(meta, [1], k=2, vocab_size=4)


@pytest.mark.parametrize("wire", ["nested", "flat", "b64"])
def test_sampling_mask_records_complete_support_and_ranks_partial_support(wire):
    supports = [[0, 1], [0, 6, 4], [2, 1, 7, 5]]
    probabilities = [[0.70, 0.30], [0.50, 0.40, 0.10], [0.50, 0.25, 0.15, 0.10]]
    meta = _sampling_mask(wire, supports, [np.log(row).tolist() for row in probabilities])

    sampled, ids, lps, complete = extract_sglang_sampling_mask_with_coverage(meta, [1, 6, 5], k=3, vocab_size=8)

    _assert_rollout_arrays(ids, lps, complete, rows=3, k=3)
    assert sampled == pytest.approx(np.log([0.30, 0.40, 0.10]))
    assert complete.tolist() == [True, True, False]
    for row, row_lps, expected_ids, expected_probs in zip(
        ids[:2], lps[:2], supports[:2], probabilities[:2], strict=True
    ):
        width = len(expected_ids)
        np.testing.assert_array_equal(row[:width], expected_ids)
        np.testing.assert_allclose(row_lps[:width], np.log(expected_probs))
        assert len(set(row)) == len(row)
        assert set(row[width:]) <= set(range(3)) - set(expected_ids)
        assert np.isneginf(row_lps[width:]).all()
    np.testing.assert_array_equal(ids[2], [2, 1, 7])
    np.testing.assert_allclose(lps[2], np.log([0.50, 0.25, 0.15]))


def test_sampling_top_k_one_keeps_a_certain_token_and_distinct_padding():
    assert validate_sglang_topk_request({"temperature": 0.8, "top_k": 1}, 3) == "sampling_mask"
    meta = _sampling_mask("nested", [[0]], [[0.0]])

    sampled, ids, lps, complete = extract_sglang_sampling_mask_with_coverage(meta, [0], k=3, vocab_size=5)

    assert sampled == [0.0]
    assert set(ids[0]) == {0, 1, 2}
    assert dict(zip(ids[0], lps[0], strict=True)) == {0: 0.0, 1: -np.inf, 2: -np.inf}
    assert complete.tolist() == [True]


@pytest.mark.parametrize(
    "wire, changes",
    [
        ("nested", {"output_token_sampling_mask": [[1, 1]]}),
        ("nested", {"output_token_sampling_mask": [[2, 3]]}),
        ("nested", {"output_token_sampling_mask": [[1, 4]]}),
        ("nested", {"output_token_sampling_logprobs": [0.0]}),
        ("nested", {"output_token_sampling_logprobs": [[0.0, -float("inf")]]}),
        ("nested", {"output_token_sampling_logprobs": [[-1.0, -2.0]]}),
        ("nested", {"output_token_sampling_logprobs": [np.log([0.4, 0.6]).tolist()]}),
        ("flat", {"output_token_sampling_mask_row_lengths": [3]}),
        ("b64", {"output_token_sampling_logprobs_flat_b64_dtype": "float64"}),
    ],
)
def test_sampling_mask_rejects_malformed_provider_records(wire, changes):
    meta = _sampling_mask(wire, [[1, 2]], [np.log([0.6, 0.4]).tolist()])
    meta.update(changes)

    with pytest.raises(ValueError):
        extract_sglang_sampling_mask_with_coverage(meta, [1], k=2, vocab_size=4)


@pytest.mark.parametrize("support", [[2, 0], [2, 0, 4, 1, 3]], ids=["complete", "truncated"])
def test_sampling_mask_score_centering_correction_gradient_vanishes_on_policy(support):
    logits = torch.tensor([[0.9, 0.1, 1.3, -0.4, 0.6]], requires_grad=True)
    k = 3
    # The complete support branch restricts the trainer to the support; the
    # partial branch uses full-vocabulary trainer normalization and a tail proxy.
    q_support = logits.detach()[0, support].log_softmax(-1)
    assert validate_sglang_topk_request({"top_k": len(support), "temperature": 1.0}, k) == "sampling_mask"
    meta = _sampling_mask("b64", [support], [q_support.tolist()])
    sampled, ids, lps, complete = extract_sglang_sampling_mask_with_coverage(
        meta, [support[-1]], k=k, vocab_size=logits.size(-1)
    )

    trainer_lps = logits.log_softmax(-1)
    correction, shift, q_mass, _ = score_centering_auto_correction(
        trainer_lps.gather(-1, torch.from_numpy(ids).long()),
        torch.from_numpy(lps),
        score_centering_weight_fn("token", 2.0),
        torch.from_numpy(complete),
    )
    grad = torch.autograd.grad(correction.sum(), logits)[0]

    assert complete.tolist() == [len(support) <= k]
    assert sampled == pytest.approx([q_support[-1].item()])
    torch.testing.assert_close(trainer_lps[0, support[-1]] - shift[0], q_support[-1])
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-6, rtol=0)
    if len(support) <= k:
        torch.testing.assert_close(shift, trainer_lps[:, support].logsumexp(-1))
        torch.testing.assert_close(q_mass, torch.ones_like(q_mass))
    else:
        torch.testing.assert_close(shift, torch.zeros_like(shift))
        torch.testing.assert_close(q_mass[0], q_support.topk(k).values.exp().sum())


@pytest.mark.parametrize(
    "params, options, match",
    [
        ({"top_k": -1, "top_p": 0.9}, {}, "top_p"),
        ({"top_k": -1, "min_p": 0.1}, {}, "min_p"),
        ({"temperature": 0.0}, {}, "temperature"),
        ({"temperature": 1e-7}, {}, "temperature"),
        ({"top_k": 1, "temperature": 0.0}, {}, "temperature"),
        ({}, {"prompt_logprobs": 0}, "prompt_logprobs"),
        ({}, {"return_original_logprob": True}, "SGLANG_RETURN_ORIGINAL_LOGPROB"),
    ],
)
def test_score_centering_rejects_unsupported_requests(params, options, match):
    with pytest.raises(ValueError, match=match):
        validate_sglang_topk_request(params, 3, **options)


@dataclass
class _LegacyGenerateRequest:
    rid: str
    input_ids: list[int]
    sampling_params: dict
    return_logprob: bool
    image_data: object = None
    video_data: object = None
    logprob_start_len: int | None = None
    top_logprobs_num: int | None = None


@dataclass
class _GenerateRequest(_LegacyGenerateRequest):
    return_flat_raw_output_top_logprobs: bool = False
    return_flat_raw_top_logprobs_b64: bool = False
    return_sampling_mask: bool = False
    sampling_logprobs_mode: str | None = None
    return_flat_raw_sampling_mask: bool = False
    return_flat_raw_sampling_mask_b64: bool = False


@pytest.fixture
def sglang_server_module(monkeypatch):
    # The async server and rollout extractors are real. Stub GPU-engine imports
    # and load under an isolated name so other tests can still import SGLang.
    module_symbols = {
        "sglang": {"__version__": "0.5.20"},
        "sglang.srt": {},
        "sglang.srt.entrypoints": {},
        "sglang.srt.entrypoints.engine": {},
        "sglang.srt.entrypoints.http_server": dict.fromkeys(
            ("Engine", "ServerArgs", "_GlobalState", "app", "set_global_state"), object()
        ),
        "sglang.srt.managers.io_struct": {
            **dict.fromkeys(
                (
                    "ContinueGenerationReqInput",
                    "PauseGenerationReqInput",
                    "ReleaseMemoryOccupationReqInput",
                    "ResumeMemoryOccupationReqInput",
                ),
                object(),
            ),
            "GenerateReqInput": _GenerateRequest,
        },
        "sglang.srt.managers.tokenizer_manager": {"ServerStatus": object()},
        "verl.workers.rollout.sglang_rollout.sglang_rollout": {"_set_envs_and_config": object()},
    }
    for name, symbols in module_symbols.items():
        module = ModuleType(name)
        module.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["sglang"].srt = sys.modules["sglang.srt"]
    sys.modules["sglang.srt"].entrypoints = sys.modules["sglang.srt.entrypoints"]
    sys.modules["sglang.srt.entrypoints"].engine = sys.modules["sglang.srt.entrypoints.engine"]

    source = Path(__file__).resolve().parents[4] / "verl/workers/rollout/sglang_rollout/async_sglang_server.py"
    spec = importlib.util.spec_from_file_location("_sglang_score_centering_server_for_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_name=lambda: "test"))
    monkeypatch.delenv("SGLANG_RETURN_ORIGINAL_LOGPROB", raising=False)
    return module


def _server_with_response(server_module, output_ids, meta_info, *, k, finish="stop"):
    requests = []

    async def generate_request(request, _):
        requests.append(request)
        yield {"output_ids": output_ids, "meta_info": {"finish_reason": {"type": finish}, **meta_info}}

    server = object.__new__(server_module.SGLangHttpServer)
    server._disaggregation_role = None
    server._default_stop_token_ids = ()
    server._topk_budget_warnings = set()
    server._topk_coverage_warnings = set()
    server.config = server_module.RolloutConfig(
        name="sglang", max_model_len=64, topk_log_probs=k, calculate_log_probs=True
    )
    server.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=8), lora={})
    server.tokenizer_manager = SimpleNamespace(generate_request=generate_request)
    server.global_steps = 9
    return server, requests


@pytest.mark.asyncio
async def test_generate_untruncated_head_request_and_response(sglang_server_module):
    meta = _top_logprobs("b64", [[3, 1]], [[-0.5, -2.0]], [7], [-3.0])
    server, requests = _server_with_response(sglang_server_module, [7], meta, k=2)

    params = {"max_tokens": 1, "logprobs": True, "top_k": -1, "topk_log_probs": 2}
    out = await server.generate([0], params, "path-1")

    (request,) = requests
    assert request.return_logprob is True and request.top_logprobs_num == 2
    assert request.return_flat_raw_output_top_logprobs and request.return_flat_raw_top_logprobs_b64
    assert request.return_sampling_mask is False
    assert "logprobs" not in request.sampling_params and "topk_log_probs" not in request.sampling_params
    assert out.token_ids == [7] and out.log_probs == [-3.0] and out.stop_reason == "stop"
    assert out.extra_fields["global_steps"] == 9
    np.testing.assert_array_equal(out.extra_fields["response_topk_ids"], [[3, 1]])
    np.testing.assert_allclose(out.extra_fields["response_topk_log_probs"], [[-0.5, -2.0]])
    assert out.extra_fields["response_support_complete"].tolist() == [False]


@pytest.mark.asyncio
async def test_generate_sampling_support_request_and_response(sglang_server_module, monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    meta = _sampling_mask("b64", [[1, 6]], [np.log([0.7, 0.3]).tolist()])
    meta["output_token_logprobs"] = [[-3.0, 6, None]]
    server, requests = _server_with_response(sglang_server_module, [6], meta, k=2)

    params = {"max_tokens": 1, "logprobs": True, "top_k": 2, "top_p": 0.9, "min_p": 0.1, "topk_log_probs": 2}
    out = await server.generate([0], params, "path-2")

    (request,) = requests
    assert request.return_sampling_mask is True and request.sampling_logprobs_mode == "support"
    assert request.return_flat_raw_sampling_mask and request.return_flat_raw_sampling_mask_b64
    assert request.return_logprob is False
    assert request.top_logprobs_num in (0, None) and not request.return_flat_raw_output_top_logprobs
    assert request.sampling_params["top_k"] == 2 and request.sampling_params["top_p"] == 0.9
    assert request.sampling_params["min_p"] == 0.1
    assert out.token_ids == [6] and out.log_probs == pytest.approx([np.log(0.3)])
    np.testing.assert_array_equal(out.extra_fields["response_topk_ids"], [[1, 6]])
    np.testing.assert_allclose(out.extra_fields["response_topk_log_probs"], np.log([[0.7, 0.3]]))
    assert out.extra_fields["response_support_complete"].tolist() == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs", [False, True])
async def test_generate_plain_logprobs_uses_legacy_request_api(sglang_server_module, monkeypatch, logprobs):
    monkeypatch.setattr(sglang_server_module, "GenerateReqInput", _LegacyGenerateRequest)
    meta = {"output_token_logprobs": [[-0.4, 7, None]]} if logprobs else {}
    server, requests = _server_with_response(sglang_server_module, [7], meta, k=2)

    out = await server.generate([0], {"max_tokens": 1, "logprobs": logprobs, "temperature": 0.0}, "plain")

    (request,) = requests
    assert type(request) is _LegacyGenerateRequest
    assert request.return_logprob is logprobs and request.top_logprobs_num is None
    assert out.token_ids == [7]
    assert out.log_probs == ([-0.4] if logprobs else None)
    assert out.extra_fields == {"global_steps": 9}


@pytest.mark.asyncio
async def test_generate_reports_limited_support_once_per_profile(sglang_server_module, caplog):
    meta = _sampling_mask("b64", [[1, 6]], [np.log([0.7, 0.3]).tolist()])
    server, requests = _server_with_response(sglang_server_module, [6], meta, k=1)
    params = {"max_tokens": 1, "logprobs": True, "top_k": 2, "topk_log_probs": 1}

    with caplog.at_level(logging.INFO):
        for request_id in ("first", "second"):
            output = await server.generate([0], dict(params), request_id)
            assert output.log_probs == pytest.approx([np.log(0.3)])
            assert output.extra_fields["response_support_complete"].tolist() == [False]

    assert len(requests) == 2
    assert sum("smaller than the effective sampling top_k" in message for message in caplog.messages) == 1
    assert sum("does not prove complete sampling support" in message for message in caplog.messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [-1, 0])
async def test_generate_untruncated_legacy_api_uses_nested_top_logprobs(sglang_server_module, monkeypatch, top_k):
    monkeypatch.setattr(sglang_server_module, "GenerateReqInput", _LegacyGenerateRequest)
    meta = _top_logprobs("nested", [[3, 1]], [[-0.5, -2.0]], [1], [-2.0])
    server, requests = _server_with_response(sglang_server_module, [1], meta, k=2)

    params = {"max_tokens": 1, "logprobs": True, "top_k": top_k, "topk_log_probs": 2}
    out = await server.generate([0], params, "legacy-untruncated")

    (request,) = requests
    assert request.top_logprobs_num == 2 and request.return_logprob
    assert request.sampling_params["top_k"] == -1
    assert out.log_probs == [-2.0]
    np.testing.assert_array_equal(out.extra_fields["response_topk_ids"], [[3, 1]])


@pytest.mark.asyncio
async def test_generate_rejects_sampling_mask_on_legacy_request_api(sglang_server_module, monkeypatch):
    monkeypatch.setattr(sglang_server_module, "GenerateReqInput", _LegacyGenerateRequest)
    server, requests = _server_with_response(sglang_server_module, [], {}, k=2)

    with pytest.raises(RuntimeError, match="return_sampling_mask"):
        params = {"max_tokens": 1, "logprobs": True, "top_k": 2, "topk_log_probs": 2}
        await server.generate([0], params, "legacy-truncated")

    assert not requests


@pytest.mark.asyncio
async def test_generate_rejects_original_sglang_logprobs_for_untruncated_head(sglang_server_module, monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    server, requests = _server_with_response(sglang_server_module, [], {}, k=2)

    with pytest.raises(ValueError, match="SGLANG_RETURN_ORIGINAL_LOGPROB"):
        params = {"max_tokens": 1, "logprobs": True, "top_k": -1, "topk_log_probs": 2}
        await server.generate([0], params, "original")

    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"top_p": 0.9}, "top_p"),
        ({"min_p": 0.1}, "min_p"),
        ({"temperature": 0.0}, "temperature"),
        ({"prompt_logprobs": 0}, "prompt_logprobs"),
    ],
)
async def test_generate_rejects_unsupported_ranked_requests_before_dispatch(sglang_server_module, overrides, match):
    server, requests = _server_with_response(sglang_server_module, [], {}, k=2)
    params = {"max_tokens": 1, "logprobs": True, "top_k": -1, "topk_log_probs": 2, **overrides}

    with pytest.raises(ValueError, match=match):
        await server.generate([0], params, "unsupported")

    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [-1, 2])
async def test_generate_zero_token_abort_preserves_empty_score_centering_arrays(sglang_server_module, top_k):
    server, requests = _server_with_response(sglang_server_module, [], {}, k=2, finish="abort")

    params = {"max_tokens": 1, "logprobs": True, "top_k": top_k, "topk_log_probs": 2}
    out = await server.generate([0], params, "abort")

    assert len(requests) == 1 and out.stop_reason == "abort"
    assert out.token_ids == [] and out.log_probs == []
    _assert_rollout_arrays(
        out.extra_fields["response_topk_ids"],
        out.extra_fields["response_topk_log_probs"],
        out.extra_fields["response_support_complete"],
        rows=0,
        k=2,
    )


def test_extract_top_logprobs_clamped_mask_entry_is_not_certified_complete():
    meta = _top_logprobs("nested", [[0, 1]], [[0.0, float(np.finfo(np.float32).min)]], [0], [0.0])
    sampled, ids, logprobs, complete = extract_sglang_topk_logprobs_with_coverage(meta, [0], k=2, vocab_size=2)
    assert sampled == [0.0]
    np.testing.assert_array_equal(ids, [[0, 1]])
    assert complete.tolist() == [False]
    assert np.isfinite(logprobs[0, 0])


def test_extract_top_logprobs_accepts_bf16_rl_on_policy_mass_tolerance():
    bf16_half = float(np.float32(-0.69140625))
    meta = _top_logprobs("nested", [[0, 1]], [[bf16_half, bf16_half]], [0], [bf16_half])
    with pytest.raises(ValueError, match="mass"):
        extract_sglang_topk_logprobs_with_coverage(meta, [0], k=2, vocab_size=2)
    sampled, _, _, complete = extract_sglang_topk_logprobs_with_coverage(meta, [0], k=2, vocab_size=2, mass_atol=1e-2)
    assert sampled == pytest.approx([bf16_half])
    assert complete.tolist() == [True]


@pytest.mark.asyncio
async def test_generate_untruncated_pins_defaults_and_matches_sglang_env_truthiness(sglang_server_module, monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "on")
    meta = _top_logprobs("nested", [[3, 1]], [[-0.5, -2.0]], [1], [-2.0])
    server, requests = _server_with_response(sglang_server_module, [1], meta, k=2)
    out = await server.generate([0], {"max_tokens": 1, "logprobs": True, "topk_log_probs": 2}, "pin-defaults")
    (request,) = requests
    assert request.sampling_params["top_k"] == -1
    assert request.sampling_params["top_p"] == 1.0
    assert request.sampling_params["min_p"] == 0.0
    assert out.log_probs == [-2.0]


@pytest.mark.asyncio
async def test_generate_nonempty_abort_missing_mask_drops_unverified_tokens(sglang_server_module):
    server, requests = _server_with_response(sglang_server_module, [4, 5], {}, k=2, finish="abort")
    out = await server.generate(
        [0], {"max_tokens": 2, "logprobs": True, "top_k": 2, "topk_log_probs": 2}, "nonempty-abort"
    )
    assert len(requests) == 1 and out.stop_reason == "abort"
    assert out.token_ids == [] and out.log_probs == []
    _assert_rollout_arrays(
        out.extra_fields["response_topk_ids"],
        out.extra_fields["response_topk_log_probs"],
        out.extra_fields["response_support_complete"],
        rows=0,
        k=2,
    )


@pytest.mark.asyncio
async def test_generate_rejects_untruncated_with_watermarking_enabled(sglang_server_module):
    server, requests = _server_with_response(sglang_server_module, [], {}, k=2)
    server.tokenizer_manager.server_args = SimpleNamespace(enable_watermarking=True)
    with pytest.raises(ValueError, match="watermarking"):
        await server.generate([0], {"max_tokens": 1, "logprobs": True, "top_k": -1, "topk_log_probs": 2}, "watermark")
    assert not requests


def test_rollout_config_rejects_tiny_temperature_and_invalid_top_p_for_sglang():
    from verl.workers.config.rollout import RolloutConfig

    with pytest.raises(ValueError, match="temperature"):
        RolloutConfig(name="sglang", topk_log_probs=2, calculate_log_probs=True, temperature=1e-7)
    with pytest.raises(ValueError, match="top_p"):
        RolloutConfig(name="sglang", topk_log_probs=2, calculate_log_probs=True, top_k=4, top_p=1.1)

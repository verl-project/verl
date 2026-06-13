"""Service-backed reward function for QA Gym long-context QA rows."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import requests

from .logging_utils import get_reward_logger, log_reward_info, log_reward_warning

logger = get_reward_logger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = "tomaarsen/Qwen3-Reranker-8B-seq-cls"
SCORE_URL = os.environ.get("QA_GYM_RERANKER_URL", "https://api.swissai.svc.cscs.ch/v1/score")
THRESHOLD = 0.5 # conservative threshold (eval script gave 0.37 as optimal)
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("QA_GYM_RERANKER_TIMEOUT_SECONDS", "5"))

REF_CAN_PREFIX = (
    "<|im_start|>system\n\n"
    "Should <Response_B> truthful answering <Question> base on <Response_A> "
    'truthful answer <Question>\n\n"yes"or"no".\n\n<|im_end|>\n'
    "<|im_start|>user\n\n"
)
REF_CAN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think><answer>"'

# Global variables for tracking reranker throughput
_THROUGHPUT_LOCK = threading.Lock()
_RERANKER_STARTED_AT: float | None = None
_RERANKER_REQUEST_COUNT = 0
_RERANKER_SUCCESS_COUNT = 0


def _record_reranker_throughput(started_at: float, elapsed_seconds: float, success: bool) -> None:
    global _RERANKER_REQUEST_COUNT, _RERANKER_STARTED_AT, _RERANKER_SUCCESS_COUNT

    with _THROUGHPUT_LOCK:
        if _RERANKER_STARTED_AT is None or started_at < _RERANKER_STARTED_AT:
            _RERANKER_STARTED_AT = started_at
        _RERANKER_REQUEST_COUNT += 1
        _RERANKER_SUCCESS_COUNT += int(success)
        total_elapsed_seconds = max(time.perf_counter() - _RERANKER_STARTED_AT, 1e-9)
        throughput = _RERANKER_REQUEST_COUNT / total_elapsed_seconds
        completed = _RERANKER_REQUEST_COUNT
        successes = _RERANKER_SUCCESS_COUNT

    log_reward_info(
        logger,
        "qa_gym",
        f"completed={completed} successes={successes} "
        f"elapsed_s={total_elapsed_seconds:.2f} "
        f"throughput={throughput:.2f}/s last_latency_s={elapsed_seconds:.2f}",
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Score a rollout answer against the QA Gym answer span using Qwen3 reranker."""
    extra_info = extra_info or {}
    question = str(extra_info.get("question") or "")
    reference = str(ground_truth or "").strip()
    candidate = str(solution_str or "").strip()

    if not question or not reference or not candidate:
        log_reward_warning(
            logger,
            "qa_gym",
            "missing question, reference, or candidate; returning 0 reward",
            data_source=data_source,
        )
        return 0.0

    try:
        score_ab = score_ref_can_direction(question, reference, candidate)
        score_ba = score_ref_can_direction(question, candidate, reference)
    except Exception as exc:
        log_reward_warning(
            logger,
            "qa_gym",
            "Qwen3 reranker request failed; returning 0 reward",
            data_source=data_source,
            exc=exc,
        )
        return 0.0

    # Use minimum instead of average to be more conservative
    ref_can_score = min(score_ab, score_ba)
    reward = 1.0 if ref_can_score > THRESHOLD else 0.0
    return reward


def score_ref_can_direction(question: str, response_a: str, response_b: str) -> float:
    text_1, text_2 = build_ref_can_score_request_text(question, response_a, response_b)
    started_at = time.perf_counter()
    success = False
    try:
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("CSCS_SERVING_API")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.post(
            SCORE_URL,
            headers=headers,
            json={
                "model": MODEL_NAME,
                "text_1": text_1,
                "text_2": [text_2],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"score response missing data: {payload}")
        success = True
        return float(data[0]["score"])
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        _record_reranker_throughput(started_at, elapsed_seconds, success)


def build_ref_can_score_request_text(
    question: str, response_a: str, response_b: str
) -> tuple[str, str]:
    query_prefix = (
        f"<Question>: {question}\n"
        f"<Response_A>: {response_a}\n"
        f"<Question>: {question}\n"
        "<Response_B>: "
    )
    return REF_CAN_PREFIX + query_prefix, response_b + REF_CAN_SUFFIX

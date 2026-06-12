"""
Reward client for the Kubernetes-hosted code execution harness.

The client builds a local test harness and submits it to the Kubernetes service's
`/run_code` endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

import requests

from .harness import HarnessFactory
from .logging_utils import get_reward_logger, log_reward_error, log_reward_info, log_reward_warning

logger = get_reward_logger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = float(os.environ.get("DEFAULT_TIMEOUT", "25"))
DEFAULT_MEMORY_LIMIT_MB = int(os.environ.get("DEFAULT_MEMORY_LIMIT_MB", "1024"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("KUBERNETES_SANDBOX_REQUEST_TIMEOUT", "30"))

_FENCE_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<code>[\s\S]*?)```", re.MULTILINE)
_THROUGHPUT_LOCK = threading.Lock()
_VERIFICATION_STARTED_AT: float | None = None
_VERIFICATION_COUNT = 0
_VERIFICATION_SUCCESS_COUNT = 0


def _record_verification_throughput(started_at: float, elapsed_seconds: float, success: bool) -> None:
    global _VERIFICATION_COUNT, _VERIFICATION_STARTED_AT, _VERIFICATION_SUCCESS_COUNT

    with _THROUGHPUT_LOCK:
        if _VERIFICATION_STARTED_AT is None or started_at < _VERIFICATION_STARTED_AT:
            _VERIFICATION_STARTED_AT = started_at
        _VERIFICATION_COUNT += 1
        _VERIFICATION_SUCCESS_COUNT += int(success)
        total_elapsed_seconds = max(time.perf_counter() - _VERIFICATION_STARTED_AT, 1e-9)
        throughput = _VERIFICATION_COUNT / total_elapsed_seconds
        completed = _VERIFICATION_COUNT
        successes = _VERIFICATION_SUCCESS_COUNT

    log_reward_info(
        logger,
        "kubernetes_sandbox",
        f"completed={completed} successes={successes} "
        f"elapsed_s={total_elapsed_seconds:.2f} "
        f"throughput={throughput:.2f}/s last_latency_s={elapsed_seconds:.2f}",
    )


def _normalize_language(language: Any) -> str:
    value = str(language or "python").strip().lower()
    if value in {"python3", "py3", "python_3", "py"}:
        return "python"
    return value or "python"


def extract_code_for_language(text: str, language: str = "python") -> str:
    """Return the last fenced block for language, or the raw text if none exists."""
    wanted = _normalize_language(language)
    last_code = text or ""

    for match in _FENCE_RE.finditer(text or ""):
        info = (match.group("lang") or "").strip()
        fence_lang = info.split()[0].lower() if info else "text"
        if _normalize_language(fence_lang) == wanted:
            last_code = (match.group("code") or "").replace("\r\n", "\n").rstrip("\n")

    return last_code


def _json_loads_maybe(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        if not value.strip():
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return []

    decoder = json.JSONDecoder()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates = []
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            candidates.append(obj)
        except json.JSONDecodeError:
            continue

    if not candidates:
        raise ValueError("No valid JSON found in sandbox response")
    return candidates[-1]


def _parse_run_code_response(response: requests.Response) -> dict[str, Any]:
    result = _safe_json_loads(response.text)
    if not isinstance(result, dict):
        return {}

    stdout = result.get("run_result", {}).get("stdout", "")
    test_results = _safe_json_loads(stdout)
    if not isinstance(test_results, list):
        test_results = []

    total = 0
    passed = 0
    for item in test_results:
        if isinstance(item, dict) and "success" in item:
            total += 1
            passed += int(bool(item["success"]))

    return {
        **result,
        "passed": passed,
        "total": total,
    }


def _score_from_result(result: dict[str, Any], continuous: bool = True) -> float:
    if not isinstance(result, dict):
        return 0.0
    try:
        passed = int(result.get("passed") or 0)
        total = int(result.get("total") or 0)
    except (TypeError, ValueError):
        return 0.0
    if total <= 0:
        return 0.0
    if continuous:
        return passed / total
    return 1.0 if passed == total else 0.0


def _sandbox_url(explicit_url: str | None = None) -> str | None:
    url = (
        explicit_url
        or os.environ.get("KUBERNETES_SANDBOX_URL")
        or os.environ.get("SCHEDULER_URL")
    )
    if not url:
        return None
    return url.rstrip("/")


def prepare_harness(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    memory_limit_mb: int | None = None,
    timeout: float | None = None,
) -> tuple[str, str]:
    """Build the code harness submitted to the Kubernetes /run_code API."""
    extra_info = extra_info or {}
    language = _normalize_language(extra_info.get("language", "python"))

    input_output = _json_loads_maybe(extra_info.get("input_output"), default={})
    if not input_output:
        input_output = _json_loads_maybe(ground_truth, default={})
    if not isinstance(input_output, dict):
        input_output = {}

    test_cases = extra_info.get("test_list", extra_info.get("test_cases", []))
    if test_cases is None:
        test_cases = []
    test_cases = _json_loads_maybe(test_cases, default=[])
    if not isinstance(test_cases, list):
        test_cases = [test_cases]

    harness_source = HarnessFactory.generate_scripts(
        language=language,
        solution_str=extract_code_for_language(solution_str, language),
        input_output=input_output,
        test_cases=test_cases,
        data_source="test" if test_cases else "input_output",
        memory_limit_mb=int(extra_info.get("memory_limit_mb", memory_limit_mb or DEFAULT_MEMORY_LIMIT_MB)),
        timeout=float(extra_info.get("timeout", timeout or DEFAULT_TIMEOUT)),
    )
    return language, harness_source[0]


def run_code(
    sandbox_url: str,
    payload: dict[str, Any],
    concurrent_semaphore: Any = None,
    data_source: str | None = None,
) -> dict[str, Any]:
    """
    Submit one generated harness to the Kubernetes sandbox.

    Args:
        sandbox_url: Base URL of the Kubernetes sandbox service.
        payload: JSON-serializable request body accepted by the /run_code endpoint.
        concurrent_semaphore: Optional semaphore supplied by the reward manager
            to enforce reward.sandbox_fusion.max_concurrent.

    Returns:
        The parsed JSON response dictionary, or an empty dictionary on failure.
    """
    url = f"{sandbox_url.rstrip('/')}/run_code"
    started_at = time.perf_counter()
    success = False
    try:
        if concurrent_semaphore is None:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        else:
            with concurrent_semaphore:
                response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        result = _parse_run_code_response(response)
        success = isinstance(result, dict)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        log_reward_error(
            logger,
            "kubernetes_sandbox",
            f"sandbox run_code request failed; sandbox_url={sandbox_url}; returning 0 reward",
            data_source=data_source,
            exc=exc,
        )
        return {}
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        _record_verification_throughput(started_at, elapsed_seconds, success)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    sandbox_fusion_url: str | None = None,
    concurrent_semaphore: Any = None,
    memory_limit_mb: int | None = None,
    timeout: float | None = None,
    continuous: bool = True,
    **_: Any,
) -> float:
    """
    Compute a coding reward through the Kubernetes sandbox service.

    If continuous is set, the score is the fraction of test cases passed.
    Otherwise, the score is binary and requires all tests to pass.
    """
    url = _sandbox_url(sandbox_fusion_url)
    if not url:
        log_reward_warning(
            logger,
            "kubernetes_sandbox",
            "sandbox URL is not configured; returning 0 reward",
            data_source=data_source,
        )
        return 0.0

    language, harness = prepare_harness(
        data_source,
        solution_str,
        ground_truth,
        extra_info=extra_info,
        memory_limit_mb=memory_limit_mb,
        timeout=timeout,
    )
    result = run_code(
        url,
        {"language": language, "code": harness},
        concurrent_semaphore,
        data_source=data_source,
    )
    return _score_from_result(result, continuous=continuous)

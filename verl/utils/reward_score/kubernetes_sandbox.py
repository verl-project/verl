"""
Reward client for the Kubernetes-hosted code execution harness.

The service exposes the same request/response shape as the code-gym scheduler's
``/evaluate_test_cases`` endpoint, but does not require manually hosting a scheduler instance.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time

from verl.utils.reward_score.harness import HarnessFactory
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float("25") #float(os.environ.get("DEFAULT_TIMEOUT", "10"))
DEFAULT_MEMORY_LIMIT_MB = int("1024") #int(os.environ.get("DEFAULT_MEMORY_LIMIT_MB", "512"))
REQUEST_TIMEOUT_SECONDS = float("30") #float(os.environ.get("KUBERNETES_SANDBOX_REQUEST_TIMEOUT", "15"))

_FENCE_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<code>[\s\S]*?)```", re.MULTILINE)
# FIXME: remove this tmp instrumentation for measuring throughput
_THROUGHPUT_LOCK = threading.Lock()
_VERIFICATION_STARTED_AT: float | None = None
_VERIFICATION_COUNT = 0
_VERIFICATION_SUCCESS_COUNT = 0

def safe_json_loads(text: str):
    text = text.strip()
    if not text:
        return []

    decoder = json.JSONDecoder()

    # try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # find all possible JSON starts and try from the end (most robust for mixed stdout)
    candidates = []
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                candidates.append(obj)
            except Exception:
                continue

    if not candidates:
        raise ValueError(f"No valid JSON found in stdout:\n{text[:500]}")

    return candidates[-1]  # take last valid JSON block

def parse_response(response):
    """
    Parses Harness stdout JSON and returns:
    - original results
    - passed / total / failed summary
    """
    results = safe_json_loads(response.text)
    stdout = safe_json_loads(results['run_result']['stdout'])
    total = passed = 0
    for msg in stdout:
        if "success" in msg:
            total += 1
            passed += bool(msg["success"])
    return {
        **results,
        "passed": passed,
        "total": total,
    }


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

    print(
        "[KUBERNETES_SANDBOX_THROUGHPUT] "
        f"completed={completed} successes={successes} "
        f"elapsed_s={total_elapsed_seconds:.2f} "
        f"throughput={throughput:.2f}/s last_latency_s={elapsed_seconds:.2f}"
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


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    return str(value)


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


def _service_data_source(data_source: str, input_output: dict[str, Any]) -> str:
    """
    Resolve the harness selector expected by the Kubernetes sandbox service.

    Some local dataset names are aliases for service-supported harnesses. TACO
    always maps to its verified harness, and any sample with input/output tests
    uses the code_generation_lite harness unless extra_info.sandbox_data_source
    explicitly overrides it in prepare_payload.
    """
    if data_source in {"Muennighoff/mbpp", "mbpp"}:
        return "Muennighoff/mbpp"
    if data_source in {"taco", "likaixin/TACO-verified"}:
        return "likaixin/TACO-verified"
    if input_output:
        return "lighteval/code_generation_lite"
    return data_source


def prepare_harness(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    memory_limit_mb: int | None = None,
    timeout: float | None = None,
):
    """Build the Kubernetes harness request body for one coding sample."""
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

    harness = HarnessFactory.generate_scripts(
        language=language,
        solution_str=extract_code_for_language(solution_str, language),
        input_output=input_output,
        test_cases=test_cases,
        data_source="test" if len(test_cases) > 0 else "input_output",
        memory_limit_mb=int(DEFAULT_MEMORY_LIMIT_MB),
        timeout=float(DEFAULT_TIMEOUT),
    )
    return harness[0]


def run_code(
    sandbox_url: str,
    payload,
    concurrent_semaphore: Any = None,
) -> dict[str, Any]:
    """
    Submit one evaluation payload to the Kubernetes sandbox.

    Args:
        sandbox_url: Base URL of the Kubernetes sandbox service.
        payload: JSON-serializable request body accepted by the /evaluate endpoint.
        concurrent_semaphore: Optional semaphore supplied by the reward manager
            to enforce reward.sandbox_fusion.max_concurrent.

    Returns:
        The parsed JSON response dictionary, or an empty dictionary on failure.
    """
    url = f"{sandbox_url.rstrip('/')}/run_code"
    started_at = time.perf_counter()
    success = False
    try:
        # FIXME: remove
        print("[KUBERNETES_SANDBOX_DEBUG] POST", url)
        if concurrent_semaphore is None:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        else:
            with concurrent_semaphore:
                response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        result = parse_response(response)
        success = isinstance(result, dict)
        # FIXME: remove
        if isinstance(result, dict):
            print("[KUBERNETES_SANDBOX_DEBUG] result", result.get("passed"), "/", result.get("total"))
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        print("[KUBERNETES_SANDBOX_DEBUG] exception", repr(exc))
        logger.error("Kubernetes sandbox request failed: %s", exc)
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
        print("[KUBERNETES_SANDBOX_DEBUG] compute_score no_url")
        return 0.0

    harness = prepare_harness(
        data_source,
        solution_str,
        ground_truth,
        extra_info=extra_info,
        memory_limit_mb=memory_limit_mb,
        timeout=timeout,
    )
    payload = {
        "language": "python",
        "code": harness,
    }
    result = run_code(url, payload, concurrent_semaphore)
    return _score_from_result(result, continuous=continuous)

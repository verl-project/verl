"""
Deprecated reward client for the code-gym scheduler + fusion sandbox.
Evaluate generated code via the /evaluate_test_cases API.

Use verl.utils.reward_score.kubernetes_sandbox for new code reward evaluation.

Credits:
https://github.com/swiss-ai/code-gym/tree/main
https://github.com/bytedance/SandboxFusion
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any

import requests

from verl.utils.import_utils import deprecated

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.environ.get("DEFAULT_TIMEOUT", "10"))
DEFAULT_MEMORY_LIMIT_MB = int(os.environ.get("DEFAULT_MEMORY_LIMIT_MB", "2048")) # Before we had: 1024
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("CODEGYM_REQUEST_TIMEOUT", "30")) # Before we had: 15

_FENCE_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<code>[\s\S]*?)```", re.MULTILINE)


def _normalize_language(language: Any) -> str:
    value = str(language or "python").strip().lower()
    if value in {"python3", "py3", "python_3", "py"}:
        return "python"
    return value or "python"


def _fence_lang(info: str) -> str:
    info = (info or "").strip()
    return info.split()[0].lower() if info else "text"


def extract_code_for_language(text: str, language: str = "python") -> str:
    """Return the last fenced block for language, or the raw text if none exists."""
    wanted = _normalize_language(language)
    last_code = text or ""

    for match in _FENCE_RE.finditer(text or ""):
        if _normalize_language(_fence_lang(match.group("lang"))) == wanted:
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
    passed = int(result.get("passed") or 0)
    total = int(result.get("total") or 0)
    if total <= 0:
        return 0.0
    if continuous:
        return passed / total
    return 1.0 if total > 0 and passed == total else 0.0


def _scheduler_url(explicit_url: str | None = None) -> str | None:
    url = explicit_url or os.environ.get("SCHEDULER_URL")
    if not url:
        return None
    return url.rstrip("/")


def scheduler_data_source(data_source: str, input_output: dict[str, Any]) -> str:
    """Choose the code-gym harness selector"""
    if data_source in {"taco", "likaixin/TACO-verified"}:
        return "likaixin/TACO-verified"
    if input_output:
        return "lighteval/code_generation_lite"
    return data_source


def build_payload(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    memory_limit_mb: int | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """
    Build the payload for the scheduler /evaluate_test_cases API based on the data source and provided information.
    """
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

    payload = {
        "data_source": extra_info.get(
            "sandbox_data_source", scheduler_data_source(data_source, input_output)
        ),
        "language": language,
        "solution_str": extract_code_for_language(solution_str, language),
        "test_cases": test_cases,
        "input_output": input_output,
        "timeout": float(extra_info.get("timeout", timeout or DEFAULT_TIMEOUT)),
        "memory_limit_mb": int(
            extra_info.get(
                "memory_limit_mb", memory_limit_mb or DEFAULT_MEMORY_LIMIT_MB
            )
        ),
    }
    return _to_json_safe(payload)


def request_unit_tests(
    scheduler_url: str,
    payload: dict[str, Any],
    concurrent_semaphore: Any = None,
) -> dict[str, Any]:
    url = f"{scheduler_url.rstrip('/')}/evaluate_test_cases"
    try:
        if concurrent_semaphore is None:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        else:
            with concurrent_semaphore:
                response = requests.post(
                    url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
                )
        response.raise_for_status()
        result = response.json()
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        logger.error("code-gym scheduler request failed: %s", exc)
        return {}


@deprecated("verl.utils.reward_score.kubernetes_sandbox.compute_score")
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
    Compute the reward score for a given solution against the given test cases,
    using code-gym scheduler and fusion sandbox.
    If continuous is set, the score is the fraction of test cases passed,
    else the score is 1.0 only if all test cases passed, 0.0 otherwise.
    """
    url = _scheduler_url(sandbox_fusion_url)
    if not url:
        return 0.0
    payload = build_payload(
        data_source,
        solution_str,
        ground_truth,
        extra_info=extra_info,
        memory_limit_mb=memory_limit_mb,
        timeout=timeout,
    )
    result = request_unit_tests(url, payload, concurrent_semaphore)
    return _score_from_result(result, continuous=continuous)

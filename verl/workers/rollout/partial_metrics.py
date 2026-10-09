# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Observed partial-rollout events and engine prefill wall time."""

import math


def engine_prefill_timing(metrics):
    """Match vLLM's prefill interval, including preemption during prefill.

    This excludes initial queueing and is not pure GPU compute time. Missing
    timestamps (including an abort before first output) are not zero latency.
    """
    start = getattr(metrics, "scheduled_ts", None)
    end = getattr(metrics, "first_token_ts", None)
    if (
        isinstance(start, int | float)
        and isinstance(end, int | float)
        and math.isfinite(start)
        and math.isfinite(end)
        and 0 < start <= end
    ):
        return {"available": True, "seconds": end - start}
    return {"available": False, "seconds": None}


def summarize_partial_attempts(attempts):
    resumes = attempts[1:]
    observed = [a["prefill"]["seconds"] for a in resumes if a["prefill"]["available"]]
    return {
        "attempts": attempts,
        "abort_count": sum(a["aborted"] for a in attempts),
        "empty_abort_count": sum(a["aborted"] and a["new_tokens"] == 0 for a in attempts),
        "resume_count": len(resumes),
        "retained_prefix_resume_count": sum(a["retained_tokens"] > 0 for a in resumes),
        "resume_prefill_observed_count": len(observed),
        "resume_prefill_available": bool(resumes) and len(observed) == len(resumes),
        "resume_prefill_observed_seconds": sum(observed) if observed else None,
    }

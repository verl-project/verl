# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Observation-only termination records, independent of client retry control."""


def backend_termination(completion=None, *, max_tokens=None, unavailable_reason=None):
    return {
        "available": completion is not None,
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": getattr(completion, "stop_reason", None),
        "segment_max_tokens": max_tokens,
        "unavailable_reason": unavailable_reason if completion is None else None,
    }

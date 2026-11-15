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
"""Problem identity for rollout requests.

No problem or GRPO-group id reaches the server client, but GRPO siblings share
an identical first-turn prompt and the same prompt recurs every epoch, so the
prompt's token hash is a stable problem id with zero dataset changes.

The id travels to the engine inside the vLLM request id
(``dasp{16hex}-{32hex}``), which the patched proposer parses back out on every
decode step.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable

PROBLEM_HASH_LEN = 16
REQUEST_ID_PREFIX = "dasp"
_REQUEST_ID_RE = re.compile(r"^dasp([0-9a-f]{16})-")


def hash_problem_id(prompt_ids: Iterable[int]) -> str:
    """Stable 16-hex id for a problem, from its first-turn prompt token ids."""
    h = hashlib.sha256()
    for tok in prompt_ids:
        h.update(int(tok).to_bytes(8, "little", signed=True))
    return h.hexdigest()[:PROBLEM_HASH_LEN]


def encode_request_id(problem_hash: str) -> str:
    """Engine-side request id carrying the problem hash plus a fresh per-turn uuid."""
    return f"{REQUEST_ID_PREFIX}{problem_hash}-{uuid.uuid4().hex}"


def parse_request_id(request_id: object) -> str | None:
    """Extract the problem hash, or None for any non-DAS or mangled id.

    Tolerant by design: vLLM may derive child request ids by appending
    suffixes, so only the prefix is matched. A parse failure simply routes the
    request through stock suffix decoding.
    """
    if not isinstance(request_id, str):
        return None
    m = _REQUEST_ID_RE.match(request_id)
    return m.group(1) if m else None

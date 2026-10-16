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
"""Key and tag protocol shared by every TransferQueue reader and writer in trainer V1.

Keys within a partition:
- ``{uid}``: prompt group, tagged with ``is_prompt`` and a prompt status.
- ``{uid}_{session_id}``: metadata-only session marker, tagged with ``is_session`` and a session status.
- ``{uid}_{session_id}_{index}``: trajectory, tagged with per-sample metadata.

``uid`` must not contain ``_``. Status values are plain strings because they are persisted in
TransferQueue checkpoints.
"""

PROMPT_PENDING = "pending"
PROMPT_RUNNING = "running"
PROMPT_FINISHED = "finished"
PROMPT_FAILURE = "failure"
PROMPT_STATUSES = (PROMPT_PENDING, PROMPT_RUNNING, PROMPT_FINISHED, PROMPT_FAILURE)
PROMPT_INFLIGHT_STATUSES = (PROMPT_PENDING, PROMPT_RUNNING)

SESSION_RUNNING = "running"
SESSION_FINISHED = "finished"
SESSION_FAILURE = "failure"
SESSION_STATUSES = (SESSION_RUNNING, SESSION_FINISHED, SESSION_FAILURE)


def session_key(uid: str, session_id: int | str) -> str:
    return f"{uid}_{session_id}"


def trajectory_key(uid: str, session_id: int | str, index: int | str) -> str:
    return f"{uid}_{session_id}_{index}"


def uid_of(key: str) -> str:
    """Return the prompt uid owning any prompt, session, or trajectory key."""
    return key.split("_", 1)[0]


def parse_trajectory_key(key: str) -> tuple[str, int, int] | None:
    """Split a trajectory key into ``(uid, session_id, index)``; return ``None`` for other key shapes."""
    parts = key.rsplit("_", 2)
    if len(parts) != 3:
        return None
    return parts[0], int(parts[1]), int(parts[2])


def prompt_tag(status: str, global_steps: int) -> dict:
    return {"is_prompt": True, "status": status, "global_steps": global_steps}


def prompt_status_tag(status: str) -> dict:
    """Partial tag update that transitions an existing prompt key to ``status``."""
    return {"status": status}


def session_tag(status: str, session_id: int) -> dict:
    return {"is_session": True, "status": status, "session_id": session_id}


def is_prompt_tag(tag: dict) -> bool:
    return bool(tag.get("is_prompt", False))


def is_session_tag(tag: dict) -> bool:
    return bool(tag.get("is_session", False))

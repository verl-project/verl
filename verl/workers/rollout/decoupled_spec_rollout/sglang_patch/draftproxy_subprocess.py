from __future__ import annotations

import time
from typing import Any


def run_draftproxy_subprocess(
    *,
    verify_replica_rank: int,
    num_speculative_steps: int,
    draft_endpoints: list[dict[str, Any]],
    stop_event,
    ready_event,
):
    _ = verify_replica_rank, num_speculative_steps, draft_endpoints
    ready_event.set()
    while not stop_event.is_set():
        time.sleep(0.5)

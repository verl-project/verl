"""Forward RayJob NDJSON logs without spawning jq for every diagnostic line."""

import json
import sys
from pathlib import Path


def forward(events, status_path, output):
    for raw in events:
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except ValueError:
            print("[watch raw] " + raw.rstrip("\r\n"), file=output, flush=True)
            continue
        kind = event.get("type")
        if kind == "log":
            print(event.get("line") or "", file=output, flush=True)
        elif kind == "final":
            status = event.get("status") or "UNKNOWN"
            print(f"[rayjob final] {status} {event.get('message') or ''}", file=output, flush=True)
            status_path.write_text(status)
        elif kind == "error":
            print(f"[rayjob error] {event.get('message') or ''}", file=output, flush=True)
        elif kind == "status":
            print("[rayjob status] " + json.dumps(event), file=output, flush=True)
        else:
            print(json.dumps(event), file=output, flush=True)


if __name__ == "__main__":
    forward(sys.stdin, Path(sys.argv[1]), sys.stdout)

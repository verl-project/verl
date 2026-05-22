# Copyright 2026 Alibaba Group
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
"""
List and delete Daytona sandboxes held by the current API key.

A crashed Harbor run (e.g. ``ray stop --force`` mid-training, OOM, SIGKILL)
skips ``DaytonaEnvironment._stop_sandbox`` and leaves sandboxes alive on the
Daytona side. They keep consuming the account's CPU quota until
``auto_stop_interval_mins`` elapses (default 30 min in
``config/harbor_agent.yaml``), which can starve the next run and produce:

    Failed to create sandbox: Total CPU limit exceeded. Maximum allowed: N.

Recommended as a preflight before re-launching Harbor jobs:

    python verl/experimental/agentic_rl_harbor/cleanup_daytona.py             # list only
    python verl/experimental/agentic_rl_harbor/cleanup_daytona.py --delete    # delete all
    python verl/experimental/agentic_rl_harbor/cleanup_daytona.py --delete --state stopped

Authentication: reads ``DAYTONA_API_KEY`` (and optional ``DAYTONA_API_URL``,
``DAYTONA_TARGET``) from the environment, same as Harbor itself.
"""

import argparse
import os
import sys


def _iter_sandboxes(client, labels):
    """Yield every sandbox across all pages (Daytona paginates ``list()``)."""
    page = 1
    while True:
        result = client.list(labels=labels, page=page)
        items = getattr(result, "items", None) or []
        for sandbox in items:
            yield sandbox
        total_pages = getattr(result, "total_pages", 1) or 1
        if page >= total_pages or not items:
            return
        page += 1


def _parse_labels(label_args):
    if not label_args:
        return None
    out = {}
    for kv in label_args:
        if "=" not in kv:
            raise SystemExit(f"--label expects key=value, got: {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _summarize(sandbox):
    return (
        f"id={sandbox.id}  "
        f"state={getattr(sandbox, 'state', '?')}  "
        f"cpu={getattr(sandbox, 'cpu', '?')}  "
        f"created={getattr(sandbox, 'created_at', '?')}"
    )


def main():
    parser = argparse.ArgumentParser(description="List / delete Daytona sandboxes for the current API key.")
    parser.add_argument("--delete", action="store_true", help="Delete matched sandboxes (default: list only).")
    parser.add_argument(
        "--state",
        action="append",
        default=None,
        help="Only act on sandboxes in this state (e.g. started, stopped, error). Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Filter by label, format key=value. Repeatable. Forwarded to Daytona list().",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-sandbox delete timeout in seconds (default 60).",
    )
    args = parser.parse_args()

    if not os.environ.get("DAYTONA_API_KEY"):
        print("error: DAYTONA_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(2)

    try:
        from daytona import Daytona
    except ImportError as e:
        print(f"error: failed to import daytona SDK: {e}", file=sys.stderr)
        print("hint: pip install 'harbor[daytona]'", file=sys.stderr)
        sys.exit(2)

    client = Daytona()
    label_filter = _parse_labels(args.label)
    state_filter = set(args.state) if args.state else None

    sandboxes = []
    for sandbox in _iter_sandboxes(client, label_filter):
        if state_filter and getattr(sandbox, "state", None) not in state_filter:
            continue
        sandboxes.append(sandbox)

    print(f"Found {len(sandboxes)} sandbox(es) matching filters.")
    for sandbox in sandboxes:
        print(f"  {_summarize(sandbox)}")

    if not args.delete:
        print("\n(list-only; pass --delete to remove them)")
        return

    if not sandboxes:
        return

    print(f"\nDeleting {len(sandboxes)} sandbox(es)...")
    failures = 0
    for sandbox in sandboxes:
        print(f"  delete {sandbox.id} ...", end=" ", flush=True)
        try:
            client.delete(sandbox, timeout=args.timeout)
            print("ok")
        except Exception as e:
            failures += 1
            print(f"failed: {e}")
    print(f"Done. {len(sandboxes) - failures} deleted, {failures} failed.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

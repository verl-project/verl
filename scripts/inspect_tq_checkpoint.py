# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Offline inspector for a TransferQueue checkpoint written by ``tq.save_checkpoint``.

Reads the checkpoint directory directly from disk (no Ray, no live TransferQueue) so you can
verify what a resumed run will see. Its main purpose is to confirm that in-flight prompts
(``pending``/``running``) and their persisted prompt fields survived the snapshot, since those are
exactly what ``PPOTrainer._reissue_inflight_prompts`` reads back and re-submits on resume.

Layout produced by ``tq.save_checkpoint`` (naming varies slightly across TransferQueue versions):

    <ckpt>/
    ├── metadata.json          # timestamp, storage unit list, user metadata (e.g. global_steps)
    ├── controller_state.pkl   # controller: per-partition key -> tag mapping (status, global_steps)
    └── <storage_dir>/*.pkl    # per storage-unit field data (simple_storage/ or storage_units/)

Usage:
    python scripts/inspect_tq_checkpoint.py /path/to/global_step_2/transfer_queue
    python scripts/inspect_tq_checkpoint.py /path/to/global_step_2/transfer_queue --partition train
    python scripts/inspect_tq_checkpoint.py /path/to/global_step_2/transfer_queue --keys 20 --show-fields
    python scripts/inspect_tq_checkpoint.py /path/to/global_step_2/transfer_queue \
        --partition train --expect-prompt-status finished=12,running=628
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Any


class _PlaceholderUnpickler(pickle.Unpickler):
    """Unpickler that tolerates missing TransferQueue/torch classes.

    ``controller_state.pkl`` references TransferQueue (and sometimes torch) classes. When those are
    not importable here, a normal ``pickle.load`` raises before we can see anything. This unpickler
    substitutes any unresolved class with a lightweight stand-in so the surrounding plain
    dict/list/str structure (which holds the tags we care about) still loads and can be walked.
    """

    class _Placeholder:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def __repr__(self):
            return f"<Placeholder {getattr(self, '_qualname', '?')}>"

        def __setstate__(self, state):
            self._state = state

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except Exception:
            placeholder = type(
                f"Placeholder_{name}",
                (_PlaceholderUnpickler._Placeholder,),
                {"_qualname": f"{module}.{name}"},
            )
            return placeholder


def _load_pickle(path: str) -> Any:
    """Load a pickle, first normally (classes resolvable) then permissively (placeholders)."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return pickle.loads(raw)
    except Exception as normal_err:
        try:
            import io

            return _PlaceholderUnpickler(io.BytesIO(raw)).load()
        except Exception as placeholder_err:
            raise RuntimeError(
                f"Failed to unpickle {path}: normal={normal_err!r}; placeholder={placeholder_err!r}"
            ) from placeholder_err


# Only these prompt states are re-issued on resume.
_INFLIGHT_PROMPT_STATUSES = {"pending", "running"}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Fetch ``name`` from a dict, a normal object, or a placeholder-unpickled object.

    Placeholder objects (classes unresolved at unpickle time) keep their real attributes in
    ``_state`` (populated via ``__setstate__``) rather than as direct attributes, so we look there
    too. Plain dicts are indexed by key."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    state = getattr(obj, "_state", None)
    if not isinstance(state, dict):
        state = getattr(obj, "__dict__", None)
    if isinstance(state, dict):
        # A placeholder object may itself nest its real attrs under "_state".
        if name in state:
            return state[name]
        inner = state.get("_state")
        if isinstance(inner, dict):
            return inner.get(name, default)
    return default


def _extract_partition_tags(controller: Any) -> dict[str, dict[str, dict]]:
    """Return ``{partition_id: {uid: tag_dict}}`` from the controller state.

    The controller keeps per-sample tags in ``partitions[pid].custom_meta`` keyed by integer
    ``global_index`` (e.g. ``{'status': 'success', 'global_steps': 1, ...}``), and the uid<->index
    mapping in ``keys_mapping`` (uid -> index) / ``revert_keys_mapping`` (index -> uid). We join
    them so tags come back keyed by the human-readable uid. ``production_status`` (a 2-D tensor of
    per-field write flags) is intentionally ignored: the business status lives in ``custom_meta``.
    """
    partitions = _get(controller, "partitions")
    if not isinstance(partitions, dict):
        return {}

    result: dict[str, dict[str, dict]] = {}
    for pid, pdata in partitions.items():
        custom_meta = _get(pdata, "custom_meta") or {}
        revert = _get(pdata, "revert_keys_mapping") or {}
        keys_mapping = _get(pdata, "keys_mapping") or {}
        # Prefer index->uid; otherwise invert uid->index.
        idx_to_uid = dict(revert) if revert else {v: k for k, v in keys_mapping.items()}

        tags: dict[str, dict] = {}
        for gidx, meta in custom_meta.items():
            if not isinstance(meta, dict):
                continue
            uid = idx_to_uid.get(gidx, f"global_index={gidx}")
            tags[str(uid)] = meta
        result[str(pid)] = tags
    return result


def _step_from_uid(uid: str) -> int | None:
    """Best-effort submission-step from a uid of the form ``<uuid>_<step>_<attempt>``."""
    m = re.match(r".*_(\d+)_\d+$", uid)
    return int(m.group(1)) if m else None


def _parse_status_counts(value: str) -> dict[str, int]:
    """Parse ``status=count,...`` for exact prompt-status assertions."""
    result: dict[str, int] = {}
    try:
        for item in value.split(","):
            status, count = item.split("=", 1)
            status = status.strip()
            if not status or status in result:
                raise ValueError
            result[status] = int(count)
            if result[status] < 0:
                raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated status=count pairs, e.g. finished=12,running=628"
        ) from exc
    return result


def _find_storage_pickles(ckpt_dir: str) -> list[str]:
    """Locate the per storage-unit pickle files under whatever subdir this version used."""
    for sub in ("simple_storage", "storage_units", "storage"):
        d = os.path.join(ckpt_dir, sub)
        if os.path.isdir(d):
            pkls = sorted(glob.glob(os.path.join(d, "*.pkl")))
            if pkls:
                return pkls
    # Fallback: any *.pkl that is not the controller state.
    return sorted(
        p
        for p in glob.glob(os.path.join(ckpt_dir, "**", "*.pkl"), recursive=True)
        if os.path.basename(p) != "controller_state.pkl"
    )


def _summarize_fields(obj: Any, _depth: int = 0) -> Counter:
    """Best-effort count of stored field names inside a storage-unit pickle.

    Storage units keep per-key field data (the persisted prompt columns: ``uid``, ``raw_prompt``,
    dataset columns, ...). Field naming/layout varies, so we count string keys of any nested dict
    as candidate field names to confirm prompt data was actually written."""
    names: Counter = Counter()
    if _depth > 12:
        return names
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                names[k] += 1
            if isinstance(v, dict | list | tuple):
                names.update(_summarize_fields(v, _depth + 1))
    elif isinstance(obj, list | tuple):
        for v in obj:
            if isinstance(v, dict | list | tuple):
                names.update(_summarize_fields(v, _depth + 1))
    else:
        state = getattr(obj, "_state", None) or getattr(obj, "__dict__", None)
        if isinstance(state, dict):
            names.update(_summarize_fields(state, _depth + 1))
    return names


def inspect_checkpoint(
    ckpt_dir: str,
    partition: str | None,
    max_keys: int,
    show_fields: bool,
    expected_prompt_status: dict[str, int] | None = None,
) -> bool:
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"Not a directory: {ckpt_dir}")

    print(f"== TransferQueue checkpoint: {ckpt_dir} ==\n")
    expectation_matched = expected_prompt_status is None
    inspected_partition = False

    # 1. metadata.json
    meta_path = os.path.join(ckpt_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print("[metadata.json]")
        print(json.dumps(meta, indent=2, default=str))
        print()
    else:
        print("[metadata.json] MISSING\n")

    # 2. controller_state.pkl -> per-partition, per-uid tags
    ctrl_path = os.path.join(ckpt_dir, "controller_state.pkl")
    if os.path.exists(ctrl_path):
        controller = _load_pickle(ctrl_path)
        by_partition = _extract_partition_tags(controller)
        if not by_partition:
            print("[controller_state.pkl] no partitions / tags found (unexpected layout)\n")
        for pid, tags in by_partition.items():
            if partition is not None and pid != partition:
                continue
            inspected_partition = True
            print(f"[controller_state.pkl] partition '{pid}': {len(tags)} tagged samples")

            prompt_tags = {uid: tag for uid, tag in tags.items() if tag.get("is_prompt", False)}
            trajectory_tags = {uid: tag for uid, tag in tags.items() if not tag.get("is_prompt", False)}
            status_counter: Counter = Counter(tag.get("status") for tag in prompt_tags.values())
            step_counter: Counter = Counter()
            inflight_by_step: dict[Any, list[str]] = defaultdict(list)
            for uid, tag in prompt_tags.items():
                status = tag.get("status")
                # step: prefer the tag's own global_steps, fall back to the uid suffix.
                step = tag.get("global_steps", _step_from_uid(uid))
                step_counter[step] += 1
                if status in _INFLIGHT_PROMPT_STATUSES:
                    inflight_by_step[step].append(uid)

            actual_status = dict(status_counter)
            print(f"  prompt count: {len(prompt_tags)}")
            print(f"  prompt status breakdown: {actual_status}")
            print(f"  trajectory tag count: {len(trajectory_tags)}")
            print(
                "  prompt global_steps distribution: "
                f"{dict(sorted(step_counter.items(), key=lambda kv: (kv[0] is None, kv[0])))}"
            )
            n_inflight = sum(len(v) for v in inflight_by_step.values())
            print(f"  in-flight prompts (pending/running, re-issued on resume): {n_inflight}")
            for step in sorted(inflight_by_step, key=lambda s: (s is None, s)):
                uids = inflight_by_step[step]
                print(f"    global_steps={step}: {len(uids)}")
                for uid in uids[:max_keys]:
                    print(f"      - {uid} ({tags[uid].get('status')})")
                if len(uids) > max_keys:
                    print(f"      ... (+{len(uids) - max_keys} more)")

            if expected_prompt_status is not None:
                expectation_matched = actual_status == expected_prompt_status
                result = "PASS" if expectation_matched else "FAIL"
                print(
                    f"  expected prompt status: {expected_prompt_status} -> {result}"
                    + ("" if expectation_matched else f" (actual: {actual_status})")
                )

            if show_fields:
                print("  sample prompt tags (first few):")
                for uid, tag in list(prompt_tags.items())[:max_keys]:
                    print(f"    {uid}: {tag}")
            print()
    else:
        print("[controller_state.pkl] MISSING\n")

    # 3. storage unit pickles -> persisted field names
    storage_pkls = _find_storage_pickles(ckpt_dir)
    print(f"[storage] found {len(storage_pkls)} storage-unit pickle(s)")
    for p in storage_pkls:
        try:
            data = _load_pickle(p)
            field_names = _summarize_fields(data)
            top = ", ".join(f"{k}({c})" for k, c in field_names.most_common(20))
            print(f"  {os.path.relpath(p, ckpt_dir)}: candidate field names -> {top or '(none found)'}")
        except Exception as e:
            print(f"  {os.path.relpath(p, ckpt_dir)}: FAILED to load ({e})")
    print()

    print(
        "Note: extraction is best-effort and version-tolerant. Key checks: (1) in-flight (pending/running) "
        "sample count matches what you expect at save time -- 0 means every trajectory was complete, so "
        "_reissue_inflight_prompts re-submits nothing; (2) storage pickles contain prompt/response fields "
        "(content, ground_truth, ...). A global_steps distribution spanning several steps is normal: "
        "completed trajectories are retained and bounded by the off-policy staleness threshold, not deleted "
        "immediately."
    )
    if expected_prompt_status is not None and not inspected_partition:
        print(f"\nExpected-status check FAIL: partition {partition!r} was not found.")
        return False
    return expectation_matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_dir", help="Path to the transfer_queue/ checkpoint dir (inside global_step_N/)")
    parser.add_argument("--partition", default=None, help="Only inspect this partition (e.g. train)")
    parser.add_argument("--keys", type=int, default=10, dest="max_keys", help="Max uids to list per group")
    parser.add_argument("--show-fields", action="store_true", help="Print a sample of raw per-uid tags")
    parser.add_argument(
        "--expect-prompt-status",
        type=_parse_status_counts,
        default=None,
        help="Require the exact prompt status counts, e.g. finished=12,running=628 (exit 1 on mismatch)",
    )
    args = parser.parse_args()
    if args.expect_prompt_status is not None and args.partition is None:
        parser.error("--expect-prompt-status requires --partition")
    matched = inspect_checkpoint(
        args.ckpt_dir,
        args.partition,
        args.max_keys,
        args.show_fields,
        args.expect_prompt_status,
    )
    if not matched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

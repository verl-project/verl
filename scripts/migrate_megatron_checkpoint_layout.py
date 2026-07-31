# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Migrate a pre-v2 Megatron checkpoint to the v2 split layout.

Old layout (schema v1, produced by verl <= the previous release)::

    <checkpoint>/
    ├── dist_ckpt/                  # optimizer + rng + (maybe) model shards
    ├── huggingface/                # mbridge HF tree (optional)
    ├── transformer_config.json     # optional
    └── ckpt_contents.json          # optional

New layout (schema v2, produced by current verl)::

    <checkpoint>/
    ├── model/
    │   ├── dist_ckpt/              # model shards (mbridge off / PEFT)
    │   └── huggingface/            # mbridge HF tree
    ├── optimizer/dist_ckpt/        # optimizer + lr_scheduler
    ├── extra/dist_ckpt/            # rng_state
    ├── transformer_config.json
    └── ckpt_contents.json

Why the migration is cheap
--------------------------

The old ``dist_ckpt/`` tree is a single torch_dist archive that may hold
optimizer, rng, and (optionally) model/peft shards mixed together.
Megatron's ``dist_checkpointing.load`` is key-driven: it only reads
entries whose key matches the sharded state dict handed to it, and
ignores unexpected keys by default.  That means we can satisfy the new
loader's per-subtree expectations simply by *pointing* all three new
``dist_ckpt/`` targets at the same physical data.

The script therefore defaults to using hardlinks (same filesystem
only), falling back to symlinks.  ``--copy`` forces a full copy if you
need three independent physical trees (uses ~3x disk space).

Typical usage
-------------

    # In-place migration of a single checkpoint directory:
    python scripts/migrate_megatron_checkpoint_layout.py \
        --checkpoint /path/to/checkpoints/run/global_step_100/actor

    # Migrate every actor/critic subdirectory under a run:
    python scripts/migrate_megatron_checkpoint_layout.py \
        --checkpoint-root /path/to/checkpoints/run --all-steps

The script is idempotent: running it again on an already-migrated
checkpoint is a no-op.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _is_v2_layout(ckpt_dir: Path) -> bool:
    return any((ckpt_dir / sub).is_dir() for sub in ("model", "optimizer", "extra"))


def _is_legacy_layout(ckpt_dir: Path) -> bool:
    return (ckpt_dir / "dist_ckpt").is_dir() or (ckpt_dir / "huggingface").is_dir()


def _link_tree(src: Path, dst: Path, mode: str) -> None:
    """Create ``dst`` as a hardlinked / symlinked / copied mirror of ``src``.

    ``mode`` is one of ``"hardlink"``, ``"symlink"``, ``"copy"``.  Hardlink
    falls back to symlink on cross-filesystem errors; symlink creates a
    single directory symlink (cheap, but requires the symlink to stay
    valid for the lifetime of the checkpoint).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            return
        os.symlink(src.resolve(), dst)
        return
    if mode == "copy":
        if dst.exists():
            return
        shutil.copytree(src, dst)
        return
    # hardlink: replicate directory structure and hardlink each file.
    dst.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src_file = Path(root) / fname
            tgt_file = target_dir / fname
            if tgt_file.exists():
                continue
            try:
                os.link(src_file, tgt_file)
            except OSError:
                shutil.copy2(src_file, tgt_file)


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def migrate_one(ckpt_dir: Path, *, dry_run: bool, mode: str) -> str:
    """Return a short status string for reporting."""
    if _is_v2_layout(ckpt_dir):
        return "skipped (already v2)"
    if not _is_legacy_layout(ckpt_dir):
        return "skipped (no recognised layout)"

    legacy_dist = ckpt_dir / "dist_ckpt"
    legacy_hf = ckpt_dir / "huggingface"

    new_model = ckpt_dir / "model"
    new_optim = ckpt_dir / "optimizer"
    new_extra = ckpt_dir / "extra"
    new_model_dist = new_model / "dist_ckpt"
    new_model_hf = new_model / "huggingface"
    new_optim_dist = new_optim / "dist_ckpt"
    new_extra_dist = new_extra / "dist_ckpt"

    actions: list[str] = []

    if legacy_dist.is_dir():
        actions.append(f"{mode}: {legacy_dist} -> {new_optim_dist}")
        actions.append(f"{mode}: {legacy_dist} -> {new_extra_dist}")
        actions.append(f"{mode}: {legacy_dist} -> {new_model_dist}")
    if legacy_hf.is_dir():
        actions.append(f"move: {legacy_hf} -> {new_model_hf}")

    if dry_run:
        for a in actions:
            print(f"  would {a}")
        return "planned"

    if legacy_dist.is_dir():
        _link_tree(legacy_dist, new_optim_dist, mode)
        _link_tree(legacy_dist, new_extra_dist, mode)
        # Model tree only needed if there's a chance the training produced
        # model shards in the mixed dist_ckpt (dist_ckpt backend or PEFT).
        # We always create it so the loader can satisfy should_load_model;
        # unused keys are ignored by dist_checkpointing.
        _link_tree(legacy_dist, new_model_dist, mode)
    if legacy_hf.is_dir():
        new_model.mkdir(parents=True, exist_ok=True)
        _move(legacy_hf, new_model_hf)

    # Remove the original dist_ckpt/ root entry only when we materialised
    # independent copies; otherwise it's still referenced by the mirrors.
    if mode == "copy" and legacy_dist.is_dir():
        shutil.rmtree(legacy_dist)

    return "migrated"


def iter_checkpoint_dirs(root: Path) -> list[Path]:
    """Yield every leaf checkpoint directory under ``root``.

    A "leaf" is any directory that looks like a Megatron role directory —
    i.e. it contains a ``dist_ckpt`` or ``huggingface`` subdir (legacy) or
    a ``model`` subdir (v2, already migrated).
    """
    out: list[Path] = []
    for dirpath, dirnames, _files in os.walk(root):
        p = Path(dirpath)
        if _is_legacy_layout(p) or _is_v2_layout(p):
            out.append(p)
            # don't descend further into an identified checkpoint
            dirnames[:] = []
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="Migrate a single role directory (e.g. .../global_step_N/actor).")
    src.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Scan the given directory and migrate every role subdirectory found.",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="With --checkpoint-root: recurse into every global_step_*/role/ directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions and exit.")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help=(
            "How to point the new dist_ckpt subtrees at the old data. "
            "hardlink is cheapest and safe when the migration stays on one filesystem; "
            "copy duplicates the data (~3x disk use)."
        ),
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        targets = [args.checkpoint]
    else:
        if not args.all_steps:
            parser.error("--checkpoint-root requires --all-steps")
        targets = iter_checkpoint_dirs(args.checkpoint_root)

    if not targets:
        print("No checkpoint directories found.", file=sys.stderr)
        return 1

    failures = 0
    for t in targets:
        if not t.is_dir():
            print(f"[skip] {t}: not a directory")
            continue
        print(f"[{t}]")
        try:
            status = migrate_one(t, dry_run=args.dry_run, mode=args.mode)
            print(f"  {status}")
        except Exception as exc:  # pragma: no cover - migration tool
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Upload model weights to Weights & Biases as an artifact.

Usage:
  export WANDB_API_KEY=...  # or set a custom env var via --api-key-env
  python scripts/push_to_wandb.py \\
      --project my_project \\
      --run-name upload_run \\
      --artifact-name my-model \\
      --aliases latest \\
      --paths /path/to/model.pt /path/to/checkpoint_dir

Notes:
- `--paths` accepts files or directories; each is added to the artifact.
- `--artifact-type` defaults to "model".
- You can attach metadata with `--metadata path/to/meta.json`.
- If the artifact name + alias already exists, uploading with the same alias (e.g., latest) will create a new version.
- For convenience/backward-compatibility, `--artifact-name name:alias` is accepted and treated as
  `--artifact-name name --aliases alias`.

Example:
python scripts/push_to_wandb.py \
    --project multiple_choice_question_study \
    --run-name upload_qwen25_3B_gsm8k \
    --artifact-name qwen25_3B_gsm8k \
    --aliases latest \
    --paths checkpoints/multiple_choice_question_study/qwen25_3B_gsm8k/global_step_XX/huggingface \
    --description "HF merge of global_step_XX" \
    --artifact-type model
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from typing import Dict, List, Optional

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload model weights to Weights & Biases.")
    parser.add_argument("--project", required=True, help="WandB project name.")
    parser.add_argument("--run-name", required=True, help="Experiment/run name.")
    parser.add_argument(
        "--entity",
        default=None,
        help="Optional WandB entity/org. Defaults to the account tied to the API key.",
    )
    parser.add_argument(
        "--api-key-env",
        default="WANDB_API_KEY",
        help="Environment variable holding the WandB API key (default: WANDB_API_KEY).",
    )
    parser.add_argument(
        "--artifact-name",
        required=True,
        help=(
            "Name for the artifact, e.g., 'my-model'. "
            "Optionally accepts 'name:alias' for backward compatibility."
        ),
    )
    parser.add_argument(
        "--aliases",
        nargs="+",
        default=None,
        help=(
            "Optional artifact aliases (e.g. latest v1). "
            "You can also pass a comma-separated list like 'latest,v1'."
        ),
    )
    parser.add_argument(
        "--artifact-type",
        default="model",
        help='Artifact type (default: "model").',
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Files or directories to include in the artifact.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional artifact description.",
    )
    parser.add_argument(
        "--metadata",
        help="Path to JSON file with optional metadata to store on the artifact.",
    )
    return parser.parse_args()


def load_json(path: Optional[str]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_artifact_name_and_aliases(raw: str) -> tuple[str, List[str]]:
    """Return (artifact_name, aliases) from a raw CLI artifact string.

    W&B artifact names may not contain ':', so we treat 'name:alias' as a
    backward-compatible shorthand for an alias.
    """
    raw = raw.strip()
    if ":" not in raw:
        return raw, []
    name, alias_part = raw.split(":", 1)
    aliases = [a for a in alias_part.split(",") if a]
    return name, aliases


def _normalize_aliases(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for value in values:
        for part in value.split(","):
            alias = part.strip()
            if alias and alias not in out:
                out.append(alias)
    return out


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Environment variable '{args.api_key_env}' is not set with a WandB API key."
        )

    wandb.login(key=api_key)

    metadata = load_json(args.metadata)
    run = wandb.init(
        project=args.project,
        name=args.run_name,
        entity=args.entity,
    )

    artifact_name, legacy_aliases = _split_artifact_name_and_aliases(args.artifact_name)
    aliases = _normalize_aliases(args.aliases)
    for alias in legacy_aliases:
        if alias not in aliases:
            aliases.append(alias)

    artifact = wandb.Artifact(
        name=artifact_name,
        type=args.artifact_type,
        description=args.description,
        metadata=metadata,
    )

    for path in args.paths:
        if not os.path.exists(path):
            raise SystemExit(f"Path does not exist: {path}")
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)

    if aliases:
        run.log_artifact(artifact, aliases=aliases)
    else:
        run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    main()

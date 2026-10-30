#!/usr/bin/env python3
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
#
# Batch wrapper for `variantlib make-variant`; adapted from
# https://github.com/sgl-project/sgl-kernel-npu (PR #506).
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from variantlib.constants import VALIDATION_WHEEL_NAME_REGEX, VARIANT_DIST_INFO_FILENAME
from variantlib.variant_dist_info import VariantDistInfo

VARIANTLIB_CMD = "variantlib"


@dataclass
class Job:
    wheel: str
    variant_label: str | None
    properties: list[str]
    null_variant: bool
    pyproject_toml: str
    output_dir: str
    skip_plugin_validation: bool
    no_isolation: bool
    installer: str | None


def _split_labels(raw_value: str) -> set[str]:
    return {x.strip() for x in raw_value.split(",") if x.strip()}


def _load_config(config_path: Path) -> Any:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _expand_wheel_paths(wheel: str, job_index: int) -> list[str]:
    wheel_path = Path(wheel)
    if not wheel_path.is_dir():
        return [wheel]

    # Support passing a subdirectory directly (e.g. 310p/910b/a3) and expand all wheels in it.
    candidates = sorted(
        p for p in wheel_path.iterdir() if p.is_file() and p.suffix == ".whl"
    )
    if not candidates:
        raise ValueError(
            f"Job #{job_index} wheel directory has no .whl files: {wheel_path}"
        )
    return [str(path) for path in candidates]


def _resolve_output_wheel(job: Job) -> Path:
    wheel_info = VALIDATION_WHEEL_NAME_REGEX.fullmatch(Path(job.wheel).name)
    if wheel_info is None:
        raise ValueError(f"Input wheel filename is invalid: {job.wheel!r}")

    base_wheel_name = wheel_info.group("base_wheel_name")
    output_dir = Path(job.output_dir)

    if job.variant_label:
        return output_dir / f"{base_wheel_name}-{job.variant_label}.whl"

    candidates = sorted(output_dir.glob(f"{base_wheel_name}-*.whl"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No output wheel found for base name {base_wheel_name!r} in {output_dir}"
        )
    raise ValueError(
        f"Multiple output wheels found for base name {base_wheel_name!r}; "
        "please set variant_label in job to disambiguate"
    )


def _verify_built_wheel(job: Job) -> str:
    output_wheel = _resolve_output_wheel(job)
    if not output_wheel.is_file():
        raise FileNotFoundError(f"Expected output wheel not found: {output_wheel}")

    output_name = output_wheel.name
    wheel_info = VALIDATION_WHEEL_NAME_REGEX.fullmatch(output_name)
    if wheel_info is None:
        raise ValueError(f"Output filename is not a valid wheel name: {output_name!r}")

    filename_label = wheel_info.group("variant_label")
    if filename_label is None:
        raise ValueError(f"Output wheel is not a variant wheel: {output_name!r}")
    if job.variant_label is not None and filename_label != job.variant_label:
        raise ValueError(
            f"Variant label mismatch: expected={job.variant_label!r}, actual={filename_label!r}"
        )

    with zipfile.ZipFile(output_wheel, "r") as zip_file:
        for name in zip_file.namelist():
            components = name.split("/", 2)
            if (
                len(components) == 2
                and components[0].endswith(".dist-info")
                and components[1] == VARIANT_DIST_INFO_FILENAME
            ):
                variant_dist_info = VariantDistInfo(zip_file.read(name), filename_label)
                break
        else:
            raise ValueError(
                f"Invalid wheel -- no {VARIANT_DIST_INFO_FILENAME} found: {output_name!r}"
            )

    actual_properties = {x.to_str() for x in variant_dist_info.variant_desc.properties}
    expected_properties = set(job.properties)
    if actual_properties != expected_properties:
        raise ValueError(
            f"Variant properties mismatch: expected={sorted(expected_properties)}, actual={sorted(actual_properties)}"
        )

    return (
        f"[VERIFY] output={output_wheel} is a Wheel Variant - Label: {filename_label}; "
        f"properties={sorted(actual_properties)}"
    )


def load_jobs(data: Any) -> list[Job]:
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object with 'variables' and 'jobs'.")

    variables_obj = data.get("variables", {})
    if not isinstance(variables_obj, dict):
        raise ValueError("Config field 'variables' must be a JSON object.")

    config_defaults: dict[str, Any] = dict(variables_obj)
    raw_jobs = data.get("jobs", [])

    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("No jobs found in config.")

    jobs: list[Job] = []

    for i, item in enumerate(raw_jobs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Job #{i} must be a JSON object.")

        wheel = item.get("wheel", config_defaults.get("wheel"))
        wheel_env = str(
            item.get("wheel_env", config_defaults.get("wheel_env", "VARIANTLIB_WHEEL"))
        )
        if not wheel:
            wheel = os.environ.get(wheel_env)
        if not wheel:
            raise ValueError(
                f"Job #{i} is missing required field: 'wheel'. "
                f"You can also set env var {wheel_env!r}."
            )

        job_pyproject_toml = item.get(
            "pyproject_toml", config_defaults.get("pyproject_toml")
        )
        pyproject_env = str(
            item.get(
                "pyproject_toml_env",
                config_defaults.get("pyproject_toml_env", "VARIANTLIB_PYPROJECT_TOML"),
            )
        )
        if not job_pyproject_toml:
            job_pyproject_toml = os.environ.get(pyproject_env)
        if not job_pyproject_toml:
            raise ValueError(
                f"Job #{i} is missing required field: 'pyproject_toml'. "
                f"You can also set env var {pyproject_env!r}."
            )

        job_output_dir = item.get("output_dir", config_defaults.get("output_dir"))
        output_dir_env = str(
            item.get(
                "output_dir_env",
                config_defaults.get("output_dir_env", "VARIANTLIB_OUTPUT_DIR"),
            )
        )
        if not job_output_dir:
            job_output_dir = os.environ.get(output_dir_env)
        if not job_output_dir:
            raise ValueError(
                f"Job #{i} is missing required field: 'output_dir'. "
                f"You can also set env var {output_dir_env!r}."
            )

        properties_raw = item.get("properties")
        property_single = item.get("property")

        properties: list[str]
        if properties_raw is not None:
            if not isinstance(properties_raw, list) or not all(
                isinstance(x, str) for x in properties_raw
            ):
                raise ValueError(
                    f"Job #{i} field 'properties' must be a list of strings."
                )
            properties = properties_raw
        elif property_single is not None:
            if not isinstance(property_single, str):
                raise ValueError(f"Job #{i} field 'property' must be a string.")
            properties = [property_single]
        else:
            properties = []

        item_variant_label = item.get("variant_label")

        null_variant = bool(item.get("null_variant", False))
        if null_variant and properties:
            raise ValueError(
                f"Job #{i} cannot set both 'null_variant' and property/properties."
            )

        expanded_wheels = _expand_wheel_paths(str(wheel), i)
        for expanded_wheel in expanded_wheels:
            jobs.append(
                Job(
                    wheel=expanded_wheel,
                    variant_label=item_variant_label,
                    properties=properties,
                    null_variant=null_variant,
                    pyproject_toml=str(job_pyproject_toml),
                    output_dir=str(job_output_dir),
                    skip_plugin_validation=bool(
                        item.get("skip_plugin_validation", True)
                    ),
                    no_isolation=bool(item.get("no_isolation", False)),
                    installer=item.get("installer"),
                )
            )

    return jobs


def load_variant_label_aliases(data: Any) -> dict[str, str]:
    if not isinstance(data, dict):
        return {}

    variables = data.get("variables")
    if variables is None:
        return {}
    if not isinstance(variables, dict):
        raise ValueError("Config field 'variables' must be a JSON object.")

    aliases = variables.get("variant_label_aliases")
    if aliases is None:
        return {}
    if not isinstance(aliases, dict):
        raise ValueError("Config field 'variant_label_aliases' must be a JSON object.")

    alias_map: dict[str, str] = {}
    for source_label, target_label in aliases.items():
        if not isinstance(target_label, str):
            raise ValueError(
                "Config field 'variant_label_aliases' must map strings to strings."
            )
        alias_map[source_label] = target_label

    return alias_map


def apply_variant_label_aliases(jobs: list[Job], aliases: dict[str, str]) -> None:
    for job in jobs:
        if job.variant_label is None:
            continue
        job.variant_label = aliases.get(job.variant_label, job.variant_label)


def build_command(job: Job) -> list[str]:
    cmd = [
        VARIANTLIB_CMD,
        "make-variant",
        "-f",
        job.wheel,
        "-o",
        job.output_dir,
        "--pyproject-toml",
        job.pyproject_toml,
    ]

    if job.null_variant:
        cmd.append("--null-variant")
    else:
        if not job.properties:
            raise ValueError(
                f"Job for wheel '{job.wheel}' must provide property/properties "
                "or set null_variant=true."
            )
        for prop in job.properties:
            cmd.extend(["--property", prop])

    if job.variant_label:
        cmd.extend(["--variant-label", job.variant_label])

    if job.skip_plugin_validation:
        cmd.append("--skip-plugin-validation")

    if job.no_isolation:
        cmd.append("--no-isolation")

    if job.installer:
        cmd.extend(["--installer", job.installer])

    return cmd


def _prepare_output_dir(job: Job) -> Path:
    output_dir = Path(job.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path exists but is not a directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_one(job: Job, dry_run: bool) -> tuple[Job, int, str]:
    cmd = build_command(job)
    pretty = shlex.join(cmd)

    if dry_run:
        return job, 0, f"[DRY-RUN] {pretty}\n"

    try:
        output_dir = _prepare_output_dir(job)
    except OSError as exc:
        return job, 1, f"[OUTPUT-DIR-ERROR] {exc}\n"

    result = subprocess.run(cmd, capture_output=True, text=True)
    out = [f"[COMMAND] {pretty}"]
    out.append(f"[OUTPUT-DIR] {output_dir}")
    if result.stdout:
        out.append("[STDOUT]\n" + result.stdout.rstrip())
    if result.stderr:
        out.append("[STDERR]\n" + result.stderr.rstrip())

    if result.returncode == 0:
        try:
            out.append(_verify_built_wheel(job))
        except Exception as exc:
            out.append("[VERIFY-ERROR]\n" + str(exc))
            return job, 1, "\n".join(out) + "\n"

    return job, result.returncode, "\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch wrapper for 'variantlib make-variant'. Reads jobs from JSON and "
            "executes them with optional parallelism."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    parser.add_argument(
        "-l",
        "--variant-label",
        action="append",
        default=[],
        help=(
            "JSON mode: run only jobs with matching variant_label. "
            "Can be repeated or passed as comma-separated values. "
            "Supports alias keys from variables.variant_label_aliases."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.jobs < 1:
        print("--jobs must be >= 1", file=sys.stderr)
        return 2

    variant_labels: list[str] = []
    for raw in args.variant_label:
        variant_labels.extend(_split_labels(raw))

    try:
        config_data = _load_config(args.config)
        jobs = load_jobs(data=config_data)
        variant_label_aliases = load_variant_label_aliases(config_data)
        apply_variant_label_aliases(jobs, variant_label_aliases)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 2

    selected_labels = {
        variant_label_aliases.get(label, label) for label in variant_labels
    }
    if selected_labels:
        jobs = [job for job in jobs if job.variant_label in selected_labels]
        if not jobs:
            print(
                f"No matching jobs for --variant-label: {sorted(selected_labels)}",
                file=sys.stderr,
            )
            return 2

    print(f"Loaded {len(jobs)} jobs from {args.config}")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, job, args.dry_run): job for job in jobs}
        for fut in as_completed(futures):
            job, code, log_text = fut.result()

            print("=" * 80)
            print(f"Wheel: {job.wheel}")
            print(log_text.rstrip())

            if code == 0:
                success += 1
            else:
                failed += 1

    print("=" * 80)
    print(f"Summary: success={success}, failed={failed}, total={len(jobs)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

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
"""
Backward-compatible entrypoint for prompt generation.

`verl.trainer.main_generation` was historically used by examples and external
scripts. The implementation now lives in `main_generation_server`, but we keep
this module to avoid breaking older commands and to translate legacy CLI keys.
"""

from __future__ import annotations

import sys
import warnings

_LEGACY_ARG_PREFIX_MAP = {
    "model.": "actor_rollout_ref.model.",
    "rollout.": "actor_rollout_ref.rollout.",
}

_LEGACY_ARG_EXACT_MAP = {
    "data.path": "data.train_files",
    "data.n_samples": "actor_rollout_ref.rollout.n",
}


def _translate_legacy_cli_args(args: list[str]) -> list[str]:
    translated_args = []
    has_rollout_name = False
    saw_legacy_rollout_arg = False

    for arg in args:
        remapped = arg
        if "=" in arg:
            key, value = arg.split("=", 1)
            normalized_key = key[1:] if key.startswith("+") else key

            if normalized_key == "actor_rollout_ref.rollout.name":
                has_rollout_name = True

            if normalized_key in _LEGACY_ARG_EXACT_MAP:
                new_key = _LEGACY_ARG_EXACT_MAP[normalized_key]
                remapped = f"{new_key}={value}"
            elif normalized_key == "data.output_path":
                remapped = f"+data.output_path={value}"
            else:
                for old_prefix, new_prefix in _LEGACY_ARG_PREFIX_MAP.items():
                    if normalized_key.startswith(old_prefix):
                        new_key = normalized_key.replace(old_prefix, new_prefix, 1)
                        remapped = f"{new_key}={value}"
                        if old_prefix == "rollout.":
                            saw_legacy_rollout_arg = True
                            if new_key == "actor_rollout_ref.rollout.name":
                                has_rollout_name = True
                        break

        translated_args.append(remapped)

    if saw_legacy_rollout_arg and not has_rollout_name:
        translated_args.append("actor_rollout_ref.rollout.name=vllm")

    return translated_args


def main():
    warnings.warn(
        "`verl.trainer.main_generation` is deprecated; use `verl.trainer.main_generation_server` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.argv = [sys.argv[0], *_translate_legacy_cli_args(sys.argv[1:])]
    from verl.trainer.main_generation_server import main as generation_server_main

    generation_server_main()


if __name__ == "__main__":
    main()

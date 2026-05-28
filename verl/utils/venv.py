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
"""Resolve per-role venv specs into Ray ``runtime_env={"py_executable": ...}``.

Per-role Hydra knobs (all default ``None``): ``actor_rollout_ref.{actor,ref,rollout}.venv``,
``critic.venv``, and the global ``trainer.venv`` fallback.

Resolution priority for each Ray worker group:
  1. Per-role ``venv`` field — colocated roles must agree, else raise.
  2. ``trainer.venv`` (preserves prior single-trainer-venv behaviour).
  3. Auto-detect ``.venvs/.venv-<engine>`` from the role's engine field
     (rollout ``name`` / actor / critic ``strategy``); silent on miss.
  4. ``None`` — worker inherits the driver's interpreter.

Specs accept: backend name (looked up under ``<verl>/.venvs/.venv-<name>``),
absolute python / venv path, or a multi-token command line such as
``"uv run --project /abs/path"`` (Ray's ``py_executable`` is experimental).
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import Iterable, Optional


def _verl_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _venvs_dir() -> Path:
    return _verl_root() / ".venvs"


def _resolve_to_python(spec: str) -> Path:
    """Single-token spec → absolute python path. Existence is *not* checked here."""
    p = Path(spec).expanduser()
    if p.is_absolute():
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
        return (p / "bin" / "python").resolve()
    return (_venvs_dir() / f".venv-{spec}" / "bin" / "python").resolve()


def _validate_command_head(spec: str, role: str) -> str:
    """Cheap executability check on the head token; spec is returned unchanged."""
    head = shlex.split(spec)[0]
    head_path = Path(head).expanduser()
    if head_path.is_absolute():
        if not head_path.is_file() or not os.access(head_path, os.X_OK):
            raise FileNotFoundError(f"{role} venv resolver: command {spec!r}: head {head!r} is not an executable file.")
    elif shutil.which(head) is None:
        raise FileNotFoundError(f"{role} venv resolver: command {spec!r}: head {head!r} is not on $PATH.")
    return spec


# Aliases — distinct strategies that intentionally share a single venv.
_VENV_ALIASES = {"fsdp2": "fsdp"}


def _auto_detect_venv(hint: Optional[str]) -> Optional[str]:
    """Return ``<verl>/.venvs/.venv-<hint>/bin/python`` if it exists, else ``None``."""
    if not hint:
        return None
    name = _VENV_ALIASES.get(hint, hint)
    candidate = _venvs_dir() / f".venv-{name}" / "bin" / "python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())
    return None


def resolve_py_executable(
    spec: Optional[str],
    *,
    role: str,
    auto_hint: Optional[str] = None,
) -> Optional[str]:
    """Resolve a venv spec for Ray's ``py_executable``; ``None`` falls through to ``auto_hint``.

    Explicit specs that don't resolve raise ``FileNotFoundError``; auto-detect
    misses stay silent (it's a backward-compat convenience). ``role`` is for
    diagnostics only.
    """
    if spec is None or spec == "":
        return _auto_detect_venv(auto_hint)
    if len(shlex.split(spec)) > 1:
        return _validate_command_head(spec, role)
    py = _resolve_to_python(spec)
    if not py.is_file() or not os.access(py, os.X_OK):
        raise FileNotFoundError(
            f"{role} venv resolver: {spec!r} -> {py} is not an executable Python "
            "interpreter. Either point the config at a real venv "
            "(e.g. actor_rollout_ref.rollout.venv=vllm after "
            "`python manage_envs.py sync vllm`), pass a uv command line such as "
            "'uv run --project /abs/path/to/verl', or leave it null to use the "
            "driver's interpreter (or auto-detected venv)."
        )
    return str(py)


def inject_py_executable(runtime_env: Optional[dict], py_executable: Optional[str]) -> Optional[dict]:
    """Return a copy of ``runtime_env`` with ``py_executable`` set, without clobbering an existing entry."""
    if py_executable is None:
        return runtime_env
    new = dict(runtime_env) if runtime_env else {}
    new.setdefault("py_executable", py_executable)
    return new


# ``str(Role.X)`` → Hydra path of that role's ``venv`` field. Fused-actor
# roles share the actor's interpreter, so all three map to ``actor.venv``.
_ROLE_VENV_PATH = {
    "actor": "actor_rollout_ref.actor.venv",
    "actor_rollout": "actor_rollout_ref.actor.venv",
    "actor_rollout_ref": "actor_rollout_ref.actor.venv",
    "ref": "actor_rollout_ref.ref.venv",
    "critic": "critic.venv",
}

# Engine field used as the auto-detect hint when ``venv`` / ``trainer.venv``
# are both null. Ref inherits the actor's strategy in stock configs.
_ROLE_ENGINE_PATH = {
    "actor": "actor_rollout_ref.actor.strategy",
    "actor_rollout": "actor_rollout_ref.actor.strategy",
    "actor_rollout_ref": "actor_rollout_ref.actor.strategy",
    "ref": "actor_rollout_ref.actor.strategy",
    "critic": "critic.strategy",
}


def resolve_group_py_executable(role_keys: Iterable[str], config) -> Optional[str]:
    """Pick the ``py_executable`` for a Ray worker group hosting ``role_keys``
    (i.e. ``class_dict.keys()``). See module docstring for the priority chain.
    """
    from omegaconf import OmegaConf

    role_keys = list(role_keys)
    role_label = "+".join(sorted(role_keys)) or "trainer"

    candidates: dict[str, str] = {}
    for key in role_keys:
        path = _ROLE_VENV_PATH.get(key)
        if path is None:
            continue
        spec = OmegaConf.select(config, path)
        if spec is None or spec == "":
            continue
        candidates[key] = spec
    distinct = set(candidates.values())
    if len(distinct) > 1:
        details = ", ".join(f"{role}={spec!r}" for role, spec in sorted(candidates.items()))
        raise ValueError(
            "colocated worker group hosts roles with disagreeing ``venv`` specs "
            f"({details}); pick one or unset all but the dominant role and rely "
            "on ``trainer.venv`` as the fallback."
        )
    if len(distinct) == 1:
        return resolve_py_executable(next(iter(distinct)), role=role_label)

    trainer_spec = OmegaConf.select(config, "trainer.venv")
    if trainer_spec is not None and trainer_spec != "":
        return resolve_py_executable(trainer_spec, role=role_label)

    # Auto-detect: only commit when every role with a hint resolves to the
    # same on-disk venv — mismatched engines can't share an interpreter.
    detected: set[str] = set()
    for key in role_keys:
        engine_path = _ROLE_ENGINE_PATH.get(key)
        if engine_path is None:
            continue
        engine = OmegaConf.select(config, engine_path)
        if not engine:
            continue
        py = _auto_detect_venv(engine)
        if py is not None:
            detected.add(py)
    if len(detected) == 1:
        return next(iter(detected))
    return None

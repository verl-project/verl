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
"""Cross-venv runtime helpers.

verl supports running rollout (inference) workers and trainer workers under
different Python interpreters / package sets in the *same* Ray job. This
module is the single source of truth for resolving the user-supplied
*venv spec* (passed via Hydra config) into a string that is suitable for
Ray's ``runtime_env={"py_executable": ...}``.

The two config knobs are:

  * ``actor_rollout_ref.rollout.venv`` — used by every rollout / inference
    Ray actor (set on :class:`verl.workers.config.RolloutConfig.venv`),
  * ``trainer.venv`` — used by every trainer Ray actor (PPO actor, critic,
    ref policy, reward model).

Each accepts any of:

  * an absolute path to a Python interpreter (``/abs/.../bin/python``),
  * an absolute path to a venv directory (we append ``bin/python``),
  * a bare backend name (``vllm`` / ``sglang`` / ``megatron`` / ...) that is
    looked up under ``<verl>/.venvs/.venv-<backend>/bin/python`` — i.e. the
    layout produced by ``manage_envs.py sync``,
  * a full command line such as ``"uv run --project /abs/path/to/verl"``.
    Ray's ``py_executable`` field accepts a string with arguments, so this
    is the recommended way to integrate with ``uv run`` (see Ray docs:
    https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-uv-for-package-management).

If the relevant config value is ``None`` (default) or empty, the resolver
returns ``None`` and the worker inherits the driver's interpreter — the
legacy single-venv behaviour.

NOTE: ``py_executable`` is marked experimental upstream (Ray docs, runtime
environments API reference). Behaviour may change with newer Ray releases.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import Optional


def _verl_root() -> Path:
    # verl/utils/venv.py -> verl/utils -> verl -> <verl pkg root>
    return Path(__file__).resolve().parent.parent.parent


def _venvs_dir() -> Path:
    return _verl_root() / ".venvs"


def _resolve_to_python(spec: str) -> Path:
    """Resolve a single-token ``spec`` to an absolute Python interpreter
    path. The returned path is **not** checked for existence here — callers
    wanting strict validation should call :func:`resolve_py_executable`.
    """
    p = Path(spec).expanduser()
    if p.is_absolute():
        # already a python binary
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
        # absolute path to a venv directory
        candidate = p / "bin" / "python"
        return candidate.resolve()
    # treat as bare backend name: <verl>/.venvs/.venv-<name>/bin/python
    return (_venvs_dir() / f".venv-{spec}" / "bin" / "python").resolve()


def _validate_command_head(spec: str, role: str) -> str:
    """Validate the head token of a multi-token ``py_executable`` command
    line and return ``spec`` unchanged for pass-through to Ray.
    """
    tokens = shlex.split(spec)
    head = tokens[0]
    head_path = Path(head).expanduser()
    if head_path.is_absolute():
        if not head_path.is_file() or not os.access(head_path, os.X_OK):
            raise FileNotFoundError(f"{role} venv resolver: command {spec!r}: head {head!r} is not an executable file.")
    elif shutil.which(head) is None:
        raise FileNotFoundError(f"{role} venv resolver: command {spec!r}: head {head!r} is not on $PATH.")
    return spec


def resolve_py_executable(spec: Optional[str], *, role: str) -> Optional[str]:
    """Turn a config-supplied venv spec (or ``None``) into a string suitable
    for Ray's ``runtime_env={"py_executable": ...}``, or ``None`` if no
    override was requested.

    Single-token specs (backend name / venv dir / abs python path) are
    resolved to an absolute interpreter path. Multi-token specs (e.g.
    ``"uv run --project /abs/path"``) are passed through verbatim after a
    cheap head-token executability check.

    Raises ``FileNotFoundError`` with a clear message if ``spec`` is set but
    cannot be turned into something runnable — failing fast at process start
    is much friendlier than failing later inside a Ray actor.

    ``role`` is used purely for diagnostics (``"rollout"`` / ``"trainer"``).
    """
    if spec is None or spec == "":
        return None
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
            "driver's interpreter."
        )
    return str(py)


def inject_py_executable(runtime_env: Optional[dict], py_executable: Optional[str]) -> Optional[dict]:
    """Return a new ``runtime_env`` dict with ``py_executable`` set, or the
    original dict / ``None`` if no override was requested. ``runtime_env`` is
    not mutated.
    """
    if py_executable is None:
        return runtime_env
    new = dict(runtime_env) if runtime_env else {}
    # Don't clobber a caller-set py_executable (allows per-call overrides).
    new.setdefault("py_executable", py_executable)
    return new

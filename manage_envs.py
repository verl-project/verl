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

"""Per-backend uv environment driver for verl.

Everything that controls *what* gets installed lives in ``pyproject.toml``
(``[project.optional-dependencies]`` + ``[tool.uv]``). This script is a thin
convenience wrapper around ``uv sync`` that:

  * picks the right Python interpreter version per backend,
  * points ``UV_PROJECT_ENVIRONMENT`` at ``verl/.venvs/.venv-<backend>``,
  * (optionally) sets recommended build-time env vars for backends that
    compile native extensions from source (``MAX_JOBS`` / ``NVTE_FRAMEWORK``
    / ``FLASH_ATTENTION_FORCE_BUILD``).

Anything you can do here you can also do directly with ``uv`` from the
``verl/`` directory::

    UV_PROJECT_ENVIRONMENT=.venvs/.venv-vllm \\
        uv sync --extra vllm --python 3.12 --link-mode=copy

Usage::

    python manage_envs.py sync vllm
    python manage_envs.py sync inference
    python manage_envs.py sync all
    python manage_envs.py list
    python manage_envs.py clean vllm
    python manage_envs.py shell vllm
    python manage_envs.py path vllm

Anything after ``--`` is forwarded verbatim to ``uv sync``::

    python manage_envs.py sync megatron -- --reinstall -v

Cross-venv runtime
------------------
For disaggregated jobs where rollout and trainer run in separate Ray actors,
``launch`` *appends Hydra overrides* (``actor_rollout_ref.rollout.venv=...``
and ``trainer.venv=...``) to the user's command — verl reads those at job
start to route each role at the right venv's Python::

    python manage_envs.py launch --rollout vllm --trainer megatron -- \\
        python -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

is equivalent to running the verl driver yourself with the resolved
overrides::

    python -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 \\
        actor_rollout_ref.rollout.venv=/abs/.venvs/.venv-vllm \\
        trainer.venv=/abs/.venvs/.venv-megatron

The driver itself can run in any venv that has verl installed (typically the
trainer venv); ``launch`` only appends config overrides, it does not switch
the driver interpreter.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

INFERENCE_BACKENDS: list[str] = ["vllm", "sglang", "trtllm", "vllm-ascend", "sglang-ascend"]
TRAINING_BACKENDS: list[str] = ["fsdp", "megatron", "mindspeed", "veomni", "nemoautomodel"]
# `cpu` is a CI / unit-test / dev-sanity env (no GPU/NPU runtime) — kept out
# of the inference/training groups but still a first-class backend.
DEV_BACKENDS: list[str] = ["cpu"]
ALL_BACKENDS: list[str] = INFERENCE_BACKENDS + TRAINING_BACKENDS + DEV_BACKENDS

# Per-backend python version. Ascend NPU backends (vllm-ascend / sglang-ascend
# / mindspeed) target the Atlas A2/A3 CANN 8.5.x base image which ships
# python 3.11; everything else uses 3.12.
PYTHON_VERSION: dict[str, str] = {
    "vllm": "3.12",
    "sglang": "3.12",
    "trtllm": "3.12",
    "vllm-ascend": "3.11",
    "sglang-ascend": "3.11",
    "fsdp": "3.12",
    "megatron": "3.12",
    "mindspeed": "3.11",
    "veomni": "3.12",
    "nemoautomodel": "3.12",
    "cpu": "3.12",
}

# Recommended build-time env vars per backend. Optional perf / reproducibility
# hints — `uv sync` works without them. Set here so users don't have to
# remember them.
#
# MAX_JOBS=32 is conservative on purpose. The reference Dockerfiles use
# different values per package (256 for apex/TE, 32 for flash-attn) because
# flash-attn's CUDA compile units are massive — high parallelism OOMs all but
# the beefiest CI hosts. We pick the floor (32) so the build succeeds
# everywhere; bump it via `MAX_JOBS=128 python manage_envs.py sync megatron`
# on hosts with plenty of RAM if you want apex/TE to compile faster.
BUILD_ENV: dict[str, dict[str, str]] = {
    "megatron": {
        "MAX_JOBS": "32",
        "NVTE_FRAMEWORK": "pytorch",
        "NVTE_BUILD_THREADS_PER_JOB": "4",
        "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
    },
    "mindspeed": {
        "MAX_JOBS": "32",
    },
    # vllm / sglang / trtllm install pre-built wheels and don't need build env.
}

GROUPS: dict[str, list[str]] = {
    "all": ALL_BACKENDS,
    "inference": INFERENCE_BACKENDS,
    "training": TRAINING_BACKENDS,
    "dev": DEV_BACKENDS,
}

VERL_DIR = Path(__file__).resolve().parent
VENVS_DIR = (VERL_DIR / ".venvs").resolve()


def _venv_path(backend: str) -> Path:
    return VENVS_DIR / f".venv-{backend}"


def _venv_python(backend: str) -> Path:
    return _venv_path(backend) / "bin" / "python"


def _require_uv() -> None:
    if shutil.which("uv") is None:
        sys.exit("error: uv is not installed. Get it from https://astral.sh/uv")


def _expand(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item in GROUPS:
            out.extend(GROUPS[item])
        elif item in ALL_BACKENDS:
            out.append(item)
        else:
            sys.exit(f"error: unknown backend {item!r}. valid: {', '.join(ALL_BACKENDS)} (or all/inference/training)")
    seen: set[str] = set()
    return [b for b in out if not (b in seen or seen.add(b))]


def _run(cmd: list[str], env_overrides: dict[str, str] | None = None) -> int:
    if env_overrides:
        prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
        print(f"$ {prefix} {' '.join(cmd)}", flush=True)
    else:
        print(f"$ {' '.join(cmd)}", flush=True)
    env = {**os.environ, **(env_overrides or {})}
    return subprocess.run(cmd, env=env, cwd=str(VERL_DIR)).returncode


def cmd_sync(args: argparse.Namespace) -> int:
    backends = _expand(args.backends)
    _require_uv()
    VENVS_DIR.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        venv = _venv_path(backend)
        env_overrides = {
            "UV_PROJECT_ENVIRONMENT": str(venv),
            **BUILD_ENV.get(backend, {}),
        }
        cmd = [
            "uv",
            "sync",
            "--link-mode=copy",
            "--extra",
            backend,
            "--python",
            PYTHON_VERSION[backend],
            *args.uv_args,
        ]
        rc = _run(cmd, env_overrides=env_overrides)
        if rc:
            print(f"error: uv sync failed for {backend}", file=sys.stderr)
            return rc
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    if "all" in args.backends:
        if VENVS_DIR.exists():
            shutil.rmtree(VENVS_DIR)
            print(f"removed {VENVS_DIR}")
        return 0
    for backend in _expand(args.backends):
        venv = _venv_path(backend)
        if venv.exists():
            shutil.rmtree(venv)
            print(f"removed {venv}")
        else:
            print(f"(no-op) {venv} does not exist")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    fmt = "{:<16} {:<10} {:<10} {}"
    print(fmt.format("BACKEND", "INSTALLED", "PYTHON", "PATH"))
    for backend in ALL_BACKENDS:
        venv = _venv_path(backend)
        py = _venv_python(backend)
        if py.exists():
            try:
                ver = subprocess.check_output(
                    [str(py), "-c", "import sys;print('%d.%d.%d'%sys.version_info[:3])"],
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                ver = "?"
            print(fmt.format(backend, "yes", ver, str(venv)))
        else:
            print(fmt.format(backend, "no", "-", str(venv)))
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    print(_venv_path(args.backend))
    return 0


def cmd_shell(args: argparse.Namespace) -> int:
    venv = _venv_path(args.backend)
    if not _venv_python(args.backend).exists():
        sys.exit(f"error: venv missing at {venv}. Run 'python manage_envs.py sync {args.backend}' first.")
    shell = os.environ.get("SHELL", "/bin/bash")
    print(f"spawning {shell} inside .venv-{args.backend} (exit to leave)")
    env = {
        **os.environ,
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "PS1": f"(verl-{args.backend}) {os.environ.get('PS1', '$ ')}",
    }
    return subprocess.run([shell], env=env).returncode


def _resolve_role_venv(spec: str, role: str) -> str:
    """Validate ``spec`` and return whatever string should go into
    ``VERL_*_VENV``. Accepts the same forms as :mod:`verl.utils.venv`:
    backend name, absolute venv directory, absolute python interpreter, or
    a full command line such as ``"uv run --project /abs/path/to/verl"``
    (Ray's ``py_executable`` accepts arguments — see
    https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-uv-for-package-management).
    """
    import shlex

    tokens = shlex.split(spec)
    if len(tokens) > 1:
        head = tokens[0]
        head_path = Path(head).expanduser()
        if head_path.is_absolute():
            if not head_path.is_file() or not os.access(head_path, os.X_OK):
                sys.exit(f"error: --{role} {spec!r}: head {head!r} is not an executable file")
        elif shutil.which(head) is None:
            sys.exit(f"error: --{role} {spec!r}: head {head!r} is not on $PATH")
        return spec

    p = Path(spec).expanduser()
    if p.is_absolute():
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
        candidate = p / "bin" / "python"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(p.resolve())
        sys.exit(
            f"error: --{role} {spec!r}: not a valid venv (no executable bin/python under {p}, "
            "and the path itself is not an executable interpreter)"
        )
    if spec not in ALL_BACKENDS:
        sys.exit(
            f"error: --{role} {spec!r}: unknown backend. "
            f"valid: {', '.join(ALL_BACKENDS)} (or pass an absolute venv path / "
            "a 'uv run --project ...' command line)"
        )
    if not _venv_python(spec).exists():
        sys.exit(
            f"error: --{role} {spec!r}: venv missing at {_venv_path(spec)}. "
            f"Run 'python manage_envs.py sync {spec}' first."
        )
    return str(_venv_path(spec))


def cmd_launch(args: argparse.Namespace) -> int:
    """Run a verl command with rollout and trainer routed at separate venvs
    by appending Hydra-style config overrides to the user's command.

    By default this appends ``actor_rollout_ref.rollout.venv=<rollout-spec>``
    and ``trainer.venv=<trainer-spec>``. Override the keys via
    ``--rollout-key`` / ``--trainer-key`` if your entry point uses different
    Hydra paths.
    """
    if not args.command:
        sys.exit("error: nothing to run after `--`. Try: launch --rollout vllm --trainer megatron -- python -m ...")
    rollout_spec = _resolve_role_venv(args.rollout, role="rollout")
    trainer_spec = _resolve_role_venv(args.trainer, role="trainer")
    overrides = [
        f"{args.rollout_key}={rollout_spec}",
        f"{args.trainer_key}={trainer_spec}",
    ]
    cmd = [*args.command, *overrides]
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(VERL_DIR)).returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_envs.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    backends_help = (
        "backend name(s); shortcuts: all, "
        "inference (vllm sglang trtllm vllm-ascend sglang-ascend), "
        "training (fsdp megatron mindspeed veomni nemoautomodel), "
        "dev (cpu)"
    )

    s = sub.add_parser("sync", help="create or refresh one or more backend venvs")
    s.add_argument("backends", nargs="+", help=backends_help)
    s.add_argument(
        "uv_args",
        nargs=argparse.REMAINDER,
        help="anything after `--` is forwarded to `uv sync`",
    )
    s.set_defaults(func=cmd_sync)

    c = sub.add_parser("clean", help="remove one or more backend venvs")
    c.add_argument("backends", nargs="+", help=backends_help)
    c.set_defaults(func=cmd_clean)

    sub.add_parser("list", help="show every backend venv and its python version").set_defaults(func=cmd_list)

    sh = sub.add_parser("shell", help="spawn an interactive shell with the venv on PATH")
    sh.add_argument("backend", choices=ALL_BACKENDS)
    sh.set_defaults(func=cmd_shell)

    p = sub.add_parser("path", help="print the path of a backend venv")
    p.add_argument("backend", choices=ALL_BACKENDS)
    p.set_defaults(func=cmd_path)

    lh = sub.add_parser(
        "launch",
        help="run a command with rollout/trainer routed at separate venvs (cross-venv runtime)",
        description=(
            "Appends Hydra config overrides for the rollout and trainer venvs to the user's "
            "command. Both flags accept a backend name (vllm/megatron/...), an absolute venv "
            "path / python path, or a 'uv run --project ...' command line."
        ),
    )
    lh.add_argument("--rollout", required=True, help="rollout/inference venv spec")
    lh.add_argument("--trainer", required=True, help="trainer/training venv spec")
    lh.add_argument(
        "--rollout-key",
        default="actor_rollout_ref.rollout.venv",
        help="Hydra path for the rollout venv override (default: %(default)s)",
    )
    lh.add_argument(
        "--trainer-key",
        default="trainer.venv",
        help="Hydra path for the trainer venv override (default: %(default)s)",
    )
    lh.add_argument("command", nargs=argparse.REMAINDER, help="command to run after `--`")
    lh.set_defaults(func=cmd_launch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if hasattr(args, "uv_args") and args.uv_args and args.uv_args[0] == "--":
        args.uv_args = args.uv_args[1:]
    if hasattr(args, "command") and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

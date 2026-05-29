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
and ``uv.lock``:

* ``[project.optional-dependencies]`` declares one extra per backend (vllm,
  sglang, megatron, fsdp, cpu, ...).
* ``[tool.uv].conflicts`` marks the backends as mutually exclusive so
  ``uv lock`` resolves each as an independent fork in a single lockfile.
* ``[tool.uv].override-dependencies`` pins ``transformers`` and (per-extra)
  the matching ``nvidia-cudnn-cu{12,13}`` / ``nvidia-nccl-cu{12,13}`` wheels
  so they line up with the apt deb versions in
  ``docker/Dockerfile.uv.cu{129,130}``. cu13 backends are vllm / sglang /
  fsdp / megatron; cu12.9 backends are trtllm / veomni / nemoautomodel.
* ``[tool.uv].sources`` / ``[tool.uv].index`` route git packages and PyTorch
  wheels.
* ``uv.lock`` is committed and is the single source of truth for resolved
  versions across every backend.

This script is a thin convenience wrapper around::

    UV_PROJECT_ENVIRONMENT=.venvs/.venv-<backend> \\
        uv sync --frozen --extra <backend> \\
                --python <python-version> \\
                --link-mode=copy

so that:

  * each backend gets a separate venv under ``.venvs/.venv-<backend>``,
  * ``--frozen`` keeps every install reproducible from ``uv.lock`` (no
    silent re-resolves),
  * recommended build-time env vars are set for backends that compile
    native extensions from source (``MAX_JOBS`` / ``NVTE_FRAMEWORK`` /
    ``FLASH_ATTENTION_FORCE_BUILD``).

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

Re-locking after a dependency change
------------------------------------
Edit ``pyproject.toml`` (and the matching apt deb in ``docker/Dockerfile.uv.cu130``
if it's a system lib), then::

    uv lock                                    # regenerate uv.lock
    python manage_envs.py sync <backend>       # validate locally
    git add pyproject.toml uv.lock             # commit them together

Cross-venv runtime
------------------
For disaggregated jobs ``launch`` *appends Hydra overrides* (one per role)
to the user's command — verl reads those at job start to route each Ray
worker group at the right venv's Python. Per-role flags map to per-role
``*.venv`` Hydra fields (see :mod:`verl.utils.venv` for the resolution
rules). ``--trainer`` is a global fallback for trainer-side groups whose
role-level field is left null.

Common case (one venv for all training, one for inference)::

    python manage_envs.py launch --rollout vllm --trainer megatron -- \\
        python -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

is equivalent to::

    python -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 \\
        actor_rollout_ref.rollout.venv=/abs/.venvs/.venv-vllm \\
        trainer.venv=/abs/.venvs/.venv-megatron

Disaggregated case (mix engines per role)::

    python manage_envs.py launch --rollout vllm --actor megatron \\
        --ref fsdp --critic fsdp -- python -m verl.trainer.main_ppo ...

emits ``actor_rollout_ref.rollout.venv=...``,
``actor_rollout_ref.actor.venv=...``,
``actor_rollout_ref.ref.venv=...`` and ``critic.venv=...`` separately. Roles
that share a Ray actor (e.g. fused ActorRolloutRef) must agree — verl
raises a clear error at job start otherwise.

The driver itself can run in any venv that has verl installed (typically
the trainer venv); ``launch`` only appends config overrides, it does not
switch the driver interpreter.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

INFERENCE_BACKENDS: list[str] = ["vllm", "sglang", "trtllm"]
TRAINING_BACKENDS: list[str] = ["fsdp", "megatron", "veomni", "nemoautomodel"]
# `cpu` is a CI / unit-test / dev-sanity env (no GPU runtime) — kept out
# of the inference/training groups but still a first-class backend.
DEV_BACKENDS: list[str] = ["cpu"]
ALL_BACKENDS: list[str] = INFERENCE_BACKENDS + TRAINING_BACKENDS + DEV_BACKENDS

# Per-backend CUDA major. The lockfile has two parallel cu forks (declared
# in pyproject.toml's [tool.uv.sources] / override-dependencies markers),
# each backed by a matching Docker base image:
#
#   * cu13 (docker/Dockerfile.uv.cu130, cuda:13.0.2) — torch 2.11 / cu130
#     wheels; matching apt cuDNN-cuda-13 / nccl-cuda13.0 debs.
#   * cu12.9 (docker/Dockerfile.uv.cu129, cuda:12.9.1) — torch 2.9.x-2.10
#     / cu129 wheels; matching apt cuDNN-cuda-12 / nccl-cuda12.9 debs.
#     Used for backends whose upstream pins still trail torch 2.11
#     (trtllm, veomni, nemoautomodel).
#
# Use this map to validate "image vs backend" combos and to drive the
# Dockerfile.uv.cu* TARGETS defaults. `cpu` carries no CUDA dep so it is
# valid in both images.
BACKEND_CUDA_MAJOR: dict[str, str] = {
    "vllm": "13",
    "sglang": "13",
    "fsdp": "13",
    "megatron": "13",
    "trtllm": "12",
    "veomni": "12",
    "nemoautomodel": "12",
    "cpu": "any",
}

# Per-backend python version. Every supported backend currently targets
# CUDA-on-x86_64-Linux + Python 3.12; Ascend NPU and aarch64 GPU variants
# are out of scope for the uv flow (see [tool.uv].environments).
PYTHON_VERSION: dict[str, str] = {
    "vllm": "3.12",
    "sglang": "3.12",
    "trtllm": "3.12",
    "fsdp": "3.12",
    "megatron": "3.12",
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
#
# Every CUDA-13 / torch-2.11 backend that lists `flash-attn==2.8.3` source-builds
# it (no precompiled cu13 wheel on PyPI yet); FLASH_ATTENTION_FORCE_BUILD=TRUE
# mirrors Dockerfile.stable.{vllm,sglang}.
_FLASH_ATTN_BUILD_ENV: dict[str, str] = {
    "MAX_JOBS": "32",
    "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
}
BUILD_ENV: dict[str, dict[str, str]] = {
    "fsdp": {**_FLASH_ATTN_BUILD_ENV},
    "megatron": {
        **_FLASH_ATTN_BUILD_ENV,
        "NVTE_FRAMEWORK": "pytorch",
        "NVTE_BUILD_THREADS_PER_JOB": "4",
    },
    "veomni": {**_FLASH_ATTN_BUILD_ENV},
    "nemoautomodel": {**_FLASH_ATTN_BUILD_ENV},
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
    """Create or refresh per-backend venvs via ``uv sync --frozen --extra``.

    Each backend resolves to its own fork in ``uv.lock`` (declared via
    ``[tool.uv].conflicts``), so ``uv sync --extra <backend>`` installs only
    that fork into the venv at ``UV_PROJECT_ENVIRONMENT``. ``--frozen``
    rejects any drift between ``pyproject.toml`` and ``uv.lock`` — bumping
    a pin requires re-running ``uv lock``.

    If ``uv.lock`` is missing (fresh checkout, or you just bumped a pin
    locally), ``--frozen`` is dropped from the first sync so ``uv`` can
    generate the lockfile as a side effect. Subsequent backends in the same
    invocation, and all subsequent invocations, find the lockfile and use
    ``--frozen`` automatically. Commit the produced ``uv.lock`` alongside
    your ``pyproject.toml`` change to keep installs reproducible.
    """
    backends = _expand(args.backends)
    _require_uv()
    VENVS_DIR.mkdir(parents=True, exist_ok=True)
    if not (VERL_DIR / "uv.lock").is_file():
        print(
            "warning: uv.lock not found — the first uv sync will run without "
            "--frozen and generate uv.lock. Commit it next to pyproject.toml.",
            file=sys.stderr,
        )
    for backend in backends:
        venv = _venv_path(backend)
        env_overrides = {
            **BUILD_ENV.get(backend, {}),
            "UV_PROJECT_ENVIRONMENT": str(venv),
        }
        # Re-check on every iteration: the first sync of a fresh checkout
        # writes uv.lock, so subsequent backends pick up --frozen.
        frozen_args = ["--frozen"] if (VERL_DIR / "uv.lock").is_file() else []
        cmd = [
            "uv",
            "sync",
            *frozen_args,
            "--extra",
            backend,
            "--python",
            PYTHON_VERSION[backend],
            "--link-mode=copy",
            *args.uv_args,
        ]
        rc = _run(cmd, env_overrides)
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


# Per-role launch flags → Hydra paths. Mirrors ``_ROLE_VENV_PATH`` in
# ``verl/utils/venv.py``; keep them in sync. ``trainer`` is the legacy-shape
# fallback and only emits ``trainer.venv=...``.
_LAUNCH_ROLE_KEYS: dict[str, str] = {
    "rollout": "actor_rollout_ref.rollout.venv",
    "actor": "actor_rollout_ref.actor.venv",
    "ref": "actor_rollout_ref.ref.venv",
    "critic": "critic.venv",
    "trainer": "trainer.venv",
}


def cmd_launch(args: argparse.Namespace) -> int:
    """Run a verl command with each role routed at a separate venv by
    appending Hydra-style config overrides to the user's command.

    Per-role flags accepted: ``--rollout``, ``--actor``, ``--ref``,
    ``--critic``, ``--trainer``. Each maps to the matching ``*.venv`` field
    (see ``_LAUNCH_ROLE_KEYS``). ``--trainer`` is the global fallback for any
    trainer-side group whose role-level field is left null. Pass ``--*-key``
    to override the Hydra path emitted for any role.
    """
    if not args.command:
        sys.exit(
            "error: nothing to run after `--`. Try:\n"
            "  launch --rollout vllm --trainer megatron -- python -m ...\n"
            "  launch --rollout vllm --actor megatron --critic fsdp -- python -m ..."
        )
    overrides: list[str] = []
    keys = {
        "rollout": args.rollout_key,
        "actor": args.actor_key,
        "ref": args.ref_key,
        "critic": args.critic_key,
        "trainer": args.trainer_key,
    }
    for role, hydra_key in keys.items():
        spec = getattr(args, role, None)
        if not spec:
            continue
        resolved = _resolve_role_venv(spec, role=role)
        overrides.append(f"{hydra_key}={resolved}")
    if not overrides:
        sys.exit("error: at least one of --rollout/--actor/--ref/--critic/--trainer must be set")
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
        "inference (vllm sglang trtllm), "
        "training (fsdp megatron veomni nemoautomodel), "
        "dev (cpu). x86_64 Linux only. Backends are split between two "
        "CUDA majors (see BACKEND_CUDA_MAJOR): vllm/sglang/fsdp/megatron "
        "build against cu13 (Dockerfile.uv.cu130, torch 2.11); "
        "trtllm/veomni/nemoautomodel build against cu12.9 "
        "(Dockerfile.uv.cu129, torch 2.9.x-2.10) until upstream catches "
        "up. Ascend NPU and aarch64 GPU variants are out of scope here."
    )

    s = sub.add_parser("sync", help="create or refresh one or more backend venvs (uv sync --frozen)")
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
        help="run a command with each role routed at a separate venv (cross-venv runtime)",
        description=(
            "Appends Hydra config overrides for one or more role venvs to the user's command. "
            "Each --<role> flag accepts a backend name (vllm/megatron/...), an absolute venv "
            "path / python path, or a 'uv run --project ...' command line. --trainer is the "
            "global fallback for any trainer-side worker group whose role-level field is null."
        ),
    )
    lh.add_argument("--rollout", default=None, help="rollout/inference venv spec")
    lh.add_argument("--actor", default=None, help="actor (PPO trainer) venv spec")
    lh.add_argument("--ref", default=None, help="reference-policy venv spec (disaggregated only)")
    lh.add_argument("--critic", default=None, help="critic venv spec")
    lh.add_argument("--trainer", default=None, help="fallback venv for any trainer-side group with no per-role spec")
    lh.add_argument(
        "--rollout-key",
        default=_LAUNCH_ROLE_KEYS["rollout"],
        help="Hydra path for the rollout venv override (default: %(default)s)",
    )
    lh.add_argument(
        "--actor-key",
        default=_LAUNCH_ROLE_KEYS["actor"],
        help="Hydra path for the actor venv override (default: %(default)s)",
    )
    lh.add_argument(
        "--ref-key",
        default=_LAUNCH_ROLE_KEYS["ref"],
        help="Hydra path for the ref venv override (default: %(default)s)",
    )
    lh.add_argument(
        "--critic-key",
        default=_LAUNCH_ROLE_KEYS["critic"],
        help="Hydra path for the critic venv override (default: %(default)s)",
    )
    lh.add_argument(
        "--trainer-key",
        default=_LAUNCH_ROLE_KEYS["trainer"],
        help="Hydra path for the trainer-fallback venv override (default: %(default)s)",
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

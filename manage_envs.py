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

Each backend has its own independent lockfile under
``requirements/<backend>.lock``, generated from ``[project.optional-dependencies].<backend>``
in ``pyproject.toml``. There is intentionally no universal ``uv.lock``:
``manage_envs.py sync vllm`` resolves *only* the vllm extra and never
fetches another backend's git sources or URL-pinned wheels.

Two-step pipeline per backend::

    # 1. Compile pyproject extra -> requirements lockfile (when missing
    #    or when --recompile is passed)
    uv pip compile pyproject.toml --extra <backend> \\
        --python <python-version> \\
        --output-file requirements/<backend>.lock

    # 2. Sync venv from the lockfile, then install verl editable
    uv venv .venvs/.venv-<backend> --python <python-version>
    uv pip sync requirements/<backend>.lock \\
        --python .venvs/.venv-<backend>/bin/python \\
        --link-mode=copy
    uv pip install -e . --no-deps \\
        --python .venvs/.venv-<backend>/bin/python

What that buys:

  * ``manage_envs.py sync vllm`` only resolves vllm extras (no megatron /
    apex / TE / flash-attn etc. appear during lock or install),
  * lockfiles can be checked in for reproducibility,
  * pyproject.toml stays the single source of truth for backend recipes,
  * ``[tool.uv.sources]`` per-extra URL / index routing still applies.

Usage::

    python manage_envs.py sync vllm
    python manage_envs.py sync inference
    python manage_envs.py sync all
    python manage_envs.py lock vllm                # (re)compile only
    python manage_envs.py lock all --recompile     # refresh every lockfile
    python manage_envs.py list
    python manage_envs.py clean vllm
    python manage_envs.py shell vllm
    python manage_envs.py path vllm

Anything after ``--`` on ``sync`` / ``lock`` is forwarded to the underlying
``uv pip ...`` invocation::

    python manage_envs.py sync megatron -- --reinstall -v

Re-locking after a dependency change
------------------------------------
Edit ``pyproject.toml`` (and the matching apt deb in
``docker/Dockerfile.uv.cu{130,129}`` if it's a system lib), then::

    python manage_envs.py lock <backend> --recompile  # refresh that lockfile
    python manage_envs.py sync <backend>              # validate locally
    git add pyproject.toml requirements/<backend>.lock

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

Disaggregated case (mix engines per role)::

    python manage_envs.py launch --rollout vllm --actor megatron \\
        --ref fsdp --critic fsdp -- python -m verl.trainer.main_ppo ...
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

# PyTorch wheel index for ``uv pip sync``. ``uv pip compile`` reads
# ``[tool.uv.sources]`` from pyproject.toml, but sync only sees the lockfile —
# pass ``--torch-backend`` so ``torch==*+cu130`` / ``+cu129`` wheels resolve.
# See https://docs.astral.sh/uv/guides/integration/pytorch/
BACKEND_TORCH_BACKEND: dict[str, str] = {
    "vllm": "cu130",
    "sglang": "cu130",
    "fsdp": "cu130",
    "megatron": "cu130",
    "trtllm": "cu129",
    "veomni": "cu129",
    "nemoautomodel": "cu129",
    "cpu": "cpu",
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

# Recommended build-time env vars per backend. Only `megatron` source-builds
# today (apex + TE v2.15); every other backend installs only prebuilt wheels
# (flash-attn 2.8.3 from mjun0812/flash-attention-prebuild-wheels, torch /
# vllm / sglang / trtllm from PyPI / pytorch indexes). MAX_JOBS=32 is the
# floor that succeeds everywhere; bump via `MAX_JOBS=128 python
# manage_envs.py sync megatron` on big hosts.
BUILD_ENV: dict[str, dict[str, str]] = {
    "megatron": {
        "MAX_JOBS": "32",
        "NVTE_FRAMEWORK": "pytorch",
        "NVTE_BUILD_THREADS_PER_JOB": "4",
    },
}

GROUPS: dict[str, list[str]] = {
    "all": ALL_BACKENDS,
    "inference": INFERENCE_BACKENDS,
    "training": TRAINING_BACKENDS,
    "dev": DEV_BACKENDS,
}

VERL_DIR = Path(__file__).resolve().parent
VENVS_DIR = (VERL_DIR / ".venvs").resolve()
REQUIREMENTS_DIR = (VERL_DIR / "requirements").resolve()


def _venv_path(backend: str) -> Path:
    return VENVS_DIR / f".venv-{backend}"


def _venv_python(backend: str) -> Path:
    return _venv_path(backend) / "bin" / "python"


def _lock_path(backend: str) -> Path:
    return REQUIREMENTS_DIR / f"{backend}.lock"


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


def _compile_lockfile(
    backend: str,
    *,
    extra_args: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> int:
    """Run ``uv pip compile pyproject.toml --extra <backend>`` and write
    the resolved requirements to ``requirements/<backend>.lock``.

    Resolution is scoped to a single extra: only that extra's transitive
    closure is fetched. Other backends' git sources / URL wheels are
    never touched.
    """
    REQUIREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(backend)
    cmd = [
        "uv",
        "pip",
        "compile",
        "pyproject.toml",
        "--extra",
        backend,
        "--python",
        PYTHON_VERSION[backend],
        "--output-file",
        str(lock),
        *(extra_args or []),
    ]
    return _run(cmd, env_overrides)


def cmd_lock(args: argparse.Namespace) -> int:
    """(Re)compile per-backend lockfiles via ``uv pip compile``.

    Without ``--recompile``, only missing lockfiles are generated. With
    ``--recompile``, every requested backend is re-resolved (use this
    after bumping a pin in ``pyproject.toml``).
    """
    backends = _expand(args.backends)
    _require_uv()
    for backend in backends:
        lock = _lock_path(backend)
        if lock.is_file() and not args.recompile:
            print(f"(skip) {lock} already exists; pass --recompile to refresh", flush=True)
            continue
        rc = _compile_lockfile(backend)
        if rc:
            print(f"error: uv pip compile failed for {backend}", file=sys.stderr)
            return rc
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Create or refresh per-backend venvs.

    For each backend:

    1. Ensure ``requirements/<backend>.lock`` exists (compile from
       ``[project.optional-dependencies].<backend>`` if missing).
    2. ``uv venv .venvs/.venv-<backend> --python <python-version>``
       (skipped if the venv already exists, so a re-run after a failed
       ``pip sync`` reuses it; use ``clean`` to force a rebuild).
    3. ``uv pip sync <lock> --python <venv-python> --link-mode=copy
       --torch-backend <cu130|cu129|cpu>`` — installs *exactly* what the
       lockfile says. The torch-backend flag is required so PyTorch CUDA
       wheels (``torch==*+cu130`` etc.) resolve from the right index;
       ``uv pip compile`` already knows the index via pyproject.toml, but
       sync does not read pyproject.toml.
    4. ``uv pip install -e . --no-deps --python <venv-python>`` to put
       verl itself into the venv in editable mode (deps come from the
       lockfile, so ``--no-deps`` is correct here).

    Step (1) is what makes ``sync vllm`` only touch vllm: the compile is
    scoped to a single extra. Step (3) installs only what is in
    ``requirements/<backend>.lock`` — verbatim — so other backends'
    packages can never leak in.
    """
    backends = _expand(args.backends)
    _require_uv()
    VENVS_DIR.mkdir(parents=True, exist_ok=True)
    REQUIREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        lock = _lock_path(backend)
        venv = _venv_path(backend)
        py = PYTHON_VERSION[backend]
        env_overrides = {**BUILD_ENV.get(backend, {})}

        if not lock.is_file():
            print(f"info: {lock} missing — compiling from pyproject.toml [{backend}]", flush=True)
            rc = _compile_lockfile(backend)
            if rc:
                print(f"error: uv pip compile failed for {backend}", file=sys.stderr)
                return rc

        if _venv_python(backend).exists():
            # Idempotent re-run: a prior `sync` may have failed mid-`pip sync`
            # (e.g. a transient wheel download error) and left the venv behind.
            # `uv venv` refuses an existing dir without --clear, so skip
            # creation and let `uv pip sync` below reconcile it to the lock.
            # Use `clean <backend>` first to force a from-scratch rebuild.
            print(f"(reuse) venv already exists at {venv}; skipping uv venv", flush=True)
        else:
            rc = _run(["uv", "venv", str(venv), "--python", py, "--link-mode=copy"])
            if rc:
                print(f"error: uv venv failed for {backend}", file=sys.stderr)
                return rc

        venv_py = str(_venv_python(backend))
        rc = _run(
            [
                "uv",
                "pip",
                "sync",
                str(lock),
                "--python",
                venv_py,
                "--link-mode=copy",
                "--torch-backend",
                BACKEND_TORCH_BACKEND[backend],
                *args.uv_args,
            ],
            env_overrides,
        )
        if rc:
            print(f"error: uv pip sync failed for {backend}", file=sys.stderr)
            return rc

        rc = _run(
            [
                "uv",
                "pip",
                "install",
                "-e",
                ".",
                "--no-deps",
                "--python",
                venv_py,
                "--link-mode=copy",
            ],
            env_overrides,
        )
        if rc:
            print(f"error: editable verl install failed for {backend}", file=sys.stderr)
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

    s = sub.add_parser(
        "sync",
        help="create or refresh one or more backend venvs (per-backend lockfile + uv pip sync)",
    )
    s.add_argument("backends", nargs="+", help=backends_help)
    s.add_argument(
        "uv_args",
        nargs=argparse.REMAINDER,
        help="anything after `--` is forwarded to the underlying `uv pip sync`",
    )
    s.set_defaults(func=cmd_sync)

    lk = sub.add_parser(
        "lock",
        help="(re)compile per-backend lockfiles in requirements/<backend>.lock",
    )
    lk.add_argument(
        "--recompile",
        action="store_true",
        help="re-resolve and overwrite each lockfile (default: only compile missing ones)",
    )
    lk.add_argument("backends", nargs="+", help=backends_help)
    lk.set_defaults(func=cmd_lock)

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

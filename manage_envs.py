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

"""Universal-lock uv environment driver for verl.

verl uses **one** ``uv.lock`` for the whole project. Every backend is a PEP
621 extra in ``pyproject.toml``; mutually exclusive ones are declared in
``[tool.uv].conflicts`` so a single ``uv lock`` resolves all of them into the
one lockfile. At runtime you materialize exactly one conflict-free
combination of extras into the project venv (``.venv``) and run everything —
every Ray worker group — from it. There is no per-backend lockfile and no Ray
``py_executable`` switching.

Typical flow::

    python manage_envs.py lock                       # (re)generate uv.lock
    python manage_envs.py sync fsdp vllm             # build .venv for a run
    source .venv/bin/activate                        # or: manage_envs.py shell ...
    python -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

    # one-shot equivalent (uv resolves + runs from .venv):
    python manage_envs.py run fsdp vllm -- python -m verl.trainer.main_ppo ...

Commands::

    lock                 # uv lock  -> regenerate the universal uv.lock
    sync   <extras...>   # uv sync --extra ...  -> materialize .venv (runtime)
    run    <extras...> -- <cmd...>   # uv run --extra ... -- <cmd>
    shell  <extras...>   # sync, then open a shell with .venv activated
    list                 # extras, conflict rules, .venv state, prefetch plan
    clean                # remove .venv
    prefetch             # FIRST-TIME / Docker only: warm the uv cache with
                         #   every backend's wheels & source builds (see below)

Anything after ``--`` on ``lock`` / ``sync`` / ``prefetch`` is forwarded to
the underlying ``uv`` invocation, e.g.::

    python manage_envs.py sync megatron -- --reinstall -v

Keeping internal packages out of uv's hands
--------------------------------------------
If your site ships its own build of some package (e.g. an internal ``ray`` or
``wandb``) that ``uv sync`` must not overwrite or delete, set
``VERL_UV_NO_INSTALL`` to a space/comma-separated list of distribution names::

    export VERL_UV_NO_INSTALL="ray wandb"
    python manage_envs.py sync fsdp vllm     # everything except ray / wandb
    uv pip install <your internal ray / wandb wheels>   # then add your builds

``sync`` / ``shell`` then pass ``--no-install-package <name>`` (uv won't install
or replace it) plus ``--inexact`` (uv won't remove an already-installed build),
and ``run`` adds ``--no-sync`` so it executes in your curated ``.venv``
untouched. Only the named distribution is skipped — not its dependencies — and
an empty / unset value keeps uv's default exact sync. (uv has no env var of its
own for this, which is why it lives here.)

`sync` vs `prefetch`
--------------------
``sync`` is the **runtime** command: it installs one conflict-free extra
combination into ``.venv`` so you can train/serve. ``prefetch`` is a
**first-time / image-build** helper: it downloads & builds *every* backend's
dependencies into the uv cache (``$UV_CACHE_DIR``, default ``~/.cache/uv``) so
later ``sync`` runs are fast/offline. Because the backends conflict,
``prefetch`` cannot produce a single usable env — it syncs throwaway envs
purely to populate the cache (passing ``--no-install-project`` so only
dependencies are cached, not verl itself) and never creates or modifies
``.venv``.

In Docker this is what makes one image serve any backend: bake ``prefetch``
as a real layer (point ``UV_CACHE_DIR`` at an in-image path and do **not** use
a ``--mount=type=cache``) so the cache ships *inside* the image. The container
then picks its combination at run time — ``sync <extras...>`` builds ``.venv``
from the baked cache, offline — instead of hard-coding one combo at build
time. Do **not** use ``prefetch`` as a runtime sync; use ``sync <extras...>``
for that.

This driver exposes two GPU torch "worlds" plus a CPU slice, all in one lock:
the cu13.0 / torch-2.11 backends (vllm, sglang, fsdp, megatron), the cu12.9 /
torch-2.9.1 backends (veomni, nemoautomodel), and the GPU-free ``cpu`` slice.
They never mix in one ``.venv`` (see the conflict sets). ``prefetch`` can be
scoped to one world via the ``cu130`` / ``cu129`` shortcuts so each Docker
image bakes only its own backends. trtllm is deferred in pyproject.toml (a
CUDA-13 RC sdist; see docker/Dockerfile.uv.cu129) and absent here.

Re-locking after a dependency change
------------------------------------
Edit ``pyproject.toml`` (and the matching apt deb in
``docker/Dockerfile.uv.cu130`` if it's a system lib), then::

    python manage_envs.py lock          # refresh uv.lock
    python manage_envs.py sync <combo>  # validate locally
    git add pyproject.toml uv.lock
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Active extras in the universal lock. Two GPU torch "worlds" + a CPU slice:
#   * cu13.0 / torch 2.11  : vllm, sglang (inference) + fsdp, megatron (training)
#   * cu12.9 / torch 2.9.1 : veomni, nemoautomodel (training)
#   * cpu   / torch 2.11   : GPU-free CI / unit-test / dev-sanity slice
# trtllm is deferred in pyproject.toml (a CUDA-13 RC) and not listed here.
INFERENCE_BACKENDS: list[str] = ["vllm", "sglang"]
TRAINING_BACKENDS: list[str] = ["fsdp", "megatron"]
# cu12.9 / torch 2.9.1 training backends (their own torch world).
CU129_BACKENDS: list[str] = ["veomni", "nemoautomodel"]
# `cpu` is the GPU-free CI / unit-test / dev-sanity slice.
DEV_BACKENDS: list[str] = ["cpu"]
ALL_EXTRAS: list[str] = INFERENCE_BACKENDS + TRAINING_BACKENDS + CU129_BACKENDS + DEV_BACKENDS

# Mutually exclusive extras — must mirror [tool.uv].conflicts in pyproject.toml.
# At most one member of each set may be synced into a single .venv. The three
# torch worlds never mix, so the cu12.9 backends join every set: vllm / sglang
# are competing inference engines; cpu is GPU-free; veomni / nemoautomodel pin a
# different torch. fsdp / megatron may co-exist with one cu130 inference engine.
# (The flash-attn-* URL-routing sub-extras also conflict in pyproject.toml, but
# they are internal and never synced directly, so they are omitted here.)
CONFLICT_SETS: list[set[str]] = [
    {"vllm", "sglang", "cpu", "veomni", "nemoautomodel"},
    {"fsdp", "cpu", "veomni", "nemoautomodel"},
    {"megatron", "cpu", "veomni", "nemoautomodel"},
]

# Single interpreter for the whole project (matches [tool.uv].environments,
# which pins python_full_version >= '3.12').
PYTHON_VERSION = "3.12"

# Recommended build-time env vars, keyed by extra. `megatron` source-builds
# apex + TE v2.15; the cu12.9 backends (veomni / nemoautomodel) source-build
# flash-attn 2.8.3 against torch 2.9.1 (there is no cu129 prebuilt wheel — uv
# can't fork a second direct URL, uv#13073). Every other extra is prebuilt
# wheels only (cu130 flash-attn, torch / vllm / sglang from the pytorch / PyPI
# indexes). MAX_JOBS=32 is the floor that succeeds everywhere; bump via e.g.
# `MAX_JOBS=128 python manage_envs.py sync megatron` on big hosts.
BUILD_ENV: dict[str, dict[str, str]] = {
    "megatron": {
        "MAX_JOBS": "32",
        "NVTE_FRAMEWORK": "pytorch",
        "NVTE_BUILD_THREADS_PER_JOB": "4",
    },
    # flash-attn source build honors MAX_JOBS (heavy nvcc compile).
    "veomni": {"MAX_JOBS": "32"},
    "nemoautomodel": {"MAX_JOBS": "32"},
}

GROUPS: dict[str, list[str]] = {
    "all": ALL_EXTRAS,
    "inference": INFERENCE_BACKENDS,
    "training": TRAINING_BACKENDS,
    "dev": DEV_BACKENDS,
    # CUDA-world shortcuts, used to scope `prefetch` per Docker image so each
    # image bakes only the backends it can actually run on its CUDA base.
    "cu130": INFERENCE_BACKENDS + TRAINING_BACKENDS,  # torch 2.11 GPU backends
    "cu129": CU129_BACKENDS,                          # torch 2.9.1 GPU backends
}

VERL_DIR = Path(__file__).resolve().parent
VENV_DIR = (VERL_DIR / ".venv").resolve()


def _require_uv() -> None:
    if shutil.which("uv") is None:
        sys.exit("error: uv is not installed. Get it from https://astral.sh/uv")


def _expand(items: list[str]) -> list[str]:
    """Expand group shortcuts (all/inference/training/dev) to extras, dedup,
    preserve order."""
    out: list[str] = []
    for item in items:
        if item in GROUPS:
            out.extend(GROUPS[item])
        elif item in ALL_EXTRAS:
            out.append(item)
        else:
            sys.exit(
                f"error: unknown extra {item!r}. valid: {', '.join(ALL_EXTRAS)} "
                "(or shortcuts: all, inference, training, dev)"
            )
    seen: set[str] = set()
    return [e for e in out if not (e in seen or seen.add(e))]


def _conflict_in(extras: list[str]) -> tuple[str, str] | None:
    """Return the first conflicting pair found in ``extras`` (or None)."""
    chosen = set(extras)
    for cs in CONFLICT_SETS:
        clash = sorted(chosen & cs)
        if len(clash) > 1:
            return clash[0], clash[1]
    return None


def _validate_combo(extras: list[str]) -> None:
    clash = _conflict_in(extras)
    if clash:
        sys.exit(
            f"error: extras {clash[0]!r} and {clash[1]!r} are mutually exclusive "
            "(see [tool.uv].conflicts in pyproject.toml). A single .venv can hold "
            "at most one of each conflict set:\n  "
            + "\n  ".join("{" + ", ".join(sorted(cs)) + "}" for cs in CONFLICT_SETS)
        )


def _extra_flags(extras: list[str]) -> list[str]:
    flags: list[str] = []
    for e in extras:
        flags += ["--extra", e]
    return flags


def _build_env(extras: list[str]) -> dict[str, str]:
    """Merge recommended build-time env vars for the selected extras."""
    env: dict[str, str] = {}
    for e in extras:
        env.update(BUILD_ENV.get(e, {}))
    return env


def _no_install_packages() -> list[str]:
    """Distribution names the operator wants uv to leave alone, parsed from the
    ``VERL_UV_NO_INSTALL`` env var (space/comma-separated). Empty by default."""
    return os.environ.get("VERL_UV_NO_INSTALL", "").replace(",", " ").split()


def _no_install_flags() -> list[str]:
    """``uv sync`` flags that keep ``_no_install_packages()`` untouched: skip
    (re)installing each one (``--no-install-package``) and don't remove an
    already-installed build (``--inexact``, since otherwise an exact sync would
    delete it as extraneous). Empty when nothing is configured, so the default
    exact sync is preserved. Install the internal builds yourself afterwards,
    e.g. ``uv pip install <wheel>``."""
    pkgs = _no_install_packages()
    if not pkgs:
        return []
    flags = ["--inexact"]
    for pkg in pkgs:
        flags += ["--no-install-package", pkg]
    return flags


def _run(cmd: list[str], env_overrides: dict[str, str | None] | None = None) -> int:
    """Run ``cmd`` in VERL_DIR. ``env_overrides`` values that are ``None``
    *unset* the variable (used to drop a stale ``VIRTUAL_ENV`` so uv doesn't
    warn when we target a throwaway env)."""
    if env_overrides:
        shown = " ".join(f"{k}={'' if v is None else v}" for k, v in env_overrides.items())
        print(f"$ {shown} {' '.join(cmd)}", flush=True)
    else:
        print(f"$ {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    for k, v in (env_overrides or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return subprocess.run(cmd, env=env, cwd=str(VERL_DIR)).returncode


def _covering_combos(extras: list[str] | None = None) -> list[list[str]]:
    """Pack ``extras`` into the fewest conflict-free combinations.

    The returned combos together touch every extra in ``extras`` (default
    ``ALL_EXTRAS``) while each one is valid for a single ``uv sync`` (no
    internal conflict). Syncing them in turn therefore fetches / builds every
    distribution those extras need in ``uv.lock`` into the shared uv cache.
    Greedy first-fit stays correct if the conflict topology in
    ``CONFLICT_SETS`` changes (over ALL_EXTRAS it yields
    ``[[vllm, fsdp, megatron], [sglang], [veomni], [nemoautomodel], [cpu]]``;
    over ``cu130`` it yields ``[[vllm, fsdp, megatron], [sglang]]``).
    """
    combos: list[list[str]] = []
    for extra in extras if extras is not None else ALL_EXTRAS:
        for combo in combos:
            if _conflict_in([*combo, extra]) is None:
                combo.append(extra)
                break
        else:
            combos.append([extra])
    return combos


def cmd_lock(args: argparse.Namespace) -> int:
    """Regenerate the universal ``uv.lock`` (``uv lock``)."""
    _require_uv()
    return _run(["uv", "lock", "--python", PYTHON_VERSION, *args.uv_args])


def cmd_sync(args: argparse.Namespace) -> int:
    """Materialize ``.venv`` for one conflict-free extra combination.

    Runs ``uv sync --extra <a> --extra <b> ...`` against the universal
    ``uv.lock``, installing exactly that combination's slice into the project
    venv plus verl itself (editable). This is the **runtime** command — the
    resulting ``.venv`` is what you train / serve from. Honors
    ``VERL_UV_NO_INSTALL`` (see module docstring) to leave internal builds of
    the listed packages alone.
    """
    _require_uv()
    rest = list(args.rest)
    if "--" in rest:
        sep = rest.index("--")
        backends, uv_args = rest[:sep], rest[sep + 1:]
    else:
        backends, uv_args = rest, []
    if not backends:
        sys.exit("error: usage: manage_envs.py sync <extras...> [-- uv args...]")
    extras = _expand(backends)
    _validate_combo(extras)
    cmd = ["uv", "sync", "--python", PYTHON_VERSION, *_extra_flags(extras), *_no_install_flags(), *uv_args]
    rc = _run(cmd, _build_env(extras) or None)
    if rc == 0:
        print(f"\n.venv ready ({', '.join(extras)}). Activate: source .venv/bin/activate", flush=True)
    return rc


def cmd_run(args: argparse.Namespace) -> int:
    """``uv run --extra ... -- <command>`` for a conflict-free combination.

    Everything after ``--`` is the command. uv ensures ``.venv`` matches the
    requested extras (syncing if needed), then runs the command inside it. When
    ``VERL_UV_NO_INSTALL`` is set, ``--no-sync`` is added so uv runs in your
    already-prepared ``.venv`` without clobbering those internal builds (``uv
    run`` has no ``--no-install-package``), so ``sync`` + ``uv pip install``
    them first.
    """
    _require_uv()
    rest = list(args.rest)
    if "--" not in rest:
        sys.exit("error: usage: manage_envs.py run <extras...> -- <command...>")
    sep = rest.index("--")
    extras = _expand(rest[:sep])
    command = rest[sep + 1:]
    if not extras:
        sys.exit("error: no extras given before `--`")
    if not command:
        sys.exit("error: nothing to run after `--`")
    _validate_combo(extras)
    cmd = ["uv", "run", "--python", PYTHON_VERSION, *_extra_flags(extras)]
    if _no_install_packages():
        cmd.append("--no-sync")
    cmd += ["--", *command]
    return _run(cmd, _build_env(extras) or None)


def cmd_shell(args: argparse.Namespace) -> int:
    """Sync the requested combination, then spawn a shell with ``.venv`` active."""
    _require_uv()
    extras = _expand(args.backends)
    _validate_combo(extras)
    rc = _run(
        ["uv", "sync", "--python", PYTHON_VERSION, *_extra_flags(extras), *_no_install_flags()],
        _build_env(extras) or None,
    )
    if rc:
        return rc
    shell = os.environ.get("SHELL", "/bin/bash")
    print(f"spawning {shell} inside .venv [{', '.join(extras)}] (exit to leave)", flush=True)
    env = {
        **os.environ,
        "VIRTUAL_ENV": str(VENV_DIR),
        "PATH": f"{VENV_DIR / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "PS1": f"(verl:{'+'.join(extras)}) {os.environ.get('PS1', '$ ')}",
    }
    return subprocess.run([shell], env=env).returncode


def cmd_clean(_args: argparse.Namespace) -> int:
    """Remove the project ``.venv`` (and the legacy ``.venvs/`` dir if present)."""
    removed = False
    for path in (VENV_DIR, VERL_DIR / ".venvs"):
        if path.exists():
            shutil.rmtree(path)
            print(f"removed {path}")
            removed = True
    if not removed:
        print("(no-op) no .venv to remove")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    """Show available extras, conflict rules, .venv state, and prefetch plan."""
    print("extras (one universal uv.lock; three torch worlds):")
    print(f"  inference (cu130/torch-2.11) : {', '.join(INFERENCE_BACKENDS)}")
    print(f"  training  (cu130/torch-2.11) : {', '.join(TRAINING_BACKENDS)}")
    print(f"  cu129     (torch-2.9.1)      : {', '.join(CU129_BACKENDS)}")
    print(f"  dev       (cpu/torch-2.11)   : {', '.join(DEV_BACKENDS)}")
    print("\nmutually exclusive (at most one per `sync`):")
    for cs in CONFLICT_SETS:
        print("  {" + ", ".join(sorted(cs)) + "}")

    print("\nproject .venv:")
    py = VENV_DIR / "bin" / "python"
    if py.exists():
        try:
            ver = subprocess.check_output(
                [str(py), "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            ver = "?"
        print(f"  present (python {ver}) at {VENV_DIR}")
    else:
        print(f"  absent — run `python manage_envs.py sync <extras...>` to create {VENV_DIR}")

    combos = _covering_combos()
    print("\nprefetch plan (cache-warm combos covering every extra):")
    for combo in combos:
        print(f"  uv sync --extra {' --extra '.join(combo)}")
    print("  (scope per Docker image: `prefetch cu130` | `prefetch cu129`)")
    return 0


def cmd_prefetch(args: argparse.Namespace) -> int:
    """Warm the uv cache with backend packages — first-time / Docker only.

    NOT a runtime sync. Warms the cache for the requested extras (positional
    args; default all of ``ALL_EXTRAS``, or a CUDA world via the ``cu130`` /
    ``cu129`` shortcuts to bake one Docker image's backends). Because the
    backends conflict, there is no single env that holds them all, so this syncs
    the conflict-free covering combos from ``_covering_combos()`` into
    **throwaway** environments purely to download and build their third-party
    distributions into the uv cache (``$UV_CACHE_DIR``, default ``~/.cache/uv``).
    It passes ``--no-install-project`` so only dependencies are fetched/built —
    verl itself is skipped (it is local source, rebuilt cheaply by the real
    ``sync``), which keeps this step independent of the verl source tree. The
    project ``.venv`` is never created or touched.

    Use it once after cloning, or in a Docker layer so the cache ships *inside*
    the image: bake it as a real layer (no ``--mount=type=cache``) and a later
    ``sync <extras...>`` resolves from it offline. For an actual runtime env use
    ``sync <extras...>``.
    """
    _require_uv()
    # rest = [extras/groups...] [-- uv args...]; split on the first `--` so the
    # optional scope args don't collide with forwarded uv flags (argparse can't
    # cleanly separate a `nargs="*"` positional from a REMAINDER across `--`).
    rest = list(args.rest)
    if "--" in rest:
        sep = rest.index("--")
        scope_args, uv_args = rest[:sep], rest[sep + 1:]
    else:
        scope_args, uv_args = rest, []
    requested = _expand(scope_args) if scope_args else ALL_EXTRAS
    combos = _covering_combos(requested)
    scope = "every extra" if not scope_args else ", ".join(requested)
    print(f"prefetch: warming the uv cache for {scope} (throwaway envs; .venv untouched).")
    print("covering combos:")
    for combo in combos:
        print(f"  {' + '.join(combo)}")
    print()

    with tempfile.TemporaryDirectory(prefix="verl-prefetch-") as tmp:
        # Route uv at a throwaway env and drop any stale VIRTUAL_ENV so uv
        # doesn't warn about a mismatch with the active shell venv. Only the
        # shared uv cache (UV_CACHE_DIR) is the durable output here.
        proj_env: dict[str, str | None] = {
            "UV_PROJECT_ENVIRONMENT": str(Path(tmp) / ".venv"),
            "VIRTUAL_ENV": None,
        }
        for combo in combos:
            # --no-install-project: cache deps only; skip building verl (local
            # source) so the cache layer doesn't depend on the verl tree.
            cmd = [
                "uv", "sync", "--python", PYTHON_VERSION,
                "--no-install-project", *_extra_flags(combo), *uv_args,
            ]
            rc = _run(cmd, {**proj_env, **_build_env(combo)})
            if rc:
                print(f"error: prefetch failed while warming {combo}", file=sys.stderr)
                return rc
    print(f"\nuv cache warmed for: {scope}.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_envs.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extras_help = (
        "extra name(s); shortcuts: all, inference (vllm sglang), "
        "training (fsdp megatron), cu129 (veomni nemoautomodel), dev (cpu). "
        "x86_64 Linux + Python 3.12 only. Mutually exclusive sets (at most one "
        "each per sync): {vllm, sglang, cpu, veomni, nemoautomodel}, "
        "{fsdp, cpu, veomni, nemoautomodel}, {megatron, cpu, veomni, "
        "nemoautomodel}. trtllm is deferred (a CUDA-13 RC; see pyproject.toml)."
    )

    lk = sub.add_parser("lock", help="(re)generate the universal uv.lock (uv lock)")
    lk.add_argument(
        "uv_args",
        nargs=argparse.REMAINDER,
        help="anything after `--` is forwarded to `uv lock`",
    )
    lk.set_defaults(func=cmd_lock)

    s = sub.add_parser(
        "sync",
        help="materialize .venv for one conflict-free extra combination (runtime)",
    )
    s.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="<extras...> [-- uv args...]: anything after `--` is forwarded to "
        "`uv sync`. " + extras_help,
    )
    s.set_defaults(func=cmd_sync)

    r = sub.add_parser(
        "run",
        help="uv run --extra ... -- <command> for a conflict-free combination",
    )
    r.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="<extras...> -- <command...>",
    )
    r.set_defaults(func=cmd_run)

    sh = sub.add_parser("shell", help="sync a combination then open a shell with .venv active")
    sh.add_argument("backends", nargs="+", help=extras_help)
    sh.set_defaults(func=cmd_shell)

    sub.add_parser("list", help="show extras, conflict rules, .venv state, prefetch plan").set_defaults(
        func=cmd_list
    )

    sub.add_parser("clean", help="remove the project .venv").set_defaults(func=cmd_clean)

    pf = sub.add_parser(
        "prefetch",
        help="FIRST-TIME / Docker only: warm the uv cache with backend deps (NOT a runtime sync)",
        description=(
            "Download & build backend dependencies into the uv cache "
            "($UV_CACHE_DIR, default ~/.cache/uv) by syncing conflict-free "
            "covering combos into throwaway envs (with --no-install-project). "
            "Pass extras/groups to scope it (default: all); use the cu130 / "
            "cu129 shortcuts to bake one CUDA world per Docker image. The "
            "project .venv is never created or modified. Use once after "
            "cloning, or bake it into a Docker image layer; for a runtime env "
            "use `sync`."
        ),
    )
    pf.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="[extras/groups...] [-- uv args...]: optional extras/groups scope "
        "the cache warm (default: all; e.g. `cu130` or `cu129` to bake one "
        "CUDA world); anything after `--` is forwarded to each `uv sync`",
    )
    pf.set_defaults(func=cmd_prefetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if hasattr(args, "uv_args") and args.uv_args and args.uv_args[0] == "--":
        args.uv_args = args.uv_args[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

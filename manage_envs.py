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
    prefetch             # FIRST-TIME / Docker only: generate uv.lock, then
                         #   warm the uv cache with every backend's wheels &
                         #   source builds (see below)

Anything after ``--`` on ``lock`` / ``sync`` / ``prefetch`` is forwarded to
the underlying ``uv`` invocation, e.g.::

    python manage_envs.py sync megatron -- --reinstall -v

Naming distinct venvs
---------------------
By default everything lands in the project venv ``.venv``. Pass ``--name NAME``
(or set ``VERL_VENV_NAME``) to put the real env in ``.venv-<NAME>`` instead — uv
is pointed at it via ``UV_PROJECT_ENVIRONMENT`` — so several backend combos,
users, or runs can coexist on one checkout without clobbering a single shared
venv::

    python manage_envs.py sync --name vllm-mega vllm megatron   # -> .venv-vllm-mega
    python manage_envs.py shell --name sglang-fsdp sglang fsdp  # -> .venv-sglang-fsdp

After a named ``sync`` / ``shell`` (or a syncing ``run``) the conventional
``.venv`` is (re)created as a **symlink to the composition just synced**, so it
always points at the latest env and ``source .venv/bin/activate``, CI, and the
byted wrappers keep working unchanged::

    python manage_envs.py sync --name vllm-mega vllm megatron   # .venv -> .venv-vllm-mega
    python manage_envs.py sync --name sglang-fsdp sglang fsdp   # .venv -> .venv-sglang-fsdp
    source .venv/bin/activate                                   # = the sglang-fsdp env

A nameless ``sync`` reclaims ``.venv`` as a real directory (dropping the pointer
first, so it never writes through the link into a named env). ``clean`` unlinks
the pointer instead of deleting through it, and drops it when it would dangle.
``sync`` / ``run`` / ``shell`` / ``clean`` / ``list`` all accept the same
``--name`` (or ``VERL_VENV_NAME``); for ``sync`` / ``run`` put it before the
extras. An explicit ``UV_PROJECT_ENVIRONMENT`` is honored as-is when no name is
given. ``prefetch`` is unaffected — it only warms the cache via throwaway envs.

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
**first-time / image-build** helper: it first resolves the universal
``uv.lock`` from ``pyproject.toml`` (``uv lock``), then downloads & builds
*every* backend's dependencies into the uv cache (``$UV_CACHE_DIR``, default
``~/.cache/uv``) so later ``sync`` runs are fast/offline. Because the backends
conflict, ``prefetch`` cannot produce a single usable env — it syncs throwaway
envs purely to populate the cache (passing ``--no-install-project`` so only
dependencies are cached, not verl itself) and never creates or modifies
``.venv``. The lock it produces is what every later ``sync`` consumes.

In Docker this is what makes one image serve any backend: bake ``prefetch``
as a real layer (point ``UV_CACHE_DIR`` at an in-image path and do **not** use
a ``--mount=type=cache``) so the cache ships *inside* the image. The container
then picks its combination at run time — ``sync <extras...>`` builds ``.venv``
from the baked cache, offline — instead of hard-coding one combo at build
time. Do **not** use ``prefetch`` as a runtime sync; use ``sync <extras...>``
for that.

This driver exposes one GPU torch "world" plus a CPU slice, all in one lock:
the cu13.0 / torch-2.11 backends (vllm, sglang, fsdp, megatron) and the
GPU-free ``cpu`` slice. They never mix in one ``.venv`` (see the conflict
sets). ``prefetch`` scopes the cache warm via the ``cu130`` shortcut so the
Docker image bakes only its backends. DEFERRED (commented out in
pyproject.toml until they support torch-2.11 / cu130): the cu12.9 /
torch-2.9.1 world (veomni, nemoautomodel) and trtllm (a CUDA-13 RC sdist).

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
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Active extras in the universal lock. One GPU torch "world" + a CPU slice:
#   * cu13.0 / torch 2.11  : vllm, sglang (inference) + fsdp, megatron (training)
#   * cpu   / torch 2.11   : GPU-free CI / unit-test / dev-sanity slice
# DEFERRED and absent from the active lock: the cu12.9 / torch-2.9.1 world
# (veomni, nemoautomodel) and trtllm (a CUDA-13 RC). Both stay commented out in
# pyproject.toml until those packages support torch-2.11 / cu130.
INFERENCE_BACKENDS: list[str] = ["vllm", "sglang"]
TRAINING_BACKENDS: list[str] = ["fsdp", "megatron"]
# DEFERRED — cu12.9 / torch 2.9.1 training backends (["veomni", "nemoautomodel"]).
# Re-add the names here and re-enable their extras in pyproject.toml when they
# support torch-2.11 / cu130.
CU129_BACKENDS: list[str] = []
# `cpu` is the GPU-free CI / unit-test / dev-sanity slice.
DEV_BACKENDS: list[str] = ["cpu"]
ALL_EXTRAS: list[str] = INFERENCE_BACKENDS + TRAINING_BACKENDS + CU129_BACKENDS + DEV_BACKENDS

# Mutually exclusive extras — must mirror [tool.uv].conflicts in pyproject.toml.
# At most one member of each set may be synced into a single .venv. vllm / sglang
# are competing inference engines; cpu is GPU-free. fsdp / megatron may co-exist
# with one cu130 inference engine. (The flash-attn-* URL-routing sub-extras also
# conflict in pyproject.toml, but they are internal and never synced directly,
# so they are omitted here.)
# DEFERRED (cu12.9): re-add "veomni", "nemoautomodel" to every set when the
# torch-2.9.1 world returns — they pin a different torch, so they conflict with all.
CONFLICT_SETS: list[set[str]] = [
    {"vllm", "sglang", "cpu"},
    {"fsdp", "cpu"},
    {"megatron", "cpu"},
]

# Single interpreter for the whole project (matches [tool.uv].environments,
# which pins python_full_version >= '3.12').
PYTHON_VERSION = "3.12"

GROUPS: dict[str, list[str]] = {
    "all": ALL_EXTRAS,
    "inference": INFERENCE_BACKENDS,
    "training": TRAINING_BACKENDS,
    "dev": DEV_BACKENDS,
    # CUDA-world shortcuts, used to scope `prefetch` per Docker image so each
    # image bakes only the backends it can actually run on its CUDA base.
    "cu130": INFERENCE_BACKENDS + TRAINING_BACKENDS,  # torch 2.11 GPU backends
    # DEFERRED (cu12.9): "cu129": CU129_BACKENDS,  # torch 2.9.1 GPU backends
}

VERL_DIR = Path(__file__).resolve().parent
# The conventional project venv path. A user-offered name (``--name`` /
# ``VERL_VENV_NAME``) puts the real env in ``.venv-<name>`` so distinct backend
# combos / users / runs coexist on one checkout, and ``.venv`` is maintained as a
# symlink to the latest such composition (see ``_point_default_venv``) so
# ``source .venv/bin/activate`` and tooling always reach it. uv is pointed at the
# chosen directory through ``UV_PROJECT_ENVIRONMENT`` (see ``_venv_dir``).
# NB: not ``.resolve()``-d — that would follow the ``.venv`` symlink to its
# target, and a nameless ``sync`` must operate on ``.venv`` itself.
DEFAULT_VENV_DIR = VERL_DIR / ".venv"

# A name is used verbatim as the ``.venv-<name>`` directory suffix, so keep it to
# filesystem-safe characters (no path separators, no leading dot / dash).
_VENV_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _resolve_name(cli_name: str | None) -> str | None:
    """The user-offered suffix for the project venv directory.

    ``--name`` / ``-n`` wins; otherwise the ``VERL_VENV_NAME`` env var. Returns
    ``None`` when neither is set (the default ``.venv``). The value is validated
    because it becomes a directory-name suffix.
    """
    name = cli_name if cli_name is not None else os.environ.get("VERL_VENV_NAME")
    if name is None:
        return None
    name = name.strip()
    if not name:
        return None
    if not _VENV_NAME_RE.match(name):
        sys.exit(
            f"error: invalid venv name {name!r}: use letters, digits, '.', '_' or '-' "
            "(it becomes the '.venv-<name>' directory suffix)"
        )
    return name


def _venv_dir(name: str | None) -> Path:
    """Resolve the project venv directory for this invocation.

    Priority: a user-offered ``name`` -> ``.venv-<name>``; else an explicit
    ``UV_PROJECT_ENVIRONMENT`` (uv's own override) used as-is; else the default
    ``.venv``. The result is what every command hands back to uv via
    ``UV_PROJECT_ENVIRONMENT`` so sync / run / shell / clean / list all agree on
    which env they target. Paths under the checkout are left un-resolved so we
    never follow the ``.venv`` pointer symlink to a named env.
    """
    if name:
        return VERL_DIR / f".venv-{name}"
    env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_VENV_DIR


def _proj_env(venv_dir: Path) -> dict[str, str | None]:
    """Env override routing uv at ``venv_dir`` (``UV_PROJECT_ENVIRONMENT``)."""
    return {"UV_PROJECT_ENVIRONMENT": str(venv_dir)}


def _detach_symlink_target(venv_dir: Path) -> None:
    """Drop ``venv_dir`` if it is currently a symlink, so a subsequent ``uv sync``
    builds a real directory there instead of writing THROUGH the link into the
    named env it points at.

    Only the conventional ``.venv`` is ever a symlink (the latest-composition
    pointer), so this fires when a nameless ``sync`` / ``shell`` follows an
    earlier named one: the pointer is consumed and ``.venv`` becomes a real env.
    """
    if venv_dir.is_symlink():
        prev = os.readlink(venv_dir)
        venv_dir.unlink()
        print(f"note: {venv_dir} was a symlink (-> {prev}); replacing it with a real env", flush=True)


def _point_default_venv(venv_dir: Path) -> None:
    """Point the conventional ``.venv`` at ``venv_dir`` — the composition just
    synced — so ``source .venv/bin/activate``, CI, and the byted wrappers reach
    the latest env even when it lives in ``.venv-<name>``.

    No-op when ``venv_dir`` already is ``.venv`` or was not materialized. An
    existing ``.venv`` *symlink* is repointed; a real ``.venv`` *directory* is
    left untouched (with a warning) so a materialized env is never deleted.
    """
    link = DEFAULT_VENV_DIR
    if venv_dir == link or not venv_dir.exists():
        return
    if link.is_symlink():
        try:
            if link.resolve() == venv_dir.resolve():
                return  # already points here; stay quiet on repeat syncs
        except OSError:
            pass
        link.unlink()
    elif link.exists():
        print(
            f"warning: {link} is a real directory; not repointing it at {venv_dir.name}. "
            f"Activate {venv_dir}/bin/activate directly, or `python manage_envs.py clean` "
            "to replace it with a link.",
            file=sys.stderr,
        )
        return
    # Relative link when the env sits beside ``.venv`` (portable if the checkout
    # moves); absolute for an external UV_PROJECT_ENVIRONMENT.
    rel = os.path.relpath(venv_dir, VERL_DIR)
    target = rel if not rel.startswith("..") else str(venv_dir)
    link.symlink_to(target, target_is_directory=True)
    print(f"pointed {link} -> {target} (latest composition)", flush=True)


def _pop_name(tokens: list[str]) -> tuple[str | None, list[str]]:
    """Pull a ``--name`` / ``-n`` (or ``--name=…``) value out of ``tokens``.

    argparse only binds ``--name`` when it precedes the ``REMAINDER`` positional,
    so ``sync vllm --name foo`` strands it in ``rest``; this recovers it. Scanning
    stops at the first ``--`` so a passthrough command / uv args are never touched.
    Returns ``(name, tokens_without_the_flag)``.
    """
    name: str | None = None
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "--":
            out.extend(tokens[i:])
            break
        if tok in ("--name", "-n"):
            if i + 1 >= n or tokens[i + 1] == "--":
                sys.exit("error: --name/-n requires a value")
            name, i = tokens[i + 1], i + 2
            continue
        if tok.startswith("--name="):
            name, i = tok[len("--name=") :], i + 1
            continue
        out.append(tok)
        i += 1
    return name, out


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
    """Cache-warm combos: one inference engine + one training backend each.

    A real RL run syncs exactly ONE inference engine and ONE training backend
    (e.g. ``sync vllm megatron``), so the combos that matter are the
    ``INFERENCE_BACKENDS x TRAINING_BACKENDS`` cross-product. We deliberately do
    NOT bundle both trainers into one maximal env: ``fsdp + megatron`` is never
    a real runtime selection, and because ``megatron`` is conflict-gated, the
    ``{vllm, fsdp, megatron}`` resolution fork it would warm is one nobody syncs
    — while the ``{vllm, fsdp}`` / ``{vllm, megatron}`` forks people actually
    use would go uncached and miss offline. Pairing each inference engine with
    each trainer warms exactly those runtime forks.

    Backends with no counterpart role are warmed standalone: the GPU-free
    ``cpu`` slice, mutually exclusive with everything else. The same fallback
    emits singletons when a scope contains only one cu130 role (e.g.
    ``prefetch inference``). (DEFERRED: the cu12.9 trainers veomni /
    nemoautomodel were also warmed standalone; they return with that world.)

    Over ``cu130`` it yields ``[[vllm, fsdp], [vllm, megatron], [sglang, fsdp],
    [sglang, megatron]]``; over ``ALL_EXTRAS`` it appends ``[cpu]``.
    (Inference-only or training-only runs are not pre-warmed — sync one
    inference engine + one trainer, per the RL flow.)
    """
    pool = list(extras) if extras is not None else list(ALL_EXTRAS)
    pset = set(pool)
    inference = [e for e in INFERENCE_BACKENDS if e in pset]
    training = [e for e in TRAINING_BACKENDS if e in pset]
    combos: list[list[str]] = []
    paired: set[str] = set()
    if inference and training:
        # cu130 RL pairs; inference and trainers never share a conflict set,
        # so every pair is conflict-free (guard keeps that true if it changes).
        for inf in inference:
            for tr in training:
                if _conflict_in([inf, tr]) is None:
                    combos.append([inf, tr])
                    paired.update((inf, tr))
    # Whatever the cross-product didn't consume is warmed on its own: the
    # cu12.9 trainers, the cpu slice, or a lone cu130 role with no partner.
    combos += [[e] for e in pool if e not in paired]
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
        backends, uv_args = rest[:sep], rest[sep + 1 :]
    else:
        backends, uv_args = rest, []
    stray_name, backends = _pop_name(backends)
    name = _resolve_name(args.name if args.name is not None else stray_name)
    if not backends:
        sys.exit("error: usage: manage_envs.py sync [--name NAME] <extras...> [-- uv args...]")
    extras = _expand(backends)
    _validate_combo(extras)
    venv_dir = _venv_dir(name)
    _detach_symlink_target(venv_dir)
    cmd = [
        "uv",
        "sync",
        "--python",
        PYTHON_VERSION,
        *_extra_flags(extras),
        *_no_install_flags(),
        *uv_args,
    ]
    rc = _run(cmd, _proj_env(venv_dir))
    if rc == 0:
        _point_default_venv(venv_dir)
        print(
            f"\n.venv ready ({', '.join(extras)}) at {venv_dir}. Activate: source {venv_dir}/bin/activate",
            flush=True,
        )
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
        sys.exit("error: usage: manage_envs.py run [--name NAME] <extras...> -- <command...>")
    sep = rest.index("--")
    stray_name, pre = _pop_name(rest[:sep])
    name = _resolve_name(args.name if args.name is not None else stray_name)
    extras = _expand(pre)
    command = rest[sep + 1 :]
    if not extras:
        sys.exit("error: no extras given before `--`")
    if not command:
        sys.exit("error: nothing to run after `--`")
    _validate_combo(extras)
    venv_dir = _venv_dir(name)
    cmd = ["uv", "run", "--python", PYTHON_VERSION, *_extra_flags(extras)]
    # With VERL_UV_NO_INSTALL, `uv run` gets --no-sync: it must use the env as-is
    # (including a `.venv` symlink to a curated combo), so don't detach/repoint.
    syncs = not _no_install_packages()
    if not syncs:
        cmd.append("--no-sync")
    elif venv_dir == DEFAULT_VENV_DIR:
        _detach_symlink_target(venv_dir)
    cmd += ["--", *command]
    rc = _run(cmd, _proj_env(venv_dir))
    if rc == 0 and syncs:
        _point_default_venv(venv_dir)
    return rc


def cmd_shell(args: argparse.Namespace) -> int:
    """Sync the requested combination, then spawn a shell with ``.venv`` active."""
    _require_uv()
    stray_name, backends = _pop_name(list(args.backends))
    name = _resolve_name(args.name if args.name is not None else stray_name)
    extras = _expand(backends)
    _validate_combo(extras)
    venv_dir = _venv_dir(name)
    _detach_symlink_target(venv_dir)
    rc = _run(
        ["uv", "sync", "--python", PYTHON_VERSION, *_extra_flags(extras), *_no_install_flags()],
        _proj_env(venv_dir),
    )
    if rc:
        return rc
    _point_default_venv(venv_dir)
    shell = os.environ.get("SHELL", "/bin/bash")
    print(f"spawning {shell} inside {venv_dir} [{', '.join(extras)}] (exit to leave)", flush=True)
    env = {
        **os.environ,
        "VIRTUAL_ENV": str(venv_dir),
        "UV_PROJECT_ENVIRONMENT": str(venv_dir),
        "PATH": f"{venv_dir / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "PS1": f"(verl:{'+'.join(extras)}) {os.environ.get('PS1', '$ ')}",
    }
    return subprocess.run([shell], env=env).returncode


def cmd_clean(args: argparse.Namespace) -> int:
    """Remove the resolved project venv (``.venv`` or ``.venv-<name>``), then the
    ``.venv`` pointer if it now dangles, plus the legacy ``.venvs/`` dir."""
    venv_dir = _venv_dir(_resolve_name(args.name))
    removed = False
    # Remove the target env. A bare ``.venv`` may be the latest-composition
    # symlink — unlink it rather than rmtree'ing THROUGH to the real named env.
    if venv_dir.is_symlink():
        venv_dir.unlink()
        print(f"removed symlink {venv_dir}")
        removed = True
    elif venv_dir.exists():
        shutil.rmtree(venv_dir)
        print(f"removed {venv_dir}")
        removed = True
    # Drop the ``.venv`` pointer if it targeted what we just removed (now dangling).
    link = DEFAULT_VENV_DIR
    if link != venv_dir and link.is_symlink():
        try:
            points_here = link.resolve() == venv_dir.resolve()
        except OSError:
            points_here = True  # broken link -> safe to drop
        if points_here or not link.exists():
            link.unlink()
            print(f"removed dangling symlink {link}")
            removed = True
    legacy = VERL_DIR / ".venvs"
    if legacy.exists():
        shutil.rmtree(legacy)
        print(f"removed {legacy}")
        removed = True
    if not removed:
        print(f"(no-op) no venv to remove at {venv_dir}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Show available extras, conflict rules, venv state, and prefetch plan."""
    print("extras (one universal uv.lock; cu130/torch-2.11 + cpu):")
    print(f"  inference (cu130/torch-2.11) : {', '.join(INFERENCE_BACKENDS)}")
    print(f"  training  (cu130/torch-2.11) : {', '.join(TRAINING_BACKENDS)}")
    print(f"  dev       (cpu/torch-2.11)   : {', '.join(DEV_BACKENDS)}")
    print("  cu129     (torch-2.9.1)      : DEFERRED (veomni, nemoautomodel)")
    print("\nmutually exclusive (at most one per `sync`):")
    for cs in CONFLICT_SETS:
        print("  {" + ", ".join(sorted(cs)) + "}")

    venv_dir = _venv_dir(_resolve_name(args.name))
    target_label = f"{venv_dir} -> {os.readlink(venv_dir)}" if venv_dir.is_symlink() else f"{venv_dir}"
    print(f"\nproject venv (target: {target_label}):")
    py = venv_dir / "bin" / "python"
    if py.exists():
        try:
            ver = subprocess.check_output(
                [str(py), "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            ver = "?"
        print(f"  present (python {ver}) at {venv_dir}")
    else:
        print(f"  absent — run `python manage_envs.py sync <extras...>` to create {venv_dir}")

    # Other venvs already materialized on this checkout (default, named, or the
    # `.venv` pointer), so a user can see / reuse / clean them by name.
    others = sorted(p for p in VERL_DIR.glob(".venv*") if p != venv_dir and (p / "bin" / "python").exists())
    if others:
        print("  other venvs on this checkout (target with --name):")
        for p in others:
            if p.is_symlink():
                note = f"-> {os.readlink(p)} (latest-composition pointer)"
            elif p.name.startswith(".venv-"):
                note = f"[--name {p.name[len('.venv-') :]}]"
            else:
                note = "(default)"
            print(f"    {p}  {note}")

    combos = _covering_combos()
    print("\nprefetch plan (cache-warm combos; 1 inference + 1 training each):")
    for combo in combos:
        print(f"  uv sync --extra {' --extra '.join(combo)}")
    print("  (scope per Docker image: `prefetch cu130`)")
    return 0


def cmd_prefetch(args: argparse.Namespace) -> int:
    """Generate ``uv.lock`` and warm the uv cache — first-time / Docker only.

    Two steps. (1) Resolve the universal ``uv.lock`` from ``pyproject.toml``
    (``uv lock``) so prefetch OWNS the lockfile: pyproject is the single source
    of truth and no pre-existing lock is required (it is a no-op rewrite when
    the lock is already current). (2) Warm the cache for the requested extras
    (positional args; default all of ``ALL_EXTRAS``, or the ``cu130`` shortcut
    to bake the Docker image's backends).
    Because the backends conflict, there is no single env that holds them all,
    so it syncs the conflict-free runtime combos from ``_covering_combos()``
    (one inference engine + one training backend each) into **throwaway**
    environments purely to download and build their third-party distributions
    into the uv cache (``$UV_CACHE_DIR``, default ``~/.cache/uv``), pinned with
    ``--frozen`` to the lock from step 1. ``--no-install-project`` skips verl
    itself (local source, rebuilt cheaply by the real ``sync``), keeping this
    step independent of the verl source tree. The project ``.venv`` is never
    created or touched.

    ``uv lock`` reads only ``pyproject.toml`` + the declared
    ``[tool.uv.dependency-metadata]``, so it triggers NO source build — the
    git-sourced megatron-core / mbridge are compiled in step 2, not here (apex /
    TE / flash-attn / vllm / sglang-kernel ship prebuilt from the wheelhouse).

    Use it once after cloning, or in a Docker layer so both the lock and the
    cache ship *inside* the image: bake it as a real layer (no
    ``--mount=type=cache``) and a later ``sync <extras...>`` resolves from the
    baked ``uv.lock`` + cache offline. For an actual runtime env use
    ``sync <extras...>``.
    """
    _require_uv()
    # rest = [extras/groups...] [-- uv args...]; split on the first `--` so the
    # optional scope args don't collide with forwarded uv flags (argparse can't
    # cleanly separate a `nargs="*"` positional from a REMAINDER across `--`).
    rest = list(args.rest)
    if "--" in rest:
        sep = rest.index("--")
        scope_args, uv_args = rest[:sep], rest[sep + 1 :]
    else:
        scope_args, uv_args = rest, []
    requested = _expand(scope_args) if scope_args else ALL_EXTRAS
    combos = _covering_combos(requested)
    scope = "every extra" if not scope_args else ", ".join(requested)
    print(f"prefetch: generating uv.lock + warming the uv cache for {scope} (throwaway envs; .venv untouched).")
    print("combos (1 inference + 1 training each):")
    for combo in combos:
        print(f"  {' + '.join(combo)}")
    print()

    # 1) Resolve the universal uv.lock from pyproject.toml. prefetch OWNS the
    # lock: rather than requiring a pre-committed lockfile it generates one here
    # so pyproject.toml is the single source of truth. This reads only
    # pyproject.toml + the declared [tool.uv.dependency-metadata] (no source
    # build) and is a no-op rewrite when the lock is already current. `--frozen`
    # is intentionally NOT passed here — it would forbid the resolution — but it
    # does pin the per-combo cache-warming syncs below.
    print("resolving universal uv.lock (uv lock)...", flush=True)
    rc = _run(["uv", "lock", "--python", PYTHON_VERSION], {"VIRTUAL_ENV": None})
    if rc:
        print("error: prefetch failed to resolve uv.lock", file=sys.stderr)
        return rc
    print()

    # A FRESH throwaway env per combo. The combos are mutually exclusive, so
    # reusing one env would force every sync to *uninstall* the previous
    # combo's packages before installing its own (e.g. the `cpu` combo tearing
    # out every CUDA wheel, or torch swapping cu130<->cpu). Syncing each into
    # its own empty env makes every sync a pure *append* — it only installs
    # what that combo needs and never removes anything — which also mirrors
    # exactly what a real runtime `uv sync <combo>` does. Only the shared uv
    # cache (UV_CACHE_DIR) is durable: wheels download once and the git-source
    # builds (megatron-core / mbridge) build once, then later combos hardlink
    # them from the cache instead of rebuilding. Peak disk is one env at a time
    # (each tempdir is torn down before the next).
    for combo in combos:
        with tempfile.TemporaryDirectory(prefix="verl-prefetch-") as tmp:
            env_dir = Path(tmp) / ".venv"
            # Route uv at the throwaway env and drop any stale VIRTUAL_ENV so uv
            # doesn't warn about a mismatch with the active shell venv.
            proj_env: dict[str, str | None] = {
                "UV_PROJECT_ENVIRONMENT": str(env_dir),
                "VIRTUAL_ENV": None,
            }
            # --frozen: pin to the uv.lock resolved in step 1 (no re-resolve).
            # --no-install-project: cache deps only; skip building verl (local
            # source) so the cache layer doesn't depend on the verl tree.
            # Both are hardcoded; uv rejects a flag passed twice, so drop them
            # from any forwarded uv_args (e.g. a now-redundant `-- --frozen`)
            # before appending the rest.
            warm_args = [a for a in uv_args if a not in ("--frozen", "--no-install-project")]
            cmd = [
                "uv",
                "sync",
                "--python",
                PYTHON_VERSION,
                "--frozen",
                "--no-install-project",
                *_extra_flags(combo),
                *warm_args,
            ]
            rc = _run(cmd, proj_env)
            if rc:
                print(f"error: prefetch failed while warming {combo}", file=sys.stderr)
                return rc
    print(f"\nuv.lock generated; uv cache warmed for: {scope}.")
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
        "training (fsdp megatron), dev (cpu). x86_64 Linux + Python 3.12 only. "
        "Mutually exclusive sets (at most one each per sync): "
        "{vllm, sglang, cpu}, {fsdp, cpu}, {megatron, cpu}. DEFERRED (see "
        "pyproject.toml): the cu12.9 world (veomni, nemoautomodel) and trtllm "
        "(a CUDA-13 RC)."
    )
    name_help = (
        "name a distinct project venv: targets '.venv-<NAME>' (via "
        "UV_PROJECT_ENVIRONMENT) instead of the default '.venv', so different "
        "backend combos / users / runs don't clobber one shared venv. Also "
        "settable with the VERL_VENV_NAME env var (this flag wins). For 'sync' / "
        "'run' put it before the extras."
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
        help="materialize .venv (or .venv-<name>) for one conflict-free extra combination (runtime)",
    )
    s.add_argument("--name", "-n", default=None, help=name_help)
    s.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="[--name NAME] <extras...> [-- uv args...]: anything after `--` is forwarded to `uv sync`. " + extras_help,
    )
    s.set_defaults(func=cmd_sync)

    r = sub.add_parser(
        "run",
        help="uv run --extra ... -- <command> for a conflict-free combination",
    )
    r.add_argument("--name", "-n", default=None, help=name_help)
    r.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="[--name NAME] <extras...> -- <command...>",
    )
    r.set_defaults(func=cmd_run)

    sh = sub.add_parser("shell", help="sync a combination then open a shell with .venv (or .venv-<name>) active")
    sh.add_argument("--name", "-n", default=None, help=name_help)
    sh.add_argument("backends", nargs="+", help=extras_help)
    sh.set_defaults(func=cmd_shell)

    ls = sub.add_parser("list", help="show extras, conflict rules, venv state, prefetch plan")
    ls.add_argument("--name", "-n", default=None, help=name_help)
    ls.set_defaults(func=cmd_list)

    cl = sub.add_parser("clean", help="remove the project .venv (or .venv-<name>)")
    cl.add_argument("--name", "-n", default=None, help=name_help)
    cl.set_defaults(func=cmd_clean)

    pf = sub.add_parser(
        "prefetch",
        help="FIRST-TIME / Docker only: generate uv.lock + warm the uv cache with backend deps (NOT a runtime sync)",
        description=(
            "Generate the universal uv.lock from pyproject.toml (uv lock), then "
            "download & build backend dependencies into the uv cache "
            "($UV_CACHE_DIR, default ~/.cache/uv) by syncing the conflict-free "
            "runtime combos (1 inference + 1 training each) into throwaway envs "
            "(--frozen --no-install-project). Pass extras/groups to scope the "
            "cache warm (default: all); use the cu130 shortcut to bake the "
            "cu130 world per Docker image. The project .venv is never created "
            "or modified. Use once after cloning, or bake it into a Docker image "
            "layer; for a runtime env use `sync`."
        ),
    )
    pf.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="[extras/groups...] [-- uv args...]: optional extras/groups scope "
        "the cache warm (default: all; e.g. `cu130` to bake the cu130 world); "
        "anything after `--` is forwarded to each `uv sync`",
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

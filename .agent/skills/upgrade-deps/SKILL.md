---
name: upgrade-deps
description: Upgrade, downgrade, or re-pin a Python dependency (vllm, sglang, torch, transformers, flash-attn, etc.) in verl's universal uv.lock flow.
user_invocable: true
---

When the user asks to upgrade, bump, downgrade, or change the pinned version of a package in verl's `uv` flow, follow these steps.

verl uses **one universal `uv.lock`** for the whole project. Every version is pinned in [`pyproject.toml`](pyproject.toml); the lockfile is generated from it. **Never hand-edit `uv.lock`** — edit the pin in `pyproject.toml` and re-lock. For the full background, read the "Upgrade or modify dependencies" section of [`docs/start/install.rst`](docs/start/install.rst) and the header comment in [`pyproject.toml`](pyproject.toml).

### 1. Locate the version pin

Version pins live in two places in [`pyproject.toml`](pyproject.toml):

- **Per-backend extras** under `[project.optional-dependencies]` — `verl-core` (shared runtime deps) and the backend extras `vllm`, `sglang`, `fsdp`, `megatron`, `cpu`. A package like `vllm==0.20.2` is pinned inside the `vllm` extra.
- **Global overrides** under `[tool.uv].override-dependencies` — `transformers`, `numpy`, and `kernels` are force-pinned project-wide here, overriding whatever the extras / upstreams request.

Grep for the package to find every occurrence; it may appear in several extras (e.g. `torch==2.11.0` is repeated across `vllm`, `sglang`, `fsdp`, `megatron`, `cpu`).

### 2. Edit the pin

Change the version specifier in **every** place it appears.

- If the package is `transformers`, `numpy`, or `kernels`, update it in `[tool.uv].override-dependencies` **as well as** the extras — the override wins, so a stale value there silently defeats the extra.
- Shared deps (`torch`, `transformers`, ...) span multiple extras that must agree because they share one torch "world"; bump them together, not in a single extra.

### 3. Handle coupled pins (critical for `vllm` / `torch` bumps)

Some upgrades ripple into other pins. Check and update as needed:

- **torch world**: `vllm`, `sglang`, `fsdp`, `megatron` all pin the same `torch` / `torchvision` / `torchaudio` and route to the `pytorch-cu130` index (`[tool.uv.sources]`). A vllm/sglang bump that needs a new torch must move all of them in lockstep.
- **flash-attn wheel**: the prebuilt URL in `[tool.uv.sources].flash-attn` is matched to a specific `(CUDA, torch, cp-abi)` tuple. If you change torch / CUDA / Python, swap the URL for a matching wheel from the prebuild releases, or fall back to a source build (add `flash-attn` to `[tool.uv].no-build-isolation-package` and list `flash-attn==<ver>` directly in the extra). Only **one** direct URL is allowed per package across the whole lock (uv#13073).
- **dependency-metadata**: git-source packages (`apex`, `transformer-engine`, `megatron-core`, `mbridge`, `flash-attn`) have hand-written `[[tool.uv.dependency-metadata]]` blocks and `[tool.uv.sources]` revs. If you bump one of these, update both its `version` and its git `tag`/`rev` together.

### 4. Re-lock and validate

Regenerate the single lockfile, then materialize a `.venv` to confirm it resolves and imports:

```bash
# from the verl/ project root, after editing pyproject.toml:
python manage_envs.py lock                 # or: uv lock
python manage_envs.py sync vllm fsdp       # sync a relevant conflict-free combo
python manage_envs.py run vllm fsdp -- python -c "import vllm; print(vllm.__version__)"
```

Pick a `sync` combo that exercises the changed package. Mutually exclusive extras (see `[tool.uv].conflicts`) can't be synced together; `manage_envs.py` validates this and prints the conflict sets. Run `python manage_envs.py list` to see all extras and conflict rules.

The uv flow targets **Linux x86_64 + Python 3.12 only** — `uv lock` / `uv sync` fail on macOS or other platforms. If there is no `uv` on the host, regenerate the lock inside Docker with `docker build -f docker/Dockerfile.uv.cu130 --target=lock ...` and copy `uv.lock` back out.

### 5. Sync Dockerfile system pins (after the re-lock)

`uv.lock` is the source of truth for the cuDNN / NCCL apt-deb versions baked into the Docker images. After a bump that moves torch (and thus its bundled `nvidia-*` wheels), reconcile them:

```bash
grep -E 'nvidia-(cudnn|nccl)-cu1[23]' uv.lock
# then update CUDNN_VERSION / NCCL_VERSION in docker/Dockerfile.uv.cu130 to match
```

### 6. Commit

Always commit the manifest and the lockfile **together** so the repo stays reproducible:

```bash
git add pyproject.toml uv.lock           # plus docker/Dockerfile.uv.cu130 if you touched it
```

### Quick ad-hoc test (non-persistent)

If the user only wants to *try* a version in an existing `.venv` without re-locking:

```bash
source .venv/bin/activate
uv pip install -U vllm                    # or: uv pip install vllm==<ver>
```

Warn the user this is **not** captured in `uv.lock` and the next `manage_envs.py sync` reverts it. For a lasting change, use steps 1–6.

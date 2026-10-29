# Building and Publishing Docker Images

This guide documents how the stable vLLM and SGLang training images are built and
published. Building is automated through GitHub Actions workflows that run on a
self-hosted builder machine and delegate the actual work to
[`scripts/verl-build.sh`](../../scripts/verl-build.sh) in this repository.

## Overview

| Component | Location | Responsibility |
| --- | --- | --- |
| `docker-build-vllm.yml` | `.github/workflows/` | CI for the vLLM image |
| `docker-build-sglang.yml` | `.github/workflows/` | CI for the SGLang image |
| `Dockerfile.stable.vllm` | `docker/` | vLLM image definition |
| `Dockerfile.stable.sglang` | `docker/` | SGLang image definition |
| `verl-build.sh` | `scripts/` | Build + push + verify |
| Docker Hub | `verlai/verl:<tag>` | Public image registry |
| Volcengine | `verl-ci-cn-beijing.cr.volces.com/verlai/verl:<tag>` | CI mirror registry |

Each image is published to **both** registries with the same tag. The Volcengine
reference is just the Docker Hub reference prefixed with the registry host.

## End-to-end flow

```text
push / MR / manual dispatch
        │
        ▼
GitHub Actions (runner label: docker-build-only)
  1. Check: does <image> already exist on BOTH registries?
        ├── yes ──► skip checkout + build (job succeeds, nothing rebuilt)
        └── no  ──► continue
  2. Checkout repo into the working directory (docker build context)
  3. Run scripts/verl-build.sh <dockerfile> <image>   (BYTED_PROXY passed as env)
        a. cleanup: remove old verlai/verl* images + prune build cache
        b. docker build with proxy build-args          (retried)
        c. push to Docker Hub                          (retried)  + verify digest
        d. retag to Volcengine and push                (retried)  + verify digest
```

## CI workflows

Both workflows are intentionally near-identical; only the Dockerfile path and the
default image tag differ.

- **Triggers**: `push`, `pull_request` (MR), and manual `workflow_dispatch`. The
  `push`/`pull_request` triggers are **path-filtered** to the relevant Dockerfile
  and the workflow file itself, so unrelated changes never launch an expensive build.
- **Concurrency**: `cancel-in-progress: true` keyed on `${{ github.workflow }}-${{ github.ref }}`.
  A newer push/MR update on the same ref cancels a build that is still running.
- **Skip-if-exists**: the first step queries both registries with
  `docker buildx imagetools inspect`. If the tag already exists on both, the
  checkout and build steps are skipped (they show as *skipped* in the Actions UI).
- **Force rebuild**: a manual run with the `force=true` input bypasses the
  skip-if-exists check and rebuilds/re-pushes the tag.
- **Proxy**: the build proxy is passed to the script from the `BYTED_PROXY` repository
  secret (set as `env.BYTED_PROXY` on the build step).
- **Runner**: `runs-on: [docker-build-only]` — the self-hosted builder machine.
- **Full build logs**: the workflow sets `BUILDKIT_PROGRESS=plain` so the complete,
  uncollapsed docker build progress is streamed to the CI logs (nothing is folded).

### Default tags

| Workflow | Dockerfile | Default image |
| --- | --- | --- |
| `docker-build-vllm.yml` | `docker/Dockerfile.stable.vllm` | `verlai/verl:vllm024.dev1` |
| `docker-build-sglang.yml` | `docker/Dockerfile.stable.sglang` | `verlai/verl:sgl0512.dev1` |

## The build script

[`scripts/verl-build.sh`](../../scripts/verl-build.sh) takes exactly two arguments —
the Dockerfile path (relative to the repo root) and the target Docker Hub image
reference — and must be run from the repo root (the build context is `.`). It reads the
proxy from the `BYTED_PROXY` environment variable (the CI secret) and expects the
machine to already be logged in to both registries.

```bash
BYTED_PROXY=http://sys-proxy-rd-relay.byted.org:8118 \
  bash scripts/verl-build.sh docker/Dockerfile.stable.vllm verlai/verl:vllm024.dev1
```

Behavior:

- **Cleanup first**: removes every local `verlai/verl*` image (both registries) and
  runs `docker builder prune -af` to free disk before building.
- **Proxy build-args**: reads the proxy from `BYTED_PROXY` (the CI secret) and passes it
  as `http_proxy`/`https_proxy` (lower and upper case) so package downloads succeed
  inside the build. If `BYTED_PROXY` is unset, it builds without proxy build-args.
- **Full build logs**: builds with `--progress=plain` so the complete, uncollapsed
  step-by-step output streams to the CI logs.
- **Robust retries**: `docker build` and each `docker push` are retried
  (`MAX_RETRIES`, default 3; `RETRY_DELAY`, default 30s). A failed build resumes
  from the last cached layer, so retries are cheap.
- **Digest verification**: after each push it compares the local (just-built)
  manifest digest against the digest the registry now serves for the tag, and fails
  if they differ.

Tunable per invocation, e.g. `MAX_RETRIES=5 RETRY_DELAY=60 bash scripts/verl-build.sh ...`.

## Publishing a new image

1. Edit the relevant Dockerfile under `docker/` (e.g. bump `VLLM_VERSION`).
2. Bump the image tag so it is unique (e.g. `vllm024.dev1` → `vllm024.dev2`).
   Update the `DEFAULT_IMAGE` env and the `workflow_dispatch` default in the
   matching workflow file so pushes publish the new tag.
3. Commit and push (or open an MR). The path-filtered workflow builds the new tag
   and publishes it to both registries.

To rebuild an **existing** tag (which skip-if-exists would otherwise skip), trigger
the workflow manually (`Run workflow`) with `force=true`.

## Prerequisites

Repository:

- A `BYTED_PROXY` secret set to the build proxy (e.g.
  `http://sys-proxy-rd-relay.byted.org:8118`), consumed by the build step.

The `docker-build-only` self-hosted runner must have:

- Docker with `buildx` available (used for the build, digest verification, and existence checks).
- Network access to the proxy in `BYTED_PROXY`.
- An authenticated `docker login` to Docker Hub (`verlai`) and to
  `verl-ci-cn-beijing.cr.volces.com` (Volcengine).

## Troubleshooting

- **Build was skipped unexpectedly** — the tag already exists on both registries.
  Bump the tag, or re-run manually with `force=true`.
- **`digest mismatch` error** — the push did not land or the tag was overwritten
  concurrently; re-run the build.
- **`could not read remote digest`** — registry/login/proxy issue on the builder;
  verify `docker login` and `docker buildx imagetools inspect <image>` by hand.
- **Build fails only sometimes** — usually a transient network fault; the script
  already retries. Increase `MAX_RETRIES`/`RETRY_DELAY` for flaky networks.
- **`building without proxy build-args` warning** — `BYTED_PROXY` is not set; confirm
  the secret exists and is exposed to the build step.

Last updated: 07/23/2026

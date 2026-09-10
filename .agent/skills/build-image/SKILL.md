---
name: build-image
description: Build and publish the stable vLLM/SGLang verl Docker images to Docker Hub and Volcengine via the docker-build CI. Use when the user asks to build, rebuild, bump, or publish a verl training image, or to change the docker-build workflows or the builder script.
user_invocable: true
---

Build and publish the stable training images. Full reference:
[`docs/contributing/docker_image_build.md`](../../../docs/contributing/docker_image_build.md).

## Key facts

- Two workflows drive it: `.github/workflows/docker-build-vllm.yml` and
  `.github/workflows/docker-build-sglang.yml`.
- Each image is published to **both** registries with the same tag:
  - Docker Hub: `verlai/verl:<tag>`
  - Volcengine: `verl-ci-cn-beijing.cr.volces.com/verlai/verl:<tag>`
- CI runs on the self-hosted runner `docker-build-only` and calls
  `scripts/verl-build.sh <dockerfile> <image>` (in this repo), passing the build proxy
  from the `BYTED_PROXY` secret as `env.BYTED_PROXY`.

## Publish a new image

1. Edit the Dockerfile under `docker/` (e.g. bump `VLLM_VERSION`).
2. Bump the tag to a unique value and update **both** the `DEFAULT_IMAGE` env and the
   `workflow_dispatch` default in the matching workflow file.
3. Push or open an MR. The path-filtered workflow builds and publishes the new tag.

To rebuild an **existing** tag, trigger the workflow manually with input `force=true`
(otherwise skip-if-exists skips it).

## CI behavior (do not regress these)

- Triggers: `push`, `pull_request`, `workflow_dispatch`; `push`/`pull_request` are
  path-filtered to the Dockerfile + the workflow file.
- Concurrency `cancel-in-progress: true` — a new push cancels the running build.
- First step checks both registries (`docker buildx imagetools inspect`); if the tag
  exists on both, checkout + build are skipped. `force=true` bypasses this.

## Build script contract

- Takes exactly two args: `<dockerfile>` (relative to repo root) and `<target-image>`
  (Docker Hub `repo:tag`); must run from the repo root (`.` build context).
- Reads the proxy from `BYTED_PROXY` and passes it as build-args, cleans old
  `verlai/verl*` images + build cache, retries build/push on failure, and verifies the
  pushed digest on each registry.
- Tunables: `MAX_RETRIES` (default 3), `RETRY_DELAY` (default 30s).

The script lives at `scripts/verl-build.sh`; see the doc above for the full reference.

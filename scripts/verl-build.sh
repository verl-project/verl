#!/usr/bin/env bash
# scripts/verl-build.sh
# Build a verl image from a Dockerfile, push it to Docker Hub + Volcengine,
# and verify the remote digest matches the freshly built image.
#
# Robustness:
#   - Before building, old verlai/verl* images and the docker build cache are removed.
#   - build / push / remote-digest lookups are retried on transient failures.
#   - build runs with --progress=plain so the full progress is logged (CI-friendly).
#
# Usage:
#   scripts/verl-build.sh <dockerfile> <target-image>
#     <dockerfile>    Dockerfile path relative to the repo root, e.g. docker/Dockerfile.stable.vllm
#     <target-image>  Docker Hub image ref (repo:tag),          e.g. verlai/verl:vllm024.dev1
#
# Env:
#   BYTED_PROXY               HTTP(S) proxy for the build (in CI: the BYTED_PROXY secret).
#                             If unset, the image is built without proxy build-args.
#   MAX_RETRIES (default 3)   attempts for each fallible step
#   RETRY_DELAY (default 30)  seconds between attempts
#
# Must be run from the repo root (the docker build context is ".").
# The machine is expected to be already logged in to both Docker Hub and Volcengine.

set -euo pipefail

DOCKERFILE="${1:?Usage: verl-build.sh <dockerfile> <target-image>}"
TARGET_IMAGE="${2:?Usage: verl-build.sh <dockerfile> <target-image>}"

# Proxy is provided by CI via the BYTED_PROXY secret (optional for local runs).
PROXY="${BYTED_PROXY:-}"
# Volcengine registry prefix -> verl-ci-cn-beijing.cr.volces.com/verlai/verl:xxx
VOLC_REGISTRY="verl-ci-cn-beijing.cr.volces.com"
VOLC_IMAGE="${VOLC_REGISTRY}/${TARGET_IMAGE}"

MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-30}"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "ERROR: Dockerfile '${DOCKERFILE}' not found. Run this script from the repo root." >&2
  exit 1
fi

# Pass the proxy to the build as the conventional http(s)_proxy build-args, if set.
proxy_build_args=()
if [ -n "${PROXY}" ]; then
  proxy_build_args=(
    --build-arg "http_proxy=${PROXY}"
    --build-arg "HTTP_PROXY=${PROXY}"
    --build-arg "https_proxy=${PROXY}"
    --build-arg "HTTPS_PROXY=${PROXY}"
  )
else
  echo "WARN: BYTED_PROXY is not set; building without proxy build-args." >&2
fi

# Retry a command up to MAX_RETRIES times with a fixed delay between attempts.
retry() {
  local desc="$1"; shift
  local attempt=1
  while true; do
    echo ">>> [${desc}] attempt ${attempt}/${MAX_RETRIES}"
    if "$@"; then
      return 0
    fi
    if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
      echo "ERROR: [${desc}] failed after ${MAX_RETRIES} attempts." >&2
      return 1
    fi
    echo "WARN: [${desc}] failed (attempt ${attempt}); retrying in ${RETRY_DELAY}s..." >&2
    sleep "${RETRY_DELAY}"
    attempt=$((attempt + 1))
  done
}

# Remove old verlai/verl* images (both registries) and prune the build cache.
cleanup() {
  echo ">>> [cleanup] Removing old verlai/verl* images"
  docker images --format '{{.ID}} {{.Repository}}' \
    | awk '$2 ~ /verlai\/verl/ {print $1}' \
    | sort -u \
    | xargs -r docker rmi -f 2>/dev/null || true
  echo ">>> [cleanup] Pruning docker build cache"
  docker builder prune -af || true
}

# Manifest digest of the just-built/just-pushed local image for a given repo.
local_digest() {
  local image="$1"
  local repo="${image%:*}"
  docker inspect --format='{{range .RepoDigests}}{{println .}}{{end}}' "${image}" 2>/dev/null \
    | grep "^${repo}@" | head -1 | cut -d'@' -f2 || true
}

# Digest the remote registry currently serves for the tag (retried).
remote_digest() {
  local image="$1" attempt=1 d=""
  while true; do
    d="$(docker buildx imagetools inspect "${image}" --format '{{.Manifest.Digest}}' 2>/dev/null || true)"
    if [ -n "${d}" ]; then
      printf '%s' "${d}"
      return 0
    fi
    if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
      return 1
    fi
    echo "WARN: no remote digest for ${image} yet (attempt ${attempt}); retrying in ${RETRY_DELAY}s..." >&2
    sleep "${RETRY_DELAY}"
    attempt=$((attempt + 1))
  done
}

# Push (retried), then assert the remote tag now points to exactly what we built.
push_and_verify() {
  local image="$1"
  retry "push ${image}" docker push "${image}"

  echo ">>> Verifying remote digest matches the latest build: ${image}"
  local built remote
  built="$(local_digest "${image}")"
  if ! remote="$(remote_digest "${image}")"; then
    echo "ERROR: could not read remote digest for ${image} after ${MAX_RETRIES} attempts." >&2
    exit 1
  fi
  echo "      built  : ${built:-<none>}"
  echo "      remote : ${remote:-<none>}"
  if [[ -z "${built}" || -z "${remote}" ]]; then
    echo "ERROR: could not resolve digest for ${image}" >&2
    exit 1
  fi
  if [[ "${built}" != "${remote}" ]]; then
    echo "ERROR: digest mismatch for ${image}: built ${built} != remote ${remote}" >&2
    exit 1
  fi
  echo "      OK: remote digest matches the latest build"
}

# --- Clean slate: drop old verl images + build cache before building --------
cleanup

# --- Build (retried, full progress logged) ----------------------------------
echo ">>> Building ${TARGET_IMAGE} from ${DOCKERFILE}"
retry "build ${TARGET_IMAGE}" docker build \
  --progress=plain \
  -t "${TARGET_IMAGE}" \
  -f "${DOCKERFILE}" \
  ${proxy_build_args[@]+"${proxy_build_args[@]}"} \
  .

# --- Push + verify ----------------------------------------------------------
# Docker Hub (direct push)
push_and_verify "${TARGET_IMAGE}"

# Volcengine (retag then push)
echo ">>> Retagging for Volcengine: ${VOLC_IMAGE}"
docker tag "${TARGET_IMAGE}" "${VOLC_IMAGE}"
push_and_verify "${VOLC_IMAGE}"

echo ">>> Done. Verified & pushed:"
echo "      ${TARGET_IMAGE}"
echo "      ${VOLC_IMAGE}"

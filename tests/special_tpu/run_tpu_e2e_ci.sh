#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
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
#
# TPU CI driver: runs one test of verl + verl-hardware-plugin on Cloud TPU v6e (GKE, KubeRay, Kueue).
#
# Every invocation runs one test on its own short-lived RayCluster:
#   1. image     - find the image of examples/tpu/grpo/Dockerfile (tagged by the Dockerfile's hash). It is only built
#                  here when TPU_CI_ALLOW_IMAGE_BUILD=true, i.e. for code that is already in the repository.
#   2. provision - render gke/raycluster-ci.yaml into a uniquely named RayCluster queued by Kueue: it stays suspended
#                  until its chip-count queue has TPU quota, so concurrent runs wait in FIFO order.
#   3. run       - submit the test the way examples/tpu/grpo/README.md runs jobs (this checkout as the Ray working
#                  dir, verl-hardware-plugin as a py_module) and check the job log with verify_tpu_e2e_log.py.
#   4. tear down - delete the RayCluster on exit, whether the test passed, failed, timed out or was cancelled.
#
# Tests (one per workflow, see tests/special_tpu/README.md):
#   tpu_platform  1 x v6e-1  tpu_unit_tests.yml             tests/special_tpu/tpu_smoke_test.py
#   sft           1 x v6e-4  e2e_sft_llm_tpu.yml            examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
#   grpo          2 x v6e-1  e2e_v1_separate_async_tpu.yml  examples/tpu/grpo/run_qwen3_torchtitan.sh (0.6B)
#   grpo_4b       2 x v6e-4  nightly_tpu.yml                examples/tpu/grpo/run_qwen3_torchtitan.sh (4B)
#
# Usage:
#   bash tests/special_tpu/run_tpu_e2e_ci.sh <tpu_platform|sft|grpo|grpo_4b>
#   bash tests/special_tpu/run_tpu_e2e_ci.sh teardown [test]   # delete the RayClusters of this run
#   bash tests/special_tpu/run_tpu_e2e_ci.sh build_image       # build the image of the Dockerfile if it is missing

set -euo pipefail

USAGE="usage: $0 <tpu_platform|sft|grpo|grpo_4b|teardown [test]|build_image>"
MODE="${1:?${USAGE}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=tests/special_tpu/gke/tpu-ci.env
set -a
source tests/special_tpu/gke/tpu-ci.env
set +a
NS="${TPU_CI_NAMESPACE}"

# Maximum time to wait in the Kueue queue for TPU quota.
QUEUE_TIMEOUT_MINS="${QUEUE_TIMEOUT_MINS:-120}"
# Maximum time from admission until all pods run (TPU node scale-up and image pull, or hosts still held by others).
POD_READY_TIMEOUT_MINS="${POD_READY_TIMEOUT_MINS:-25}"
# CI RayClusters older than this are leaked (e.g. the runner pod died) and get reaped.
STALE_CLUSTER_SECONDS="${STALE_CLUSTER_SECONDS:-10800}"
# The TPU node pools may be spot VMs: re-run a test this many times on a new RayCluster when a TPU node it ran on
# was preempted.
PREEMPTION_RETRIES="${PREEMPTION_RETRIES:-1}"
# Whether this run may build the image (privileged BuildKit pod that pushes to TPU_CI_IMAGE_REPO). The workflows only
# allow it for code that is already in the repository, never for pull requests from forks.
TPU_CI_ALLOW_IMAGE_BUILD="${TPU_CI_ALLOW_IMAGE_BUILD:-false}"
IMAGE_BUILD_TIMEOUT_MINS="${IMAGE_BUILD_TIMEOUT_MINS:-90}"
# How long a run that may not build waits for the image to appear (built by docker-build-tpu.yml).
IMAGE_WAIT_MINS="${IMAGE_WAIT_MINS:-0}"
TPU_CI_DOCKERFILE="${TPU_CI_DOCKERFILE:-examples/tpu/grpo/Dockerfile}"
PORT_FORWARD_PORT="${PORT_FORWARD_PORT:-$((28000 + RANDOM % 1000))}"

RUN_ID="${GITHUB_RUN_ID:-local$(date +%s)}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
WORK_DIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/verl-tpu-ci-${RUN_ID}"

PF_PID=""
LOG_FOLLOW_PID=""
CURRENT_RAY_JOB_ID=""
RAY_CLUSTER_NAME=""
HEAD_POD=""
TPU_NODES=""
KUEUE_QUEUE=""
TIER=""
WORKER_REPLICAS=1
WORKER_TPU_CHIPS=1
PLUGIN_PKG=""
RUNTIME_ENV=""
# Appended to the RayCluster name when a test is re-run after a preemption.
CLUSTER_SUFFIX=""

log() {
    echo "[TPU CI $(date +%H:%M:%S)] $*"
}

connect_gke_cluster() {
    if [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then
        # Self-hosted runner pod in the CI cluster: its in-cluster service account.
        unset KUBECONFIG
        if ! command -v kubectl >/dev/null 2>&1; then
            curl -fsSLO "https://dl.k8s.io/release/${KUBECTL_VERSION:-v1.31.0}/bin/linux/amd64/kubectl"
            install -m 0755 kubectl /usr/local/bin/kubectl
            rm -f kubectl
        fi
        log "Running in a pod of the CI cluster; using its service account."
    else
        # A private kubeconfig, so that a run from a workstation does not switch the current kubectl context.
        export KUBECONFIG="${TPU_CI_KUBECONFIG:-${WORK_DIR}/kubeconfig}"
        mkdir -p "$(dirname "${KUBECONFIG}")"
        log "Connecting to GKE cluster ${TPU_CI_GKE_CLUSTER} (${TPU_CI_GKE_REGION}, ${TPU_CI_GKE_PROJECT})"
        gcloud container clusters get-credentials "${TPU_CI_GKE_CLUSTER}" \
            --region "${TPU_CI_GKE_REGION}" --project "${TPU_CI_GKE_PROJECT}" --dns-endpoint
    fi
}

cleanup() {
    local exit_code=$?
    set +e
    if [[ -n "${CURRENT_RAY_JOB_ID}" && -n "${HEAD_POD}" ]]; then
        log "Stopping Ray job ${CURRENT_RAY_JOB_ID}..."
        kubectl exec -n "${NS}" "${HEAD_POD}" -c ray-head -- \
            ray job stop --address http://127.0.0.1:8265 "${CURRENT_RAY_JOB_ID}" >/dev/null 2>&1
        CURRENT_RAY_JOB_ID=""
    fi
    local pid
    for pid in ${PF_PID} ${LOG_FOLLOW_PID}; do
        kill "${pid}" 2>/dev/null
    done
    if [[ -n "${RAY_CLUSTER_NAME}" ]]; then
        log "Deleting RayCluster ${RAY_CLUSTER_NAME} (exit code ${exit_code})..."
        kubectl delete raycluster -n "${NS}" "${RAY_CLUSTER_NAME}" --wait=false --ignore-not-found
        RAY_CLUSTER_NAME=""
    fi
    exit "${exit_code}"
}

reap_stale_clusters() {
    local stale name
    stale="$(kubectl get raycluster -n "${NS}" -l verl-ci/managed=true -o json | python3 -c '
import datetime, json, sys
now = datetime.datetime.now(datetime.timezone.utc)
for item in json.load(sys.stdin)["items"]:
    created = datetime.datetime.fromisoformat(item["metadata"]["creationTimestamp"].replace("Z", "+00:00"))
    if (now - created).total_seconds() > int(sys.argv[1]):
        print(item["metadata"]["name"])
' "${STALE_CLUSTER_SECONDS}")"
    for name in ${stale}; do
        log "Deleting leaked CI RayCluster ${name} (older than ${STALE_CLUSTER_SECONDS} s)."
        kubectl delete raycluster -n "${NS}" "${name}" --wait=false --ignore-not-found || true
    done
}

ensure_ray_cli() {
    # The Ray job CLI must match the Ray of the image (FROM rayproject/ray:<version> in the Dockerfile).
    local want have
    want="$(sed -n 's|^FROM rayproject/ray:\([0-9.]*\).*|\1|p' examples/tpu/grpo/Dockerfile | head -n 1)"
    have="$(ray --version 2>/dev/null | awk '{print $NF}' || true)"
    if [[ -z "${want}" || "${have}" == "${want}" ]]; then
        return 0
    fi
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
        log "Installing the Ray job CLI ray[default]==${want} (the Ray of the image)."
        python3 -m pip install --quiet "ray[default]==${want}"
    elif [[ -z "${have}" ]]; then
        log "ERROR: install the Ray job CLI first: pip install \"ray[default]==${want}\"" >&2
        exit 1
    else
        log "Warning: Ray CLI ${have} differs from the image's Ray ${want}."
    fi
}

readme_plugin_pin() {
    # The verl-hardware-plugin repository and commit that step 1 of examples/tpu/grpo/README.md checks out.
    python3 - examples/tpu/grpo/README.md <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
repos = re.findall(r"^git clone (\S+/verl-hardware-plugin(?:\.git)?)\s*$", text, re.M)
commits = re.findall(r"^git -C verl-hardware-plugin checkout ([0-9a-f]{40})\s*$", text, re.M)
if len(repos) != 1 or len(commits) != 1:
    sys.exit(
        f"{sys.argv[1]}: expected one 'git clone <url>/verl-hardware-plugin.git' line and one "
        f"'git -C verl-hardware-plugin checkout <40-hex commit id>' line, found {len(repos)} and {len(commits)}."
    )
print(repos[0], commits[0])
PY
}

fetch_hardware_plugin() {
    # verl-hardware-plugin provides the TPU platform, engine, rollout and weight sync. It is shipped to the Ray jobs as
    # a py_module, as in examples/tpu/grpo/README.md: by default the commit that README checks out, or
    # VERL_HARDWARE_PLUGIN_REPO / VERL_HARDWARE_PLUGIN_COMMIT, or a local checkout in VERL_HARDWARE_PLUGIN_DIR.
    local plugin_dir="${VERL_HARDWARE_PLUGIN_DIR:-}" source
    if [[ -n "${plugin_dir}" ]]; then
        plugin_dir="$(cd "${plugin_dir}" && pwd)"
        source="local checkout"
    else
        local pin readme_repo readme_commit repo commit head
        pin="$(readme_plugin_pin)"
        read -r readme_repo readme_commit <<<"${pin}"
        repo="${VERL_HARDWARE_PLUGIN_REPO:-${readme_repo}}"
        commit="${VERL_HARDWARE_PLUGIN_COMMIT:-${readme_commit}}"
        plugin_dir="${WORK_DIR}/verl-hardware-plugin"
        source="${repo} ${commit}"
        rm -rf "${plugin_dir}"
        git init -q "${plugin_dir}"
        if ! git -C "${plugin_dir}" fetch -q --depth 1 "${repo}" "${commit}"; then
            log "ERROR: cannot fetch ${commit} from ${repo}: use a full 40-character commit id, a branch or a tag" \
                "of that repository." >&2
            exit 1
        fi
        git -C "${plugin_dir}" checkout -q FETCH_HEAD
        head="$(git -C "${plugin_dir}" rev-parse HEAD)"
        if [[ "${commit}" =~ ^[0-9a-f]{40}$ && "${head}" != "${commit}" ]]; then
            log "ERROR: fetched ${head} instead of ${commit}." >&2
            exit 1
        fi
        if [[ "${head}" != "${readme_commit}" ]]; then
            log "Note: examples/tpu/grpo/README.md checks out verl-hardware-plugin ${readme_commit}; this run tests ${head}."
        fi
    fi
    PLUGIN_PKG="${plugin_dir}/verl_hardware_plugin"
    if [[ ! -f "${PLUGIN_PKG}/__init__.py" ]]; then
        log "ERROR: ${PLUGIN_PKG} is not the verl_hardware_plugin package." >&2
        exit 1
    fi
    log "verl under test: $(git log -1 --format='%H %s' 2>/dev/null || echo "${REPO_ROOT}")"
    log "verl-hardware-plugin under test: $(git -C "${plugin_dir}" log -1 --format='%H %s' 2>/dev/null || echo '?')" \
        "(${source})"
    # The runtime environment of step 5 of examples/tpu/grpo/README.md, with this run's plugin path.
    RUNTIME_ENV="$(PLUGIN_PKG="${PLUGIN_PKG}" python3 -c '
import json, os
print(json.dumps({
    "py_modules": [os.environ["PLUGIN_PKG"]],
    "excludes": [".git", ".venv"],
    "env_vars": {
        "PYTHONPATH": ".",
        "PYTHONUNBUFFERED": "1",
        "VERL_PLATFORM": "tpu",
        "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    },
}))')"
}

runtime_env_with() {
    # RUNTIME_ENV plus extra env_vars (KEY=VALUE arguments). A test sets
    # `local JOB_RUNTIME_ENV; JOB_RUNTIME_ENV="$(runtime_env_with KEY=VALUE ...)"` before submit_and_verify_ray_job.
    RUNTIME_ENV="${RUNTIME_ENV}" python3 -c '
import json, os, sys
env = json.loads(os.environ["RUNTIME_ENV"])
env["env_vars"].update(kv.split("=", 1) for kv in sys.argv[1:])
print(json.dumps(env))' "$@"
}

# ---------------------------------------------------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------------------------------------------------

registry_access_token() {
    # Artifact Registry token: of the runner's Google service account (Workload Identity) in a runner pod, of the
    # gcloud account otherwise.
    if [[ -n "${KUBERNETES_SERVICE_HOST:-}" ]]; then
        curl -sf -H "Metadata-Flavor: Google" \
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" |
            python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
    else
        gcloud auth print-access-token
    fi
}

image_digest() {
    # Manifest digest of the image reference $1 (<host>/<path>:<tag>); empty if the registry does not have it.
    local image="$1" host path tag headers status
    host="${image%%/*}" path="${image#*/}" tag="${path##*:}" path="${path%:*}"
    headers="$(curl -s -o /dev/null -D - -H "Authorization: Bearer $(registry_access_token)" \
        -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json" \
        "https://${host}/v2/${path}/manifests/${tag}" | tr -d '\r')"
    status="$(awk 'NR == 1 { print $2 }' <<<"${headers}")"
    if [[ "${status}" == "401" || "${status}" == "403" ]]; then
        log "ERROR: cannot read ${host}/${path} (HTTP ${status}): the runner's Google service account needs" \
            "roles/artifactregistry.reader on it." >&2
        exit 1
    fi
    awk 'tolower($1) == "docker-content-digest:" { print $2 }' <<<"${headers}"
}

start_image_build() {
    # Build ${image} from TPU_CI_DOCKERFILE with daemonless BuildKit in a Kubernetes Job (gke/image-build-job.yaml).
    # The Dockerfile is the whole build context. Runs that start at the same time for the same Dockerfile share the
    # Job: AlreadyExists means another run created it.
    local image="$1" name="$2" token
    log "Building ${image} from ${TPU_CI_DOCKERFILE} (BuildKit Job ${name}, about 25 minutes)."
    token="$(registry_access_token)"
    python3 - "${token}" "${TPU_CI_IMAGE_REPO%%/*}" >"${WORK_DIR}/docker-config.json" <<'PY'
import base64, json, sys
auth = base64.b64encode(f"oauth2accesstoken:{sys.argv[1]}".encode()).decode()
print(json.dumps({"auths": {sys.argv[2]: {"auth": auth}}}))
PY
    kubectl create configmap -n "${NS}" "${name}" --from-file=Dockerfile="${TPU_CI_DOCKERFILE}" 2>/dev/null || true
    kubectl create secret generic -n "${NS}" "${name}" \
        --from-file=config.json="${WORK_DIR}/docker-config.json" 2>/dev/null || true
    rm -f "${WORK_DIR}/docker-config.json"
    NAME="${name}" NAMESPACE="${NS}" IMAGE="${image}" BUILDKIT_IMAGE="${TPU_CI_BUILDKIT_IMAGE}" \
        TIMEOUT_SECONDS="$((IMAGE_BUILD_TIMEOUT_MINS * 60))" \
        python3 - tests/special_tpu/gke/image-build-job.yaml <<'PY' | kubectl create -f - 2>/dev/null || true
import os
import string
import sys

keys = ["NAME", "NAMESPACE", "IMAGE", "BUILDKIT_IMAGE", "TIMEOUT_SECONDS"]
print(string.Template(open(sys.argv[1]).read()).substitute({k: os.environ[k] for k in keys}))
PY
}

ensure_ci_image() {
    # The Ray pods run the image of the Dockerfile under test, tagged TPU_CI_IMAGE_REPO:df-<hash of the Dockerfile>
    # and pinned by digest (a node may still cache an older image under the same tag). Set TPU_CI_IMAGE to use a
    # prebuilt image instead.
    if [[ -n "${TPU_CI_IMAGE:-}" ]]; then
        log "Using the prebuilt image TPU_CI_IMAGE=${TPU_CI_IMAGE}."
        return 0
    fi
    local hash image name digest status started=0 now
    hash="$(sha256sum "${TPU_CI_DOCKERFILE}" | cut -c1-16)"
    image="${TPU_CI_IMAGE_REPO}:df-${hash}"
    # One build Job per image (repository and Dockerfile), shared by the runs that need it.
    name="verl-ci-image-$(printf '%s' "${image}" | sha256sum | cut -c1-16)"
    local build_deadline=0 wait_deadline=$(($(date +%s) + IMAGE_WAIT_MINS * 60)) next_note=0
    while true; do
        digest="$(image_digest "${image}")"
        if [[ -n "${digest}" ]]; then
            break
        fi
        # Status of the build Job of this Dockerfile: "" (no Job), "running", "succeeded" or "failed".
        status="$(kubectl get job -n "${NS}" "${name}" \
            -o jsonpath='{.status.succeeded}{","}{.status.failed}' 2>/dev/null | awk -F, '
                $1 == "1" { print "succeeded"; next } $2 == "1" { print "failed"; next } { print "running" }' || true)"
        now="$(date +%s)"
        if [[ "${status}" == "failed" && "${TPU_CI_ALLOW_IMAGE_BUILD}" == "true" && "${started}" == "0" ]]; then
            # An earlier build of this Dockerfile failed (the Job is kept for a day): retry once.
            log "Deleting the failed image build Job ${name} to retry it."
            kubectl delete job,configmap,secret -n "${NS}" "${name}" --ignore-not-found --wait=true
            status=""
        fi
        if [[ "${status}" == "failed" || ("${build_deadline}" != "0" && "${now}" -gt "${build_deadline}") ]]; then
            log "ERROR: building ${image} failed or timed out. Last lines of the build log:" >&2
            kubectl logs -n "${NS}" "job/${name}" --tail=80 >&2 || true
            exit 1
        elif [[ "${status}" == "succeeded" ]]; then
            log "ERROR: the build Job ${name} succeeded but ${image} is not in the registry." >&2
            exit 1
        elif [[ -z "${status}" && "${TPU_CI_ALLOW_IMAGE_BUILD}" == "true" ]]; then
            start_image_build "${image}" "${name}"
            started=1
            build_deadline=$((now + IMAGE_BUILD_TIMEOUT_MINS * 60 + 600))
        elif [[ -z "${status}" ]]; then
            local how="Actions -> docker-build-tpu -> Run workflow"
            if [[ "${GITHUB_REF:-}" == refs/pull/*/merge ]]; then
                how="${how}, ref ${GITHUB_REF%/merge}/head"
            fi
            if ((now >= next_note)); then
                log "No image ${image} for ${TPU_CI_DOCKERFILE} yet, and this run may not build it (pull requests from" \
                    "forks never build images). After reviewing the Dockerfile, a maintainer builds it with: ${how}."
                next_note=$((now + 300))
            fi
            if ((now >= wait_deadline)); then
                log "ERROR: no image for this Dockerfile (waited ${IMAGE_WAIT_MINS} minutes). Build it, then re-run." >&2
                exit 1
            fi
        else
            log "Image build ${name}: $(kubectl logs -n "${NS}" "job/${name}" --tail=1 2>/dev/null | cut -c1-160 || true)"
            # Only the run that started the build (or a run that may build) enforces the build deadline.
            if [[ "${build_deadline}" == "0" ]]; then
                build_deadline=$((now + IMAGE_BUILD_TIMEOUT_MINS * 60 + 600))
            fi
        fi
        sleep 60
    done
    if [[ "${started}" == "1" ]]; then
        kubectl delete configmap,secret -n "${NS}" "${name}" --ignore-not-found >/dev/null || true
    fi
    export TPU_CI_IMAGE="${image}@${digest}"
    log "Image ${TPU_CI_IMAGE} (built from ${TPU_CI_DOCKERFILE})."
}

# ---------------------------------------------------------------------------------------------------------------------
# RayCluster
# ---------------------------------------------------------------------------------------------------------------------

provision_ray_cluster() {
    # shape: "v6e-1" = a whole 1x1 node of the ct6e-standard-1t pool (1 chip per worker);
    #        "v6e-4" = a 2x2 single-host subslice of the 2x4 (v6e-8) node pools (4 chips per worker).
    local test_name="$1" shape="$2" replicas="$3" head_cpu="$4" head_mem="$5" worker_cpu="$6" worker_mem="$7"
    local node_topology torch_topology annotation_key annotation_value slicebuilder
    WORKER_REPLICAS="${replicas}"
    case "${shape}" in
        v6e-1)
            WORKER_TPU_CHIPS=1 node_topology=1x1 torch_topology=1,1,1
            annotation_key=verl-ci/tpu-slice-topology annotation_value=1x1
            slicebuilder=localhost:8471
            ;;
        v6e-4)
            WORKER_TPU_CHIPS=4 node_topology=2x4 torch_topology=2,2,1
            annotation_key=cloud.google.com/gke-tpu-slice-topology annotation_value=2x2
            slicebuilder=localhost:8471,localhost:8472,localhost:8473,localhost:8474
            ;;
        *)
            log "ERROR: unknown worker shape ${shape}" >&2
            exit 1
            ;;
    esac
    local chips=$((WORKER_TPU_CHIPS * WORKER_REPLICAS))
    TIER="v6e-${chips}chip"
    KUEUE_QUEUE="${TPU_CI_KUEUE_QUEUE_PREFIX}-${chips}chip"
    # CI never changes cluster-wide objects: the queues are created by an admin.
    if ! kubectl get localqueue -n "${NS}" "${KUEUE_QUEUE}" >/dev/null 2>&1; then
        log "ERROR: Kueue LocalQueue ${NS}/${KUEUE_QUEUE} does not exist. A cluster admin creates the TPU CI queues" \
            "with 'kubectl apply -f tests/special_tpu/gke/kueue-tpu-ci.yaml'." >&2
        exit 1
    fi
    # Keep the name short: KubeRay derives pod and service names from it (63-character limit).
    RAY_CLUSTER_NAME="$(echo "verl-ci-${test_name}-${RUN_ID}-${RUN_ATTEMPT}" | tr '[:upper:]_' '[:lower:]-' |
        cut -c1-37 | sed 's/-*$//')${CLUSTER_SUFFIX}"

    log "Creating RayCluster ${RAY_CLUSTER_NAME} (${replicas} x ${shape}, queue ${KUEUE_QUEUE}, image ${TPU_CI_IMAGE})"
    RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME}" NAMESPACE="${NS}" KUEUE_QUEUE="${KUEUE_QUEUE}" TIER="${TIER}" \
        TEST_NAME="${test_name//_/-}" CI_RUN_ID="${RUN_ID}" \
        CI_OWNER="${GITHUB_REPOSITORY:-local}/${GITHUB_REF_NAME:-$(hostname)}#${RUN_ID}" \
        RAY_SERVICE_ACCOUNT="${TPU_CI_RAY_SERVICE_ACCOUNT}" IMAGE="${TPU_CI_IMAGE}" BUCKET="${TPU_CI_BUCKET}" \
        HEAD_CPU="${head_cpu}" HEAD_MEMORY="${head_mem}" WORKER_CPU="${worker_cpu}" WORKER_MEMORY="${worker_mem}" \
        WORKER_REPLICAS="${replicas}" WORKER_TPU_CHIPS="${WORKER_TPU_CHIPS}" WORKER_NODE_TOPOLOGY="${node_topology}" \
        SLICE_ANNOTATION_KEY="${annotation_key}" SLICE_ANNOTATION_VALUE="${annotation_value}" \
        TORCH_TPU_TOPOLOGY="${torch_topology}" SLICEBUILDER_ADDRESSES="${slicebuilder}" \
        python3 - tests/special_tpu/gke/raycluster-ci.yaml <<'PY' | kubectl apply -f -
import os
import string
import sys

print(string.Template(open(sys.argv[1]).read()).substitute(os.environ))
PY
}

wait_for_ray_cluster() {
    # 1. Kueue admission (the RayCluster is un-suspended).
    local deadline=$((SECONDS + QUEUE_TIMEOUT_MINS * 60)) last_report=-1000 suspended
    while ((SECONDS < deadline)); do
        suspended="$(kubectl get raycluster -n "${NS}" "${RAY_CLUSTER_NAME}" -o jsonpath='{.spec.suspend}')"
        if [[ "${suspended}" != "true" ]]; then
            log "RayCluster ${RAY_CLUSTER_NAME} admitted by Kueue."
            break
        fi
        if ((SECONDS - last_report >= 60)); then
            log "Queued in Kueue (${KUEUE_QUEUE}), waiting for TPU quota. Clusters of this tier:"
            kubectl get raycluster -n "${NS}" -l "verl-ci/managed=true,verl-ci/tier=${TIER}" \
                -o custom-columns=NAME:.metadata.name,SUSPENDED:.spec.suspend,STATE:.status.state,OWNER:.metadata.annotations.verl-ci/owner \
                --no-headers || true
            last_report=${SECONDS}
        fi
        sleep 10
    done
    if [[ "$(kubectl get raycluster -n "${NS}" "${RAY_CLUSTER_NAME}" -o jsonpath='{.spec.suspend}')" == "true" ]]; then
        log "ERROR: RayCluster not admitted within ${QUEUE_TIMEOUT_MINS} minutes." >&2
        exit 1
    fi

    # 2. Head and TPU worker pods running with all containers ready.
    local want_pods=$((1 + WORKER_REPLICAS)) ready_pods=0
    deadline=$((SECONDS + POD_READY_TIMEOUT_MINS * 60))
    while ((SECONDS < deadline)); do
        ready_pods="$(kubectl get pods -n "${NS}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" --no-headers 2>/dev/null |
            awk '{split($2, r, "/"); if (r[1] == r[2] && $3 == "Running") c++} END {print c+0}')"
        if [[ "${ready_pods}" -ge "${want_pods}" ]]; then
            break
        fi
        log "Ready pods: ${ready_pods}/${want_pods} (1 head + ${WORKER_REPLICAS} TPU worker(s))."
        sleep 10
    done
    kubectl get pods -n "${NS}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide
    if [[ "${ready_pods}" -lt "${want_pods}" ]]; then
        log "ERROR: pods of ${RAY_CLUSTER_NAME} not ready within ${POD_READY_TIMEOUT_MINS} minutes of admission." >&2
        kubectl get events -n "${NS}" --sort-by=.lastTimestamp 2>/dev/null |
            awk -v name="${RAY_CLUSTER_NAME}" 'index($0, name)' | tail -n 20 >&2 || true
        exit 1
    fi
    HEAD_POD="$(kubectl get pods -n "${NS}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=head" \
        -o jsonpath='{.items[0].metadata.name}')"
    TPU_NODES="$(kubectl get pods -n "${NS}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
        -o jsonpath='{.items[*].spec.nodeName}')"

    # 3. TPU workers registered their chips with Ray.
    local want_tpu="0.0/$((WORKER_TPU_CHIPS * WORKER_REPLICAS)).0 TPU" tpu_usage
    deadline=$((SECONDS + 300))
    while ((SECONDS < deadline)); do
        tpu_usage="$(kubectl exec -n "${NS}" "${HEAD_POD}" -c ray-head -- ray status 2>/dev/null |
            awk '/[0-9.]+\/[0-9.]+ TPU([[:space:]]|$)/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); print $0; exit}' ||
            true)"
        log "Ray TPU resources: ${tpu_usage:-<initializing>} (want ${want_tpu})"
        if [[ "${tpu_usage}" == "${want_tpu}" ]]; then
            return 0
        fi
        sleep 10
    done
    log "ERROR: ${RAY_CLUSTER_NAME} did not register $((WORKER_TPU_CHIPS * WORKER_REPLICAS)) TPU chips with Ray." >&2
    exit 1
}

connect_ray_dashboard() {
    local svc="${RAY_CLUSTER_NAME}-head-svc" deadline=$((SECONDS + 60))
    # Runner pods live in the same cluster: use the head service directly.
    while ((SECONDS < deadline)); do
        if curl -sf "http://${svc}.${NS}.svc.cluster.local:8265/api/version" >/dev/null 2>&1; then
            export RAY_ADDRESS="http://${svc}.${NS}.svc.cluster.local:8265"
            log "Ray dashboard: ${RAY_ADDRESS}"
            return 0
        fi
        if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
            break
        fi
        sleep 5
    done
    log "Port-forwarding svc/${svc} to localhost:${PORT_FORWARD_PORT}."
    kubectl port-forward -n "${NS}" "svc/${svc}" "${PORT_FORWARD_PORT}:8265" >/dev/null 2>&1 &
    PF_PID=$!
    deadline=$((SECONDS + 60))
    while ((SECONDS < deadline)); do
        if curl -sf "http://127.0.0.1:${PORT_FORWARD_PORT}/api/version" >/dev/null 2>&1; then
            export RAY_ADDRESS="http://127.0.0.1:${PORT_FORWARD_PORT}"
            log "Ray dashboard: ${RAY_ADDRESS}"
            return 0
        fi
        sleep 1
    done
    log "ERROR: cannot reach the Ray dashboard of ${RAY_CLUSTER_NAME}." >&2
    exit 1
}

print_environment() {
    # Like the "Check npu and CANN info" / "Check final pip list" steps of the NPU and GPU workflows: record the TPU
    # slices and the image's key packages, so a failure can be traced to an environment change.
    local worker
    log "Ray cluster resources:"
    kubectl exec -n "${NS}" "${HEAD_POD}" -c ray-head -- ray status 2>/dev/null | sed -n '/^Resources/,/^Demands/p' || true
    worker="$(kubectl get pods -n "${NS}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
        -o jsonpath='{.items[0].metadata.name}')"
    log "TPU environment of ${worker}:"
    kubectl exec -n "${NS}" "${worker}" -c ray-worker -- sh -c 'env | sort | awk -F= "/^(TPU_|TORCH_TPU_)/"' || true
    log "Key packages of ${TPU_CI_IMAGE}:"
    kubectl exec -n "${NS}" "${worker}" -c ray-worker -- python3 -m pip list 2>/dev/null |
        awk '{name = tolower($1); gsub(/_/, "-", name)}
             name ~ /^(torch|torchvision|torch-tpu|tpu-sync-torch|jax|jaxlib|libtpu|vllm|vllm-torchtpu|transformers|ray|tensordict|transferqueue)$/' ||
        true
}

warm_model_cache() {
    # The workers read the model through Cloud Storage FUSE. Several trainer processes faulting in a cold multi-GB
    # checkpoint at once can take more than 10 minutes, so read it once sequentially on every worker first.
    local model_dir="$1" worker pids=()
    for worker in $(kubectl get pods -n "${NS}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
        -o jsonpath='{.items[*].metadata.name}'); do
        log "Warming the page cache of ${worker} with ${model_dir}..."
        kubectl exec -n "${NS}" "${worker}" -c ray-worker -- sh -c "cat ${model_dir}/*.safetensors > /dev/null" &
        pids+=($!)
    done
    wait "${pids[@]}"
}

diagnose_cluster() {
    # Explain a failed job: show the pods and whether the TPU nodes they started on are still there (a preempted spot
    # node kills its worker pod, and the job, mid-run).
    local node status
    log "Pods of ${RAY_CLUSTER_NAME}:" >&2
    kubectl get pods -n "${NS}" -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide >&2 || true
    for node in ${TPU_NODES}; do
        status="$(kubectl get node "${node}" --no-headers 2>/dev/null | awk '{print $2}')"
        if [[ "${status}" != "Ready" ]]; then
            log "TPU node ${node} is ${status:-gone}: most likely a spot preemption." >&2
            touch "${WORK_DIR}/preempted"
        fi
    done
}

show_log() {
    # The tail of a log, without the per-second VERL_TIMELINE telemetry of the TPU recipes. The workflows upload the
    # whole logs as an artifact (tpu-logs-<test>-<attempt>).
    awk '!/VERL_TIMELINE\{/' "$1" | tail -n 200
}

verify_log() {
    local verifier="$1" log_file="$2" verifier_args="$3" summary
    # shellcheck disable=SC2086  # verifier_args is a list of flags
    summary="$(python3 tests/special_tpu/verify_tpu_e2e_log.py "${verifier}" "${log_file}" ${verifier_args})"
    echo "${summary}"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        echo "${summary}" >>"${GITHUB_STEP_SUMMARY}"
    fi
}

submit_and_verify_ray_job() {
    # Usage: submit_and_verify_ray_job <verifier> <timeout_mins> <verifier args> -- <entrypoint...>
    local verifier="$1" timeout_mins="$2" verifier_args="$3"
    shift 4
    local sub_id log_file status_out status_lower final_status="TIMEOUT"
    sub_id="ci_${verifier}_$(date +%Y%m%d_%H%M%S)"
    log_file="${WORK_DIR}/${sub_id}.log"

    log "Submitting Ray job ${sub_id}: $*"
    ray job submit --address "${RAY_ADDRESS}" --submission-id "${sub_id}" --working-dir "${REPO_ROOT}" \
        --runtime-env-json "${JOB_RUNTIME_ENV:-${RUNTIME_ENV}}" --no-wait -- "$@"
    CURRENT_RAY_JOB_ID="${sub_id}"

    # Stream the log while the job runs: the driver log is gone if its node is preempted before the job ends.
    kubectl exec -n "${NS}" "${HEAD_POD}" -c ray-head -- \
        ray job logs --address http://127.0.0.1:8265 --follow "${sub_id}" >"${log_file}.stream" 2>&1 &
    LOG_FOLLOW_PID=$!

    # Poll through kubectl exec, so that a dropped dashboard connection does not fail the run.
    local deadline=$((SECONDS + timeout_mins * 60))
    while ((SECONDS < deadline)); do
        status_out="$(kubectl exec -n "${NS}" "${HEAD_POD}" -c ray-head -- \
            ray job status --address http://127.0.0.1:8265 "${sub_id}" 2>/dev/null |
            awk "/Job '.*' (succeeded|failed|was stopped)|Status message|Status for job/" || true)"
        log "${sub_id}: $(tail -n 1 <<<"${status_out}")"
        if [[ -z "${status_out}" ]] && ! kubectl get pod -n "${NS}" "${HEAD_POD}" >/dev/null 2>&1; then
            log "ERROR: the Ray head pod ${HEAD_POD} is gone (evicted or preempted)." >&2
            final_status="HEAD_LOST"
            break
        fi
        status_lower="${status_out,,}"
        if [[ "${status_lower}" == *"' succeeded"* || "${status_lower}" == *"': succeeded"* ]]; then
            final_status="SUCCEEDED"
            break
        elif [[ "${status_lower}" =~ \':?\ (failed|stopped|was\ stopped) ]]; then
            final_status="FAILED"
            break
        fi
        sleep 20
    done
    if [[ "${final_status}" == "TIMEOUT" ]]; then
        log "Ray job ${sub_id} exceeded ${timeout_mins} minutes; stopping it."
        kubectl exec -n "${NS}" "${HEAD_POD}" -c ray-head -- \
            ray job stop --address http://127.0.0.1:8265 "${sub_id}" >/dev/null 2>&1 || true
    fi
    CURRENT_RAY_JOB_ID=""

    if kubectl exec -n "${NS}" "${HEAD_POD}" -c ray-head -- \
        ray job logs --address http://127.0.0.1:8265 "${sub_id}" >"${log_file}" 2>/dev/null; then
        rm -f "${log_file}.stream"
    else
        log "Could not fetch the job log; using the log streamed during the run."
        mv "${log_file}.stream" "${log_file}"
    fi
    kill "${LOG_FOLLOW_PID}" 2>/dev/null || true
    wait "${LOG_FOLLOW_PID}" 2>/dev/null || true
    LOG_FOLLOW_PID=""
    show_log "${log_file}"

    if [[ "${final_status}" != "SUCCEEDED" ]]; then
        log "ERROR: Ray job ${sub_id} ended with status ${final_status}." >&2
        diagnose_cluster
        exit 1
    fi
    verify_log "${verifier}" "${log_file}" "${verifier_args}"
    log "${sub_id} PASSED."
}

run_in_tpu_worker() {
    # Usage: run_in_tpu_worker <verifier> <timeout_mins> <verifier args> -- <command...>
    # Run a command in the (first) TPU worker container instead of as a Ray job, e.g. torchrun on a single-host slice:
    # copy this checkout and the plugin into the pod, run the command there with the environment of the Ray jobs, and
    # verify its output.
    local verifier="$1" timeout_mins="$2" verifier_args="$3"
    shift 4
    local worker remote=/tmp/verl-tpu-ci log_file rc=0
    worker="$(kubectl get pods -n "${NS}" -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
        -o jsonpath='{.items[0].metadata.name}')"
    log_file="${WORK_DIR}/${verifier}_in_worker_$(date +%Y%m%d_%H%M%S).log"
    local kexec=(kubectl exec -n "${NS}" "${worker}" -c ray-worker --)

    log "Copying the checkout and verl-hardware-plugin to ${worker}:${remote}"
    "${kexec[@]}" sh -c "rm -rf ${remote} && mkdir -p ${remote}/repo ${remote}/plugin"
    tar -C "${REPO_ROOT}" --exclude=.git --exclude=.venv -cf - . |
        kubectl exec -i -n "${NS}" "${worker}" -c ray-worker -- tar -xf - -C "${remote}/repo"
    tar -C "$(dirname "${PLUGIN_PKG}")" -cf - "$(basename "${PLUGIN_PKG}")" |
        kubectl exec -i -n "${NS}" "${worker}" -c ray-worker -- tar -xf - -C "${remote}/plugin"

    log "Running in ${worker}: $*"
    # shellcheck disable=SC2016  # expanded by the shell in the pod
    timeout "$((timeout_mins * 60))" "${kexec[@]}" bash -c \
        'cd "$0/repo" && export PYTHONPATH="$0/plugin:$0/repo:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 VERL_PLATFORM=tpu VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin && exec "$@"' \
        "${remote}" "$@" >"${log_file}" 2>&1 || rc=$?
    show_log "${log_file}"
    if [[ "${rc}" -ne 0 ]]; then
        log "ERROR: '$*' in ${worker} exited with code ${rc}." >&2
        diagnose_cluster
        exit 1
    fi
    verify_log "${verifier}" "${log_file}" "${verifier_args}"
    log "${verifier} in ${worker} PASSED."
}

# ---------------------------------------------------------------------------------------------------------------------
# Tests. Each runs on its own RayCluster; settings are passed to the example scripts as environment variables, like the
# examples' READMEs do.
# ---------------------------------------------------------------------------------------------------------------------

test_tpu_platform() {
    # torch_tpu basics, the same MLP on TPU and CPU, and the verl TPU platform of verl-hardware-plugin on 1 chip.
    submit_and_verify_ray_job smoke 15 "" -- python3 tests/special_tpu/tpu_smoke_test.py
}

test_sft() {
    # Qwen3-0.6B GSM8K multi-turn SFT, TorchTitan FSDP on one v6e-4 slice, in both launch modes of the script.
    # 1. Ray single controller (verl.trainer.sft_trainer_ray), as a Ray job.
    submit_and_verify_ray_job sft 30 "--steps 8" -- \
        env RAY_DATA_HOME="${TPU_CI_DATA_HOME}" SMOKE_TEST=1 NNODES_TRAINER=1 N_CHIPS_TRAINER=4 SFT_TRAINER_MODE=ray \
        bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
    # 2. torchrun (verl.trainer.sft_trainer, the script's default) in the TPU container. Not as a Ray job: a Ray driver
    #    that reserves all chips of a host gets Ray's host-default TPU environment (without TPU_HOST_BOUNDS and
    #    TPU_CHIPS_PER_HOST_BOUNDS), which does not describe a 2x2 subslice.
    run_in_tpu_worker sft 30 "--steps 8" -- \
        env RAY_DATA_HOME="${TPU_CI_DATA_HOME}" SMOKE_TEST=1 NNODES_TRAINER=1 N_CHIPS_TRAINER=4 \
        bash examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh
}

test_grpo() {
    # The 5-step Qwen3-0.6B GRPO smoke test of examples/tpu/grpo/README.md (2 x v6e-1): TorchTitan trainer on
    # tpu-group-0, vLLM rollout on tpu-group-1, V1 separate-async trainer, tpu-sync weight sync with its parity checks
    # (logged at INFO and required by the verifier).
    local JOB_RUNTIME_ENV
    JOB_RUNTIME_ENV="$(runtime_env_with VERL_LOGGING_LEVEL=INFO)"
    submit_and_verify_ray_job grpo 45 "--steps 5" -- \
        env RAY_DATA_HOME="${TPU_CI_DATA_HOME}" MODEL_SIZE=0.6B VERIFY_PARITY=all \
        bash examples/tpu/grpo/run_qwen3_torchtitan.sh
}

test_grpo_4b() {
    # The Qwen3-4B GRPO example (torch.compile, splash attention, SimpleFSDP, tpu-sync) for GRPO_4B_STEPS steps, with
    # validation before training and after the last step.
    local steps="${GRPO_4B_STEPS:-20}" JOB_RUNTIME_ENV
    warm_model_cache "${TPU_CI_DATA_HOME}/assets/hf/Qwen3-4B"
    JOB_RUNTIME_ENV="$(runtime_env_with VERL_LOGGING_LEVEL=INFO)"
    submit_and_verify_ray_job grpo_4b 120 "--steps ${steps}" -- \
        env RAY_DATA_HOME="${TPU_CI_DATA_HOME}" MODEL_SIZE=4B TOTAL_TRAINING_STEPS="${steps}" TEST_FREQ="${steps}" \
        VERIFY_PARITY=all bash examples/tpu/grpo/run_qwen3_torchtitan.sh
}

provision_test_cluster() {
    case "$1" in
        tpu_platform) provision_ray_cluster "$1" v6e-1 1 2 8Gi 32 128Gi ;;
        sft) provision_ray_cluster "$1" v6e-4 1 4 16Gi 96 400Gi ;;
        grpo) provision_ray_cluster "$1" v6e-1 2 4 16Gi 32 128Gi ;;
        grpo_4b) provision_ray_cluster "$1" v6e-4 2 8 32Gi 96 400Gi ;;
    esac
}

delete_test_cluster() {
    if [[ -n "${PF_PID}" ]]; then
        kill "${PF_PID}" 2>/dev/null || true
        PF_PID=""
    fi
    log "Deleting RayCluster ${RAY_CLUSTER_NAME}..."
    kubectl delete raycluster -n "${NS}" "${RAY_CLUSTER_NAME}" --wait=false --ignore-not-found || true
    RAY_CLUSTER_NAME="" HEAD_POD="" TPU_NODES=""
}

run_test() {
    local test_name="$1" attempt rc
    for ((attempt = 1; ; attempt++)); do
        rm -f "${WORK_DIR}/preempted"
        provision_test_cluster "${test_name}"
        wait_for_ray_cluster
        connect_ray_dashboard
        print_environment
        # Run the test in a subshell with errexit, so that its failure (`exit 1`) comes back here.
        set +e
        (
            set -e
            "test_${test_name}"
        )
        rc=$?
        set -e
        if [[ "${rc}" -eq 0 ]]; then
            return 0
        fi
        if [[ ! -f "${WORK_DIR}/preempted" || "${attempt}" -gt "${PREEMPTION_RETRIES}" ]]; then
            exit "${rc}"
        fi
        log "A TPU node of ${RAY_CLUSTER_NAME} was preempted; re-running ${test_name} on a new RayCluster" \
            "(retry ${attempt} of ${PREEMPTION_RETRIES})."
        delete_test_cluster
        CLUSTER_SUFFIX="-r${attempt}"
    done
}

mkdir -p "${WORK_DIR}"
case "${MODE}" in
    teardown)
        # Safety net of the workflows (e.g. a cancelled run): delete the RayClusters of this run.
        connect_gke_cluster
        selector="verl-ci/managed=true,verl-ci/run-id=${RUN_ID}"
        if [[ -n "${2:-}" ]]; then
            selector="${selector},verl-ci/test=${2//_/-}"
        fi
        log "Deleting the RayClusters matching ${selector}"
        kubectl delete raycluster -n "${NS}" -l "${selector}" --wait=false --ignore-not-found || true
        ;;
    build_image)
        connect_gke_cluster
        TPU_CI_ALLOW_IMAGE_BUILD=true
        ensure_ci_image
        ;;
    tpu_platform | sft | grpo | grpo_4b)
        connect_gke_cluster
        trap cleanup EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        reap_stale_clusters
        ensure_ray_cli
        ensure_ci_image
        fetch_hardware_plugin
        run_test "${MODE}"
        ;;
    *)
        echo "${USAGE}" >&2
        exit 1
        ;;
esac

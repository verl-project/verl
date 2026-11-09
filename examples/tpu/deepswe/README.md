# How to Prepare GKE and Run the DeepSWE Recipe in `verl`

To run the DeepSWE training recipe ([`run_qwen3_5_35b_deepswe.sh`](./run_qwen3_5_35b_deepswe.sh)), **your TPU machines and your sandbox CPU node pool must live in the same GKE cluster**. During rollout, the agent framework running on the Ray workers talks directly to the in-cluster Kubernetes API (`kubernetes.default.svc`) to claim R2E-Gym container sandboxes and stream bash/editor tool calls into those pods via `kubectl exec` (`pods/exec`).

Below is each resource you need to have in place, what it is used for, how it should be configured, and the commands to set it up.

---

## 1. GKE Cluster with GCSFuse and Workload Identity

### What it is for
The GKE cluster hosts both the Ray TPU job and the container sandboxes. It needs the **Cloud Storage FUSE CSI driver** (`GcsFuseCsiDriver`) so TPU worker pods can mount your GCS bucket at `/data` to read the model checkpoint and dataset, and **Workload Identity** so pods can authenticate to GCS and Artifact Registry without static keys.

### How it should look
- A Regional or Zonal GKE cluster in the same region as your TPU capacity and Artifact Registry mirror (for example, `europe-west4` for TPU v5p or `us-central1` for TPU v7x).
- Addons enabled: `GcsFuseCsiDriver` (and optionally `RayOperator` if using `RayCluster` CRDs).
- Workload Identity pool enabled: `<PROJECT>.svc.id.goog`.
- A small default/system CPU node pool (e.g., 1–2 `n2-standard-16` nodes) to host the Ray Head pod, which coordinates the cluster and requires ~12 vCPUs and 48 GiB of memory.

### Suggested commands
```bash
export PROJECT="<GCP_PROJECT>"
export REGION="<REGION>"          # e.g., europe-west4 or us-central1
export ZONE="<ZONE>"              # e.g., europe-west4-b
export CLUSTER="<CLUSTER_NAME>"
export NAMESPACE="trellis"        # or "default"

gcloud container clusters create "${CLUSTER}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --addons=RayOperator,GcsFuseCsiDriver \
  --workload-pool="${PROJECT}.svc.id.goog" \
  --machine-type=n2-standard-16 \
  --num-nodes=2

gcloud container clusters get-credentials "${CLUSTER}" --region="${REGION}" --project="${PROJECT}"
kubectl create namespace "${NAMESPACE}" || true
```

---

## 2. Dedicated Sandbox CPU Node Pool (`sandbox-cpu-pool`)

### What it is for
Each R2E-Gym rollout episode spins up an isolated repository container (such as `numpy`, `pillow`, `pandas`, or `tornado`) where the ReAct agent runs bash commands, edits code, and executes `pytest` suites. Running dozens of concurrent container sandboxes on a dedicated CPU node pool prevents their CPU and disk I/O bursts from starving the Ray head pod or interfering with the TPU host VMs.

### How it should look
- **Machine type & sizing**: CPU nodes with plenty of vCPUs and local disk (e.g., `n2-standard-32` or `n2-standard-64` with `200GB+` `pd-balanced` boot disks to cache multiple R2E-Gym container images). For 32–128 concurrent sandboxes (~2 vCPUs and 4 GiB RAM per sandbox), provision ~500–2,000 total vCPUs (e.g., 16–64 `n2-standard-32` nodes with autoscaling).
- **Node label**: `cloud.google.com/gke-nodepool=sandbox-cpu-pool` so `K8sAgentSandbox` can target this pool via `NODE_SELECTOR_KEY` and `NODE_SELECTOR_VAL`.
- **Node taint**: `cloud.google.com/gke-nodepool=sandbox-cpu-pool:NoSchedule` so regular cluster workloads do not land on sandbox nodes (`K8sAgentSandbox` automatically adds a matching toleration to every sandbox pod it creates).

### Suggested command
```bash
gcloud container node-pools create sandbox-cpu-pool \
  --cluster="${CLUSTER}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=n2-standard-32 \
  --num-nodes=16 \
  --enable-autoscaling --min-nodes=4 --max-nodes=64 \
  --disk-type=pd-balanced --disk-size=200GB \
  --node-labels="cloud.google.com/gke-nodepool=sandbox-cpu-pool" \
  --node-taints="cloud.google.com/gke-nodepool=sandbox-cpu-pool:NoSchedule" \
  --workload-metadata=GKE_METADATA \
  --scopes=cloud-platform
```

---

## 3. Two TPU Node Pools (1 Trainer Slice + 1 Rollout Slice)

### What it is for
`verl`'s `separate_async` trainer runs TorchTitan FSDP2 training on one TPU slice while `vLLM` generates rollouts on a second TPU slice, syncing weights between them after each step via `tpu-sync`. Having two separate TPU slices of equal size prevents training and inference from competing for HBM on the same chips.

### How they should look
- **Two identical TPU slices** in the same cluster: one for the trainer (`tpu-group-0`) and one for the rollout server (`tpu-group-1`).
- **For TPU v5p**: Two 16-chip slices (`ct5p-hightpu-4t`, topology `2x2x4`, 4 hosts x 4 chips per slice = 16 chips per slice, 32 chips total).
- **For TPU v7x**: Two 4-chip / 8-TensorCore slices (`tpu7x-standard-4t`, topology `2x2x1`, 1 host x 8 TensorCores per slice) or two 8-chip / 16-TensorCore slices (`2x2x2`, 2 hosts per slice).

### Suggested commands
```bash
# Option A: Two 16-chip TPU v5p slices (4 hosts per slice, 2x2x4 topology)
for SLICE in trainer-v5p-slice rollout-v5p-slice; do
  gcloud container node-pools create "${SLICE}" \
    --cluster="${CLUSTER}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --node-locations="${ZONE}" \
    --machine-type=ct5p-hightpu-4t \
    --tpu-topology=2x2x4 \
    --num-nodes=4 \
    --workload-metadata=GKE_METADATA \
    --scopes=cloud-platform
done

# Option B: Two TPU v7x slices (1 host per slice, 2x2x1 topology)
for SLICE in trainer-v7x-slice rollout-v7x-slice; do
  gcloud container node-pools create "${SLICE}" \
    --cluster="${CLUSTER}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --node-locations="${ZONE}" \
    --machine-type=tpu7x-standard-4t \
    --num-nodes=1 \
    --workload-metadata=GKE_METADATA \
    --scopes=cloud-platform
done
```

---

## 4. Agent Sandbox CRDs, Controller, and Namespace RBAC

### What it is for
`uni-agent`'s `K8sAgentSandbox` (`uni_agent/sandbox/k8s_agent_sandbox.py`) uses `agent_sandbox_rl` (`SandboxFleet`) to manage pre-warmed pools of R2E-Gym containers in Kubernetes. Under the hood, this relies on the `kubernetes-sigs/agent-sandbox` controller and four custom resources (`Sandbox`, `SandboxTemplate`, `SandboxWarmPool`, and `SandboxClaim`), plus RBAC permissions for the Ray worker's Kubernetes ServiceAccount to create those resources and exec into the claimed sandbox pods.

### How it should look
- **CRDs & Controller**: `sandboxes.agents.x-k8s.io`, `sandboxtemplates.extensions.agents.x-k8s.io`, `sandboxwarmpools.extensions.agents.x-k8s.io`, and `sandboxclaims.extensions.agents.x-k8s.io` installed in the cluster, with `agent-sandbox-controller` running in `agent-sandbox-system`.
- **RBAC (`Role` + `RoleBinding`)**: In the namespace where your Ray pods run (e.g., `trellis` or `default`), the ServiceAccount used by your Ray pods (`xpk-sa` or `default`) must have `get`, `list`, `watch`, `create`, `update`, `patch`, `delete` on `pods`, `services`, `sandboxes`, `sandboxtemplates`, `sandboxwarmpools`, and `sandboxclaims`, plus `get` and `create` on `pods/exec` and `pods/log`.

### Suggested commands
Check if the CRDs already exist, and install them if missing:
```bash
kubectl get crd sandboxes.agents.x-k8s.io \
  sandboxtemplates.extensions.agents.x-k8s.io \
  sandboxwarmpools.extensions.agents.x-k8s.io \
  sandboxclaims.extensions.agents.x-k8s.io || {
  export AGENT_SANDBOX_VERSION="v0.2.1"
  kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/manifest.yaml"
  kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/extensions.yaml"
}
```

Apply the RBAC permissions in your target namespace:
```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: xpk-sa
  namespace: ${NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: agent-sandbox-manager
  namespace: ${NAMESPACE}
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods/exec", "pods/log"]
    verbs: ["get", "create"]
  - apiGroups: ["agents.x-k8s.io"]
    resources: ["sandboxes"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["extensions.agents.x-k8s.io"]
    resources: ["sandboxclaims", "sandboxtemplates", "sandboxwarmpools"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: agent-sandbox-manager-binding
  namespace: ${NAMESPACE}
subjects:
  - kind: ServiceAccount
    name: xpk-sa
    namespace: ${NAMESPACE}
  - kind: ServiceAccount
    name: default
    namespace: ${NAMESPACE}
roleRef:
  kind: Role
  name: agent-sandbox-manager
  apiGroup: rbac.authorization.k8s.io
EOF
```

---

## 5. Regional Artifact Registry Mirror for R2E-Gym Images

### What it is for
Each row in the R2E-Gym dataset specifies a Docker Hub image (`namanjain12/<repo>_final:<commit>`). If dozens of GKE nodes pull those multi-gigabyte images directly from Docker Hub at once, Docker Hub blocks the pulls with `429 Too Many Requests` rate-limit errors. Hosting a mirror of the 8 R2E-Gym repositories (`aiohttp_final`, `datalad_final`, `numpy_final`, `pandas_final`, `pillow_final`, `pyramid_final`, `scrapy_final`, `tornado_final`) in a regional Artifact Registry repository lets `K8sAgentSandbox` rewrite `namanjain12/*` to your regional mirror via `SANDBOX_IMAGE_PREFIX`.

### How it should look
- An Artifact Registry Docker repository in the same region as your cluster (for example, `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix`) containing the mirrored `<repo>_final:<commit>` tags.
- Pass `SANDBOX_IMAGE_PREFIX="<REGION>-docker.pkg.dev/<PROJECT>/<REPO>"` when launching the job.

---

## 6. GCS Bucket with Model Checkpoint, Preprocessed Dataset, and Code

### What it is for
The TPU worker pods mount a GCS bucket at `/data` via GCSFuse. Storing the text-only `Qwen3.5-35B-A3B` checkpoint, the preprocessed R2E-Gym Parquet files (`train.parquet` and `test.parquet`), and the three code repositories on GCS makes them immediately available to every TPU node.

### How it should look
Under your GCS bucket `gs://<BUCKET>/<USER_DIR>` (which maps to `RAY_DATA_HOME=/data/<USER_DIR>` inside the pods):
- `assets/hf/Qwen3.5-35B-A3B/`: Text-only Hugging Face checkpoint (`*.safetensors` + tokenizer files).
- `data/r2e_gym_easy/train.parquet` and `data/r2e_gym_easy/test.parquet`: Generated by `uni_agent.tasks.r2e_gym.preprocess`.
- `src/deepswe_bundle.tar.gz`: Tarball of the `uni-agent`, `verl`, and `verl-hardware-plugin` checkouts so pods can extract them to `/tmp/deepswe_src` on startup.

### Suggested commands
```bash
export BUCKET="<GCS_BUCKET>"
export USER_DIR="aaskhat"

# 1. Clone the three PR branches
git clone -b feat/deepswe-qwen3_5-35b-v7x https://github.com/askhat-g/verl.git
git clone -b feat/deepswe-qwen3_5-35b-v7x https://github.com/askhat-g/verl-hardware-plugin.git
git clone -b feat/deepswe-qwen3_5-35b-v7x https://github.com/askhat-g/uni-agent.git

# 2. Preprocess R2E-Gym into Parquet files and upload to GCS
PYTHONPATH=uni-agent:verl python3 -m uni_agent.tasks.r2e_gym.preprocess \
  --local-save-dir /tmp/r2e_gym_easy
gcloud storage cp -r /tmp/r2e_gym_easy "gs://${BUCKET}/${USER_DIR}/data/"

# 3. Upload the text-only Qwen3.5-35B-A3B checkpoint to GCS
gcloud storage cp -r /path/to/Qwen3.5-35B-A3B "gs://${BUCKET}/${USER_DIR}/assets/hf/"

# 4. Bundle and upload the three repositories
tar -czf /tmp/deepswe_bundle.tar.gz \
  --exclude='.git' --exclude='__pycache__' \
  uni-agent verl verl-hardware-plugin
gcloud storage cp /tmp/deepswe_bundle.tar.gz "gs://${BUCKET}/${USER_DIR}/src/deepswe_bundle.tar.gz"
```

---

## 7. Multi-Slice Ray `JobSet` on GKE

### What it is for
The `JobSet` starts the Ray cluster across your pods: 1 Ray head pod on the CPU pool, 1 `trainer` replicated job on the first TPU slice, and 1 `rollout` replicated job on the second TPU slice.

### How it should look
- **Head pod**: Runs `ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=0 --num-gpus=0 --block` on the system CPU pool and extracts `/tmp/deepswe_src`.
- **Trainer and Rollout TPU worker pods**:
  1. Mount `gs://<BUCKET>` at `/data` via `gcsfuse.csi.storage.gke.io` and a `64Gi` in-memory `emptyDir` at `/dev/shm`.
  2. **Unset `MEGASCALE_*` variables before `ray start`**: GKE's TPU `JobSet` webhook automatically injects `MEGASCALE_COORDINATOR_ADDRESS` and `MEGASCALE_NUM_SLICES` across all jobs in a `JobSet`. Unsetting `MEGASCALE_*`, `TPU_WORKER_HOSTNAMES`, and `TPU_WORKER_ID` before launching `ray start` ensures the trainer slice and rollout slice form two independent `libtpu` meshes instead of waiting for a single combined multi-slice barrier.
  3. **Pre-warm checkpoint shards**: Sequentially read `*.safetensors` from `/data/.../Qwen3.5-35B-A3B` into the OS page cache before starting Ray so multi-process `mmap` page faults do not trigger slow random reads over GCSFuse.
  4. **Advertise slice affinity**: Pass `--resources='{"TPU": <CHIPS_PER_HOST>, "tpu-group-0": 1}'` on the trainer pods and `"tpu-group-1": 1` on the rollout pods so `verl-hardware-plugin` places the trainer and rollout worker groups on separate slices.

### Suggested commands
Apply your `JobSet` manifest and verify all pods join Ray:
```bash
kubectl apply -f verl-nap-aaskhat-deepswe-jobset.yaml
kubectl get pods -n "${NAMESPACE}" -l jobset.sigs.k8s.io/jobset-name=verl-nap-aaskhat-deepswe

HEAD_POD=$(kubectl get pods -n "${NAMESPACE}" \
  -l jobset.sigs.k8s.io/jobset-name=verl-nap-aaskhat-deepswe,verl-nap/role=head \
  -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- ray status
```

---

## 8. Running the Recipe in `verl`

### What it is for
[`run_qwen3_5_35b_deepswe.sh`](./run_qwen3_5_35b_deepswe.sh) wires `uni-agent`'s `AgentFrameworkRolloutAdapter` (`ReActAgent` + `r2e_gym` + `K8sAgentSandbox`) into `verl`'s V1 async PPO/RLOO trainer ([`run_qwen3_5_torchtitan.sh`](../grpo/run_qwen3_5_torchtitan.sh)).

### How the launch command should look
- Because `ray job submit` starts in `/app` with a clean environment on the head pod, pass your `export` statements inside `bash -c "..."` so both the shell script and `ray.init(runtime_env=...)` inherit them.
- Set `SANDBOX_NAMESPACE`, `SANDBOX_IMAGE_PREFIX`, `NODE_SELECTOR_KEY=cloud.google.com/gke-nodepool`, and `NODE_SELECTOR_VAL=sandbox-cpu-pool` so `K8sAgentSandbox` provisions warm pools in your namespace on the sandbox CPU pool using the mirrored images.
- When running on **TPU v5p**, set `TPU_ACCELERATOR_TYPE=v5p` and `TORCH_TPU_TOPOLOGY=2,2,4` (`NNODES_TRAINER=4`, `N_CHIPS_TRAINER=4`, `NNODES_ROLLOUT=4`, `N_CHIPS_ROLLOUT=4`). The script automatically adds the required TPU v5p XLA/libtpu flags (`--xla_tpu_scoped_vmem_limit_kib=32768`, `USE_MOE_SPARSE_CORE=0`, and `--slicebuilder_use_insecure_grpc=true`).
- When running on **TPU v7x**, set `TPU_ACCELERATOR_TYPE=tpu7x` (`NNODES_TRAINER=1`, `N_CHIPS_TRAINER=8`, `NNODES_ROLLOUT=1`, `N_CHIPS_ROLLOUT=8` for `2x2x1` slices).

### Suggested command (TPU v5p)
```bash
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- \
  ray job submit --address=http://127.0.0.1:8265 --no-wait -- \
  bash -c "
    cd /tmp/deepswe_src/verl &&
    export PYTHONPATH='/tmp/deepswe_src/uni-agent:/tmp/deepswe_src/verl:/tmp/deepswe_src/verl-hardware-plugin:/torchtitan:/torchtitan/torchtitan' &&
    export RAY_DATA_HOME='/data/${USER_DIR}' &&
    export UNI_AGENT_HOME='/tmp/deepswe_src/uni-agent' &&
    export PROJECT_NAME='verl-tpu-deepswe' &&
    export EXPERIMENT_NAME='qwen3_5_35b_deepswe_r2e_gym_v5p_20steps' &&
    export TPU_ACCELERATOR_TYPE='v5p' &&
    export TORCH_TPU_TOPOLOGY='2,2,4' &&
    export NNODES_TRAINER=4 &&
    export N_CHIPS_TRAINER=4 &&
    export NNODES_ROLLOUT=4 &&
    export N_CHIPS_ROLLOUT=4 &&
    export TOTAL_TRAINING_STEPS=20 &&
    export TEST_FREQ=20 &&
    export VAL_BEFORE_TRAIN=False &&
    export VAL_MAX_SAMPLES=16 &&
    export TRAIN_BATCH_SIZE=4 &&
    export PPO_MINI_BATCH_SIZE=4 &&
    export ROLLOUT_N=4 &&
    export MAX_PROMPT_LEN=4096 &&
    export MAX_RESPONSE_LEN=12288 &&
    export MAX_MODEL_LEN=16384 &&
    export PAD_TO_LENGTH_BUCKET=2048 &&
    export SANDBOX_NAMESPACE='${NAMESPACE}' &&
    export SANDBOX_IMAGE_PREFIX='europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix' &&
    export NODE_SELECTOR_KEY='cloud.google.com/gke-nodepool' &&
    export NODE_SELECTOR_VAL='sandbox-cpu-pool' &&
    exec bash /tmp/deepswe_src/verl/examples/tpu/deepswe/run_qwen3_5_35b_deepswe.sh \
      \"trainer.logger=['console','wandb']\" \
      \"+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='\${PYTHONPATH}'\"
  "
```

### Suggested commands to monitor the run
```bash
# Follow job logs (prints the WandB URL once workers initialize)
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- \
  ray job logs --address=http://127.0.0.1:8265 -f <JOB_ID>

# Check warm pools and active sandbox claims in your namespace
kubectl get sandboxwarmpools,sandboxclaims -n "${NAMESPACE}"
```

## 9. Optimized TPU v7x Configuration

`run_qwen3_5_35b_deepswe_v7x_optimized.sh` wraps the recipe above with the settings measured fastest on TPU v7x:
a 2x2x2 trainer slice and three 2x2x1 rollout slices, 32k context (8192 prompt + 24576 response), and
16 issues x 16 attempts = 256 trajectories per step.

| Configuration | Steady-state step time |
| --- | --- |
| Recipe defaults (MAX_NUM_SEQS 64, CONCURRENCY 256, half-batch prefetch) | ~490 s |
| + MAX_NUM_SEQS 128, CONCURRENCY 384, full-batch prefetch, skip-padded MoE, GDN dynamic tiling | ~370 s |
| + rollout expert parallelism (EP 8) with the fused EP MoE kernel | ~320 s |

Step times alternate between a long step (waits for a full rollout) and a short step (its batch was prefetched while
the previous step trained); the numbers above average consecutive pairs from step 3 on. The rollout is the critical
path; the trainer phase (old log-probs + update) is ~115 s per step.

Rollout correction is the base recipe's sequence rejection (`rollout_rs=seq_mean_k1`, a keep-band on the geometric-mean
`pi_train / pi_rollout` ratio) with the band's lower edge widened from 0.999 to 0.998: on this stack the trainer and
rollout log-probs differ by a small systematic bias (`rollout_corr/log_ppl_diff` ~ +0.0009), which makes the 0.999 edge
reject ~45% of sequences; 0.998 brings that to ~5% at no speed cost. Set `ROLLOUT_RS_THRESHOLD=0.999_1.002` to use the
base recipe's band.

The script needs the verl-hardware-plugin TPU engine options `gdn_row_slot_len`, `splash_max_block_size` and
`frozen_param_regex`, and a vLLM-TorchTPU build with `USE_MOE_FUSED_EP_KERNEL` (the Dockerfile's default
`VLLM_TORCHTPU_COMMIT` has it). Build the image from `examples/tpu/grpo/Dockerfile` with
`--build-arg TORCHTITAN_COMMIT=4e16e67aed8c1e702e6f440b1ff81619bb6873d4` (google-pytorch/torchtitan#142, as for
Qwen3.5-397B in `examples/tpu/grpo/README.md`): `gdn_row_slot_len` uses its `gated_delta_document_resets`, which the
Dockerfile's default torchtitan commit does not have. The numbers above were measured with that build. Launch it the same way as the base
recipe (section 8), after setting the dataset, model and sandbox variables:

```bash
ray job submit --address=http://127.0.0.1:8265 --no-wait -- \
  bash examples/tpu/deepswe/run_qwen3_5_35b_deepswe_v7x_optimized.sh
```

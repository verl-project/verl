# GRPO on Cloud TPU with `verl` + `verl-hardware-plugin`

GRPO on GSM8K with Qwen3 on Cloud TPU: a TorchTitan trainer on one slice, a vLLM rollout on the other, and `tpu-sync` weight sync between them. TPU support comes from [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin).

By default, this guide walks through the minimal **2-chip setup** (2 TPU `v6e-1` slices: 1 trainer chip and 1 rollout chip) for the Qwen3-0.6B smoke test. See [Section 6](#6-run-qwen3-4b-on-2-v6e-4-slices-8-chips) for scaling up to Qwen3-4B on 2 `v6e-4` slices (8 chips) or 2 `v6e-8` slices (16 chips).

| File | |
|---|---|
| [`run_qwen3_0_6b_torchtitan.sh`](run_qwen3_0_6b_torchtitan.sh) | 5-step smoke test (Qwen3-0.6B, 2 `v6e-1` slices / 2 chips by default) |
| [`run_qwen3_4b_torchtitan.sh`](run_qwen3_4b_torchtitan.sh) | 100-step GRPO run (Qwen3-4B, 2 `v6e-4` slices / 8 chips) |
| [`run_qwen3_5_torchtitan.sh`](run_qwen3_5_torchtitan.sh) | Qwen3.5 GRPO on TPU v7x (default `Qwen3.5-35B-A3B` MoE; also `Qwen3.5-0.8B`) |
| [`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml) | KubeRay `RayCluster`: 1 CPU head pod and 2 TPU `v6e-1` slices (configurable for `v6e-4` / `v6e-8`) |
| [`Dockerfile`](Dockerfile) | Image of the Ray pods |

You need `gcloud`, `kubectl`, Docker with Buildx, the Ray CLI (`pip install "ray[default]>=2.56"`), the Hugging Face CLI (`pip install -U huggingface_hub`), and verl's Python dependencies to preprocess the data (see the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html)).

## 1. Get the code

```bash
git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/verl-hardware-plugin.git
# 10/27/2026 HEAD of verl-hardware-plugin
git -C verl-hardware-plugin checkout 1eb3a61c6d3773e4f7f5f3c18e60c71e19ca2016
cd verl
```

Run all the commands below from the root of `verl`.

## 2. Build the image

```bash
docker buildx build -t <IMAGE> --push examples/tpu/grpo
```

## 3. Upload the model and the data

The Ray pods mount a GCS bucket `<BUCKET>` at `/data`. The scripts read models from `/data/assets/hf/` and GSM8K from `/data/data/gsm8k/`. Download and preprocess them outside the `verl` checkout (Ray uploads the checkout with every job), then copy them to the bucket:

```bash
hf download Qwen/Qwen3-0.6B --local-dir /tmp/Qwen3-0.6B
PYTHONPATH=. python examples/data_preprocess/gsm8k.py --local_save_dir /tmp/gsm8k
gcloud storage cp -r /tmp/Qwen3-0.6B gs://<BUCKET>/assets/hf/
gcloud storage cp -r /tmp/gsm8k gs://<BUCKET>/data/
```

To keep them under another directory of the bucket, run the scripts with `RAY_DATA_HOME=/data/<DIR>` (step 5).

## 4. Create the Ray cluster (2 `v6e-1` slices, 2 chips)

Create a GKE cluster with the Ray operator, Cloud Storage FUSE and Workload Identity, and a node pool with 2 `ct6e-standard-1t` nodes; each node is a single-chip `v6e-1` slice (1 slice for the trainer, 1 slice for the rollout). The default CPU node pool hosts the Ray head pod, which requests 12 CPUs and 48 GiB of memory.

```bash
gcloud container clusters create <CLUSTER> --region <REGION> --node-locations <ZONE> \
  --addons=RayOperator,GcsFuseCsiDriver --workload-pool=<PROJECT>.svc.id.goog \
  --machine-type n2-standard-16 --num-nodes 1
gcloud container node-pools create <NODE_POOL> --cluster <CLUSTER> --region <REGION> --node-locations <ZONE> \
  --machine-type ct6e-standard-1t --num-nodes 2 --spot --disk-type hyperdisk-balanced \
  --workload-metadata GKE_METADATA --scopes cloud-platform
gcloud container clusters get-credentials <CLUSTER> --region <REGION>
```

The Ray pods read the bucket as a Google service account `<GSA_EMAIL>` (create one with `gcloud iam service-accounts create <NAME>`; its email is `<NAME>@<PROJECT>.iam.gserviceaccount.com`), through the Kubernetes service account `ray-ksa` that the YAML creates:

```bash
gcloud storage buckets add-iam-policy-binding gs://<BUCKET> \
  --member serviceAccount:<GSA_EMAIL> --role roles/storage.objectViewer
gcloud iam service-accounts add-iam-policy-binding <GSA_EMAIL> \
  --member "serviceAccount:<PROJECT>.svc.id.goog[default/ray-ksa]" --role roles/iam.workloadIdentityUser
```

In [`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml), replace `<IMAGE>` (step 2), `<BUCKET>` (step 3) and `<GSA_EMAIL>`. Then create the Ray cluster and forward its dashboard port:

```bash
kubectl apply -f examples/tpu/grpo/ray-tpu-v6e-2slice.yaml
kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster  # wait for 3 pods at 2/2 Running
kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 &
```

## 5. Run the Qwen3-0.6B smoke test (2 chips)

Ray ships `verl` and `verl-hardware-plugin` (from `../verl-hardware-plugin`) to the cluster with every job:

```bash
export RAY_ADDRESS=http://localhost:23333
RUNTIME_ENV='{"py_modules": ["../verl-hardware-plugin/verl_hardware_plugin"], "excludes": [".git", ".venv"],
  "env_vars": {"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1", "VERL_PLATFORM": "tpu",
    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"}}'

ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
```

The 5-step smoke test takes about 5-6 minutes to start up and log the first step metric, then ~2 s per step once compiled. Follow the job with `ray job logs -f <JOB_ID>`.

The scripts take their settings from environment variables, e.g. `-- RAY_DATA_HOME=/data/<DIR> TOTAL_TRAINING_STEPS=10 bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh`, and pass extra arguments to `verl.trainer.main_ppo` as Hydra overrides. Fused QKV projection (`FUSE_QKV=True`) is enabled by default (`FUSE_QKV=False` disables it).

## 6. Run Qwen3-4B on 2 `v6e-4` slices (8 chips)

`run_qwen3_4b_torchtitan.sh` runs a 100-step GRPO job on 2 `v6e-4` slices (4 trainer chips + 4 rollout chips).

1. **Upload the 4B model** to your GCS bucket:
   ```bash
   hf download Qwen/Qwen3-4B --local-dir /tmp/Qwen3-4B
   gcloud storage cp -r /tmp/Qwen3-4B gs://<BUCKET>/assets/hf/
   ```

2. **Create a `v6e-4` node pool** with 2 `ct6e-standard-4t` nodes (4 chips per node):
   ```bash
   gcloud container node-pools create <V6E4_NODE_POOL> --cluster <CLUSTER> --region <REGION> --node-locations <ZONE> \
     --machine-type ct6e-standard-4t --num-nodes 2 --spot --disk-type hyperdisk-balanced \
     --workload-metadata GKE_METADATA --scopes cloud-platform
   ```

3. **Update [`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml)** under `workerGroupSpecs[0].template.spec` to request 4 TPU chips (`2x2` topology) per worker pod:
   ```yaml
   containers:
     - name: ray-worker
       ...
       resources:
         limits:
           memory: 400Gi
           google.com/tpu: "4"
         requests:
           cpu: "48"
           memory: 400Gi
           google.com/tpu: "4"
   ...
   nodeSelector:
     cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
     cloud.google.com/gke-tpu-topology: 2x2
   ```
   Then recreate the Ray cluster and port-forward its dashboard:
   ```bash
   kubectl delete -f examples/tpu/grpo/ray-tpu-v6e-2slice.yaml
   kubectl apply -f examples/tpu/grpo/ray-tpu-v6e-2slice.yaml
   kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster  # wait for 3 pods at 2/2 Running
   kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 &
   ```

4. **Submit the 4B job** (or run the 0.6B smoke test across all 8 chips):
   ```bash
   # 100-step Qwen3-4B run on 2 v6e-4 slices (defaults to N_CHIPS_TRAINER=4 N_CHIPS_ROLLOUT=4):
   ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh

   # Optional: 5-step Qwen3-0.6B smoke test on 2 v6e-4 slices:
   ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
     N_CHIPS_TRAINER=4 N_CHIPS_ROLLOUT=4 bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
   ```
   The 4B run takes about 10 minutes to start, then about 18-27 s per step (~22 s average over 100 steps).

To scale further to **2 `v6e-8` slices (16 chips)**, create two multi-host node pools with `--machine-type ct6e-standard-4t --num-nodes 2 --tpu-topology 2x4`, set `numOfHosts: 2` and `cloud.google.com/gke-tpu-topology: 2x4` (with `memory: 400Gi`, `cpu: "48"`, `google.com/tpu: "4"`) in [`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml), and run with `NNODES_TRAINER=2 N_CHIPS_TRAINER=4 NNODES_ROLLOUT=2 N_CHIPS_ROLLOUT=4`.

## 7. Qwen3.5 on TPU v7x

[`run_qwen3_5_torchtitan.sh`](run_qwen3_5_torchtitan.sh) runs GRPO on GSM8K with Qwen3.5 (default `Qwen3.5-35B-A3B` MoE) on TPU v7x, with one trainer slice and one rollout slice (e.g. `2x2x1` each). `MODEL_PATH` must be a text-only Qwen3.5 checkpoint (`Qwen3_5[Moe]ForCausalLM`, i.e. the language model of the Hugging Face checkpoint without the vision encoder). Submit it as in step 5:

```bash
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env MODEL_NAME=Qwen3.5-35B-A3B bash examples/tpu/grpo/run_qwen3_5_torchtitan.sh
```

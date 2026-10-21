# GRPO on Cloud TPU with `verl` + `verl-hardware-plugin`

GRPO on GSM8K with Qwen3 on 2 TPU v6e-4 slices (8 chips): a TorchTitan trainer on one slice, a vLLM rollout on the other, and `tpu-sync` weight sync between them. TPU support comes from [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin).

| File | |
|---|---|
| [`run_qwen3_0_6b_torchtitan.sh`](run_qwen3_0_6b_torchtitan.sh) | 5-step smoke test (Qwen3-0.6B) |
| [`run_qwen3_4b_torchtitan.sh`](run_qwen3_4b_torchtitan.sh) | 100-step GRPO run (Qwen3-4B) |
| [`ray-tpu-v6e4-2slice.yaml`](ray-tpu-v6e4-2slice.yaml) | KubeRay `RayCluster`: 1 CPU head pod and 2 TPU v6e-4 slices |
| [`Dockerfile`](Dockerfile) | Image of the Ray pods |

You need `gcloud`, `kubectl`, Docker with Buildx, the Ray CLI (`pip install "ray[default]>=2.56"`), the Hugging Face CLI (`pip install -U huggingface_hub`), and verl's Python dependencies to preprocess the data (see the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html)).

## 1. Get the code

```bash
git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl
```

Run all the commands below from the root of `verl`.

## 2. Build the image

```bash
docker buildx build -t <IMAGE> --push examples/tpu/grpo
```

## 3. Upload the models and the data

The Ray pods mount a GCS bucket `<BUCKET>` at `/data`. The scripts read the models from `/data/assets/hf/` and GSM8K from `/data/data/gsm8k/`. Download and preprocess them outside the `verl` checkout (Ray uploads the checkout with every job), then copy them to the bucket:

```bash
hf download Qwen/Qwen3-0.6B --local-dir /tmp/Qwen3-0.6B
hf download Qwen/Qwen3-4B --local-dir /tmp/Qwen3-4B
PYTHONPATH=. python examples/data_preprocess/gsm8k.py --local_save_dir /tmp/gsm8k
gcloud storage cp -r /tmp/Qwen3-0.6B /tmp/Qwen3-4B gs://<BUCKET>/assets/hf/
gcloud storage cp -r /tmp/gsm8k gs://<BUCKET>/data/
```

To keep them under another directory of the bucket, run the scripts with `RAY_DATA_HOME=/data/<DIR>` (step 5).

## 4. Create the Ray cluster

Create a GKE cluster with the Ray operator, Cloud Storage FUSE and Workload Identity, and a node pool with 2 `ct6e-standard-4t` nodes; each node is a v6e-4 slice. The default node pool hosts the Ray head pod, which requests 12 CPUs and 48 GiB of memory.

```bash
gcloud container clusters create <CLUSTER> --region <REGION> --node-locations <ZONE> \
  --addons=RayOperator,GcsFuseCsiDriver --workload-pool=<PROJECT>.svc.id.goog \
  --machine-type n2-standard-16 --num-nodes 1
gcloud container node-pools create <NODE_POOL> --cluster <CLUSTER> --region <REGION> --node-locations <ZONE> \
  --machine-type ct6e-standard-4t --num-nodes 2 --spot --disk-type hyperdisk-balanced \
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

In [`ray-tpu-v6e4-2slice.yaml`](ray-tpu-v6e4-2slice.yaml), replace `<IMAGE>` (step 2), `<BUCKET>` (step 3) and `<GSA_EMAIL>`. Then create the Ray cluster and forward its dashboard port:

```bash
kubectl apply -f examples/tpu/grpo/ray-tpu-v6e4-2slice.yaml
kubectl get pods -l ray.io/cluster=ray-tpu-v6e4-cluster  # wait for 3 pods at 2/2 Running
kubectl port-forward svc/ray-tpu-v6e4-cluster-head-svc 23333:8265 &
```

## 5. Run

Ray ships `verl` and `verl-hardware-plugin` (from `../verl-hardware-plugin`) to the cluster with every job:

```bash
export RAY_ADDRESS=http://localhost:23333
RUNTIME_ENV='{"py_modules": ["../verl-hardware-plugin/verl_hardware_plugin"], "excludes": [".git", ".venv"],
  "env_vars": {"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1", "VERL_PLATFORM": "tpu",
    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin", "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"}}'

ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
```

The smoke test takes about 10 minutes. The 4B run needs about 10 minutes to start, then about 30-40 s per step. Follow a job with `ray job logs -f <JOB_ID>`.

The scripts take their settings from environment variables, e.g. `-- RAY_DATA_HOME=/data/<DIR> TOTAL_TRAINING_STEPS=5 bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh`, and pass extra arguments to `verl.trainer.main_ppo` as Hydra overrides.

To use 2 v6e-8 slices (16 chips) instead, create two node pools with `--num-nodes 2 --tpu-topology 2x4`, set `numOfHosts: 2` and `cloud.google.com/gke-tpu-topology: 2x4` in the YAML, and run the scripts with `NNODES_TRAINER=2 NNODES_ROLLOUT=2`.

# GRPO on Cloud TPU with `verl` + `verl-hardware-plugin`

GRPO on GSM8K with Qwen3 on Cloud TPU: a TorchTitan trainer on one slice, a vLLM rollout on the other, and `tpu-sync` weight sync between them. TPU support comes from [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin).

By default, this guide walks through the minimal **2-chip setup** (2 TPU `v6e-1` slices: 1 trainer chip and 1 rollout chip) for the Qwen3-0.6B smoke test. See [Section 6](#6-scale-dense-qwen3-17b-4b-8b-14b-32b) for scaling [`run_qwen3_torchtitan.sh`](run_qwen3_torchtitan.sh) across all dense Qwen3 sizes (`0.6B`, `1.7B`, `4B`, `8B`, `14B`, `32B`) on `v6e-4`, `v6e-8`, `v6e-16` and `v6e-32` slices.

| File | |
|---|---|
| [`run_qwen3_torchtitan.sh`](run_qwen3_torchtitan.sh) | Dense Qwen3 GRPO (`MODEL_SIZE=0.6B \| 1.7B \| 4B \| 8B \| 14B \| 32B`; defaults to the 5-step `0.6B` smoke test on 2 `v6e-1` slices) |
| [`run_qwen3_5_torchtitan.sh`](run_qwen3_5_torchtitan.sh) | Qwen3.5 GRPO on TPU v7x (default `Qwen3.5-35B-A3B` MoE; also `Qwen3.5-0.8B`) |
| [`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml) | KubeRay `RayCluster`: 1 CPU head pod and 2 TPU `v6e-1` slices (configurable for `v6e-4` / `v6e-8` / `v6e-16` / `v6e-32`) |
| [`Dockerfile`](Dockerfile) | Image of the Ray pods |
| [`run_qwen3_30b_a3b_torchtitan.sh`](run_qwen3_30b_a3b_torchtitan.sh) | Qwen3-30B-A3B (MoE) GRPO on TPU (default v7x `2x2x1` + `2x2x1`) |

You need `gcloud`, `kubectl`, Docker with Buildx, the Ray CLI (`pip install "ray[default]>=2.56"`), the Hugging Face CLI (`pip install -U huggingface_hub`), and verl's Python dependencies to preprocess the data (see the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html)).

## 1. Get the code

```bash
git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/verl-hardware-plugin.git
# 11/05/2026 HEAD of verl-hardware-plugin
git -C verl-hardware-plugin checkout 0bc8bbac944e753686f89718c0c5eac61e84b912
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

Ray ships `verl` and `verl-hardware-plugin` (from `../verl-hardware-plugin`) to the cluster with every job. [`run_qwen3_torchtitan.sh`](run_qwen3_torchtitan.sh) defaults to `MODEL_SIZE=0.6B`:

```bash
export RAY_ADDRESS=http://localhost:23333
RUNTIME_ENV='{"py_modules": ["../verl-hardware-plugin/verl_hardware_plugin"], "excludes": [".git", ".venv"],
  "env_vars": {"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1", "VERL_PLATFORM": "tpu",
    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"}}'

ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- bash examples/tpu/grpo/run_qwen3_torchtitan.sh
```

The 5-step smoke test takes about 5-6 minutes to start up and log the first step metric, then ~2 s per step once compiled. Follow the job with `ray job logs -f <JOB_ID>`.

The script selects its dense Qwen3 preset with `MODEL_SIZE=0.6B | 1.7B | 4B | 8B | 14B | 32B` (default `0.6B`), takes settings from environment variables (e.g. `ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- env RAY_DATA_HOME=/data/<DIR> TOTAL_TRAINING_STEPS=10 bash examples/tpu/grpo/run_qwen3_torchtitan.sh`), and passes extra arguments to `verl.trainer.main_ppo` as Hydra overrides. Fused QKV projection (`FUSE_QKV=True`) is enabled by default (`FUSE_QKV=False` disables it).

## 6. Scale dense Qwen3 (1.7B, 4B, 8B, 14B, 32B)

[`run_qwen3_torchtitan.sh`](run_qwen3_torchtitan.sh) supports all six dense Qwen3 sizes via `MODEL_SIZE`. Each preset runs a TorchTitan SimpleFSDP trainer across one TPU slice and vLLM rollout across the other. To fit within the 32 GiB of HBM per `v6e` chip, the `8B`, `14B`, and `32B` presets train on 1 sequence per micro-batch and turn the KL-to-reference loss off (`USE_KL_LOSS=False`; the reference model would keep a second fp32 copy of the weights on every trainer chip).

| `MODEL_SIZE` | Slices | GKE node pool(s) | [`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml) | Trainer | Rollout | `USE_KL_LOSS` | Default steps |
|---|---|---|---|---|---|---|---|
| `0.6B` (default) | 2 `v6e-1` (2 chips) | 1 pool: `ct6e-standard-1t --num-nodes 2` | `numOfHosts: 1`, `1x1`, `128Gi`, `cpu: "32"`, `tpu: "1"` | FSDP1 | 1 x TP1 | `True` | 5 |
| `1.7B` | 2 `v6e-4` (8 chips) | 1 pool: `ct6e-standard-4t --num-nodes 2` | `numOfHosts: 1`, `2x2`, `400Gi`, `cpu: "48"`, `tpu: "4"` | FSDP4 | 1 x TP4 | `True` | 100 |
| `4B` | 2 `v6e-4` (8 chips) | 1 pool: `ct6e-standard-4t --num-nodes 2` | `numOfHosts: 1`, `2x2`, `400Gi`, `cpu: "48"`, `tpu: "4"` | FSDP4 | 1 x TP4 | `True` | 100 |
| `8B` | 2 `v6e-8` (16 chips) | 2 pools (1/slice): `ct6e-standard-4t --num-nodes 2 --tpu-topology 2x4` | `numOfHosts: 2`, `2x4`, `400Gi`, `cpu: "48"`, `tpu: "4"` | FSDP8 | 1 x TP8 | `False` | 100 |
| `14B` | 2 `v6e-16` (32 chips) | 2 pools (1/slice): `ct6e-standard-4t --num-nodes 4 --tpu-topology 4x4` | `numOfHosts: 4`, `4x4`, `400Gi`, `cpu: "48"`, `tpu: "4"` | FSDP16 | 4 x TP4 | `False` | 100 |
| `32B` | 2 `v6e-32` (64 chips) | 2 pools (1/slice): `ct6e-standard-4t --num-nodes 8 --tpu-topology 4x8` | `numOfHosts: 8`, `4x8`, `600Gi`, `cpu: "160"`, `tpu: "4"` | FSDP32 | 1 x TP32 | `False` | 20 |

`Qwen3-14B` has 40 attention heads (not divisible by 16), so its rollout defaults to one single-host `TP=4` vLLM replica per rollout host (`4 x TP4` on `v6e-16`).

1. **Upload the checkpoint** for your chosen `<SIZE>` (`1.7B`, `4B`, `8B`, `14B`, or `32B`) to your GCS bucket:
   ```bash
   SIZE=4B  # 1.7B | 4B | 8B | 14B | 32B
   hf download "Qwen/Qwen3-${SIZE}" --local-dir "/tmp/Qwen3-${SIZE}"
   gcloud storage cp -r "/tmp/Qwen3-${SIZE}" gs://<BUCKET>/assets/hf/
   ```
   Loading the 62 GB `Qwen3-32B` checkpoint through Cloud Storage FUSE is slow; to speed up start-up, stage it to the same pod-visible local path on every host (for example `/tmp/Qwen3-32B`, mounted from each host's `/tmp` in [`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml), or another mounted `hostPath`) and point `MODEL_PATH` at it.

2. **Create the TPU node pool(s)** matching the table above:
   - For single-host `v6e-4` (`1.7B` or `4B`), create one 2-node pool:
     ```bash
     gcloud container node-pools create <V6E4_NODE_POOL> --cluster <CLUSTER> --region <REGION> --node-locations <ZONE> \
       --machine-type ct6e-standard-4t --num-nodes 2 --spot --disk-type hyperdisk-balanced \
       --workload-metadata GKE_METADATA --scopes cloud-platform
     ```
   - For multi-host `v6e-8` (`8B`), `v6e-16` (`14B`), or `v6e-32` (`32B`), create **two** node pools (one per TPU slice) with the `--num-nodes` and `--tpu-topology` from the table:
     ```bash
     for SLICE in 0 1; do
       gcloud container node-pools create "<NODE_POOL>-${SLICE}" --cluster <CLUSTER> --region <REGION> --node-locations <ZONE> \
         --machine-type ct6e-standard-4t --num-nodes <NODES_PER_SLICE> --tpu-topology <TOPOLOGY> \
         --spot --disk-type hyperdisk-balanced --workload-metadata GKE_METADATA --scopes cloud-platform
     done
     ```

3. **Update [`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml)** under `workerGroupSpecs[0]` with the `numOfHosts`, `resources`, and `cloud.google.com/gke-tpu-topology` values from the table (also listed in the YAML comments), then recreate the Ray cluster and port-forward its dashboard:
   ```bash
   kubectl delete -f examples/tpu/grpo/ray-tpu-v6e-2slice.yaml
   kubectl apply -f examples/tpu/grpo/ray-tpu-v6e-2slice.yaml
   kubectl get pods -l ray.io/cluster=ray-tpu-v6e-cluster  # wait for all pods at 2/2 Running
   kubectl port-forward svc/ray-tpu-v6e-cluster-head-svc 23333:8265 &
   ```

4. **Submit the job** for the preset matching your cluster (`MODEL_SIZE=1.7B | 4B | 8B | 14B | 32B`; for example, `4B` on 2 `v6e-4` slices):
   ```bash
   ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
     env MODEL_SIZE=4B bash examples/tpu/grpo/run_qwen3_torchtitan.sh
   ```
   The `4B` run takes about 10 minutes to start, then about 18-27 s per step (~22 s average over 100 steps).

   Alternatively, to run the 5-step `0.6B` smoke test across a larger cluster (for example, 2 `v6e-4` slices), override the trainer and rollout topology:
   ```bash
   ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
     env MODEL_SIZE=0.6B N_CHIPS_TRAINER=4 N_CHIPS_ROLLOUT=4 bash examples/tpu/grpo/run_qwen3_torchtitan.sh
   ```

   `run_qwen3_torchtitan.sh` defaults to `TPU_EAGER_MODE=DEFER_AND_FUSE`. On an earlier image, `update_actor` hung at 32 trainer ranks with it; if a 32-rank (`32B`) run hangs in `update_actor`, set `TPU_EAGER_MODE=DEFER_NEVER`.

   Set `VERIFY_PARITY` to enable `tpu-sync` weight-sync verification (`engine_kwargs.tpu-sync.verify_parity`):

   | `VERIFY_PARITY` | Check |
   |---|---|
   | `off` (default) | None |
   | `norm` | After every sync, compare the weight norms of the trainer and the rollout |
   | `exact` | On the step-0 sync, compare every received tensor bit for bit with the checkpoint vLLM loaded |
   | `all` | `norm` and `exact` |

   For example, `ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- env MODEL_SIZE=8B VERIFY_PARITY=norm TOTAL_TRAINING_STEPS=10 bash examples/tpu/grpo/run_qwen3_torchtitan.sh`.

## 7. Qwen3.5 on TPU v7x

[`run_qwen3_5_torchtitan.sh`](run_qwen3_5_torchtitan.sh) runs GRPO on GSM8K with Qwen3.5 (default `Qwen3.5-35B-A3B` MoE) on TPU v7x, with one trainer slice and one rollout slice (e.g. `2x2x1` each). `MODEL_PATH` must be a text-only Qwen3.5 checkpoint (`Qwen3_5[Moe]ForCausalLM`, i.e. the language model of the Hugging Face checkpoint without the vision encoder). Submit it as in step 5:

```bash
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env MODEL_NAME=Qwen3.5-35B-A3B bash examples/tpu/grpo/run_qwen3_5_torchtitan.sh
```

**Graph-mode actor update (optional).** With a `verl-hardware-plugin` whose `engine_kwargs.tpu.graph_trainer` takes a settings dict (e.g. `git -C ../verl-hardware-plugin checkout 8a9e485da66ea76d40b01b4940db08928fc45061`), `TRAINER_MODE=graph_trainer` runs the actor update through TorchTitan's `TpuGraphTrainer`. It needs `USE_KL_LOSS=False`, `EXPERT_PARALLEL_SIZE=1` and one micro-batch per rank (`USE_DYNAMIC_BSZ=True` with `STATIC_MICRO_BATCH_LEN` = `MAX_TOKEN_LEN_PER_GPU`). `GRAPH_TRAINER_SPLIT_LAYERS=2` compiles two layers per forward / backward executable instead of one step graph, as TorchTitan's `qwen35_moe_35b_split_tokamax_benchmark` SFT recipe does; without it the 40-layer step compiles very slowly.

```bash
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env MODEL_NAME=Qwen3.5-35B-A3B TRAINER_MODE=graph_trainer GRAPH_TRAINER_SPLIT_LAYERS=2 \
  USE_KL_LOSS=False PARAM_DTYPE=bfloat16 OPTIM_IMPL=fused SPARSECORE_OFFLOAD=True TRAINER_SCOPED_VMEM_LIMIT_KIB=65472 \
  TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=4 \
  USE_DYNAMIC_BSZ=True MAX_TOKEN_LEN_PER_GPU=32768 STATIC_MICRO_BATCH_LEN=32768 STATIC_MAX_RESPONSE_LEN=2048 \
  bash examples/tpu/grpo/run_qwen3_5_torchtitan.sh
```

**Best measured graph-mode configuration (Qwen3.5-35B-A3B, v7x `2x2x1` trainer + `2x2x1` rollout).** This adds the Tokamax KDA kernels for the GatedDeltaNet layers (`MODEL_KERNELS=moe,kda,lce`) and the trainer libtpu flags of TorchTitan's `qwen35_moe_35b_split_tokamax_benchmark` recipe to the command above. Measured: `update_actor` 4.45-4.51 s per step and `perf/mfu/actor` 0.104 on average (0.085-0.128, it follows the real token count of the step) over 10 steps, against 21.96 s / 0.0107 for the best eager configuration; the first step traces and compiles in about 220 s. It needs:

1. The image built with the TorchTitan commit the numbers were measured with (`tpu_flags` and the Tokamax KDA kernels): `docker buildx build -t <IMAGE> --push --build-arg TORCHTITAN_COMMIT=c56f441fca0f3349014622d02cc2165e25957d95 examples/tpu/grpo`.
2. A `verl-hardware-plugin` with the graph-mode KDA document budget, [verl-project/verl-hardware-plugin#184](https://github.com/verl-project/verl-hardware-plugin/pull/184) (not merged yet): `git -C ../verl-hardware-plugin fetch origin pull/184/head:graph-trainer-kda && git -C ../verl-hardware-plugin checkout graph-trainer-kda`. Without it, graph mode rejects `kda`.

```bash
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env MODEL_NAME=Qwen3.5-35B-A3B TRAINER_MODE=graph_trainer GRAPH_TRAINER_SPLIT_LAYERS=2 MODEL_KERNELS=moe,kda,lce \
  USE_KL_LOSS=False PARAM_DTYPE=bfloat16 OPTIM_IMPL=fused \
  SPARSECORE_OFFLOAD=True TRAINER_SCOPED_VMEM_LIMIT_KIB=65472 TRAINER_EXTRA_LIBTPU_INIT_ARGS=--xla_tpu_dvfs_p_state=7 \
  TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=4 \
  USE_DYNAMIC_BSZ=True MAX_TOKEN_LEN_PER_GPU=32768 STATIC_MICRO_BATCH_LEN=32768 STATIC_MAX_RESPONSE_LEN=2048 \
  bash examples/tpu/grpo/run_qwen3_5_torchtitan.sh
```

Each rank trains one 32K-token micro-batch per step (16 sequences), so the whole mini-batch must fit: raise `MAX_TOKEN_LEN_PER_GPU` / `STATIC_MICRO_BATCH_LEN` together if the run reports more than one micro-batch. The KDA document budget is pinned on the first step from the most sequences per micro-batch; set `+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.kda_document_budget=<N>` if a later step packs more.

### Qwen3.5-397B-A17B (expert parallelism)

Expert parallelism (`EXPERT_PARALLEL_SIZE > 1`) needs torchtitan from [google-pytorch/torchtitan#142](https://github.com/google-pytorch/torchtitan/pull/142), which enables Qwen3.5 with EP on TPU. Build the image at its head commit:

```bash
docker buildx build -t <IMAGE> --push \
  --build-arg TORCHTITAN_COMMIT=4e16e67aed8c1e702e6f440b1ff81619bb6873d4 examples/tpu/grpo
```

Full-parameter 8K GRPO on 64 TPU v7x chips (a `2x4x4` trainer slice and four `2x2x2` rollout slices):

```bash
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env MODEL_NAME=Qwen3.5-397B-A17B \
  NNODES_TRAINER=8 EXPERT_PARALLEL_SIZE=8 PARAM_DTYPE=bfloat16 FP32_MASTER=True OPTIMIZER_HOST_OFFLOAD=True USE_KL_LOSS=False \
  EP_DISPATCHER=static EP_TOKEN_CHUNKS=8 EP_PAIRED_GROUPED_MM=True AC_SAVE_SPLASH_AND_ROUTING=True \
  SPARSECORE_OFFLOAD=True TRAINER_SCOPED_VMEM_LIMIT_KIB=65472 \
  USE_DYNAMIC_BSZ=True MAX_TOKEN_LEN_PER_GPU=16384 STATIC_MICRO_BATCH_LEN=16384 STATIC_MAX_RESPONSE_LEN=7680 \
  NUM_WARMUP_BATCHES=3 MAX_NUM_SEQS=512 MAX_NUM_BATCHED_TOKENS=8192 \
  NNODES_ROLLOUT=8 ROLLOUT_TENSOR_PARALLEL_SIZE=16 ROLLOUT_EXPERT_PARALLEL_SIZE=16 ROLLOUT_LOAD_FORMAT=dummy \
  TRAIN_BATCH_SIZE=64 ROLLOUT_N=16 MAX_PROMPT_LEN=512 MAX_RESPONSE_LEN=7680 \
  bash examples/tpu/grpo/run_qwen3_5_torchtitan.sh
```

## 8. Qwen3-30B-A3B (MoE)

[`run_qwen3_30b_a3b_torchtitan.sh`](run_qwen3_30b_a3b_torchtitan.sh) runs GRPO on GSM8K with Qwen3-30B-A3B (128 experts, 8 active per token). It needs `verl-hardware-plugin` with Qwen3 MoE training (verl-hardware-plugin#88 and #112), e.g. `git -C ../verl-hardware-plugin checkout 3b5213ce624c930ea1c992bfa8c633c2e0372490`.

| TPU | Trainer slice + rollout slice | Ray cluster ([`ray-tpu-v6e-2slice.yaml`](ray-tpu-v6e-2slice.yaml), `replicas: 2`) |
|---|---|---|
| v7x (default) | `2x2x1` + `2x2x1` | `numOfHosts: 1`, `google.com/tpu: "4"`, `gke-tpu-accelerator: tpu7x`, `gke-tpu-topology: 2x2x1` |
| v5p | `2x2x4` + `2x2x4` | `numOfHosts: 4`, `google.com/tpu: "4"`, `gke-tpu-accelerator: tpu-v5p-slice`, `gke-tpu-topology: 2x2x4` |

Upload the checkpoint as in step 3 (the recipe reads `/data/assets/hf/Qwen3-30B-A3B`, or set `MODEL_PATH`), create the Ray cluster for the TPU as in step 4, then submit as in step 5:

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir /tmp/Qwen3-30B-A3B
gcloud storage cp -r /tmp/Qwen3-30B-A3B gs://<BUCKET>/assets/hf/

# TPU v7x
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- bash examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh

# TPU v5p: also add "TPU_ACCELERATOR_TYPE": "v5p", "TORCH_TPU_TOPOLOGY": "2,2,4", "USE_MOE_SPARSE_CORE": "0" and
# "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": "USE_MOE_SPARSE_CORE" to env_vars in RUNTIME_ENV
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env NNODES_TRAINER=4 N_CHIPS_TRAINER=4 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=4 bash examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh
```

**bfloat16 FSDP gradient reduction.** `MIXED_PRECISION_REDUCE` defaults to `bfloat16` on the actor, reducing `update_actor` latency by ~9–18% across eager and graph modes with matching 30-step GSM8K reward and KL; set `MIXED_PRECISION_REDUCE=float32` to keep `float32` reduce-scatter. Optionally set `AC_SAVE_SPLASH_AND_ROUTING=True` in eager mode to save Splash attention and MoE routing outputs.

**Graph-mode actor update (optional).** With `verl-hardware-plugin` including PR #134 (`engine_kwargs.tpu.trainer_mode=graph_trainer` and dict `engine_kwargs.tpu.graph_trainer`, e.g. `git -C ../verl-hardware-plugin checkout 8a9e485da66ea76d40b01b4940db08928fc45061`), `TRAINER_MODE=graph_trainer` runs the actor update through TorchTitan's `TpuGraphTrainer`: forward, fused log-prob, PPO loss and backward of a micro-batch are traced once and compiled as one program. The first step pays a one-time trace and compile. Each rank must fit its mini-batch in one micro-batch, so raise `MAX_TOKEN_LEN_PER_GPU` if the run reports more than one; `STATIC_MICRO_BATCH_LEN` defaults to it.

- **Lowest per-step latency (`8192` tokens/chip):** default unsplit `tpu_selective` (`GRAPH_TRAINER_SAVE_REGIONS=moe.routing,attention.inner_attention`), with `GRAPH_TRAINER_FSDP_BARRIER_POLICY` auto-defaulting to `prefetch` on `v5p*` and `interleave` elsewhere (`prefetch` exceeds compile HBM on `tpu7x` 2x2x1):

```bash
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env TRAINER_MODE=graph_trainer bash examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh
```

- **Fast compile, lower HBM, and large-batch MFU scaling (`16384`–`32768` tokens/chip):** `GRAPH_TRAINER_SPLIT_LAYERS=1` defaults to `memory_policy=default`, `fsdp_barrier_policy=none`, `enable_remat_barriers=false`, and `backward_partition_mode=fsdp_pipeline` (cuts compile time to ~1.5 min, saves ~10 GB HBM, and enables `@16384` on `tpu7x` / `@32768` on `v5p` where unsplit OOMs). Raise the batch with the token budget so each chip's micro-batch is mostly real tokens:

```bash
# TPU v7x (8 trainer chips), highest MFU in the table below
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env TRAINER_MODE=graph_trainer GRAPH_TRAINER_SPLIT_LAYERS=1 MAX_TOKEN_LEN_PER_GPU=16384 \
  bash examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh

# TPU v5p 2x2x4 (16 trainer chips, RUNTIME_ENV as above), highest MFU in the table below
ray job submit --working-dir . --runtime-env-json "$RUNTIME_ENV" -- \
  env NNODES_TRAINER=4 N_CHIPS_TRAINER=4 NNODES_ROLLOUT=4 N_CHIPS_ROLLOUT=4 \
  TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=128 TRAINER_MODE=graph_trainer GRAPH_TRAINER_SPLIT_LAYERS=1 \
  MAX_TOKEN_LEN_PER_GPU=32768 bash examples/tpu/grpo/run_qwen3_30b_a3b_torchtitan.sh
```

Best measured configurations (Qwen3-30B-A3B GSM8K GRPO, `MAX_RESPONSE_LEN=1024`, thinking off, `verify_parity=all`; `update_actor` and `perf/mfu/actor` are medians over steps 2-8, first step excluded). On v7x, `32 x 8 @8192` needs more than one micro-batch per rank, so its `8192` rows use `TRAIN_BATCH_SIZE=PPO_MINI_BATCH_SIZE=16`:

| Hardware (trainer) | Batch @ tokens/chip | Settings | `update_actor` | MFU | Eager (`AC_SAVE_SPLASH_AND_ROUTING=True`), bf16 reduce |
|---|---|---|---|---|---|
| v5p 2x2x4 | 32 x 8 @8192 | `TRAINER_MODE=graph_trainer` (`prefetch`) | 3.29 s | 0.078 | 3.65 s / 0.072 |
| v5p 2x2x4 | 128 x 8 @32768 | `+ GRAPH_TRAINER_SPLIT_LAYERS=1` | 9.03 s | 0.110 | 9.72 s / 0.101 |
| v7x 2x2x1 | 16 x 8 @8192 | `TRAINER_MODE=graph_trainer` (`interleave`) | 2.02 s | 0.050 | 2.13 s / 0.047 |
| v7x 2x2x1 | 32 x 8 @16384 | `+ GRAPH_TRAINER_SPLIT_LAYERS=1` | 2.86 s | 0.067 | 3.16 s / 0.059 |

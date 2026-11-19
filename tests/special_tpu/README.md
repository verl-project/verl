# TPU CI

End-to-end tests of verl on Cloud TPU v6e. verl's TPU support comes from [`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin) (TPU platform, TorchTitan TPU engine, vLLM TPU rollout, `tpu-sync` weight sync), and the TPU recipes live in [`examples/tpu`](../../examples/tpu). The TPU CI runs those recipes the way their READMEs do, on the verl commit under test, so that a verl change that breaks TPU, or a TPU recipe that stops working, shows up on the pull request.

## Workflows

Like the NVIDIA, Ascend and AMD CI, there is one workflow per test category, named after the GPU workflow it mirrors with a `_tpu` suffix:

| Workflow | Mirrors | Test (`run_tpu_e2e_ci.sh <test>`) | TPU | Trigger |
|---|---|---|---|---|
| [`tpu_unit_tests.yml`](../../.github/workflows/tpu_unit_tests.yml) | `gpu_unit_tests.yml`, `npu_unit_tests.yml` | `tpu_platform`: [`tpu_smoke_test.py`](tpu_smoke_test.py) — `torch_tpu` ops and autograd, the same MLP on TPU and CPU, and that `import verl` loads the plugin, selects the `tpu` device and registers the TorchTitan TPU engine and the `tpu-sync` / `ray-plasma-store` weight sync backends; then the verl unit tests that run on TPU (see [Unit tests](#unit-tests)) | 1 x v6e-1 | push / PR |
| [`e2e_sft_llm_tpu.yml`](../../.github/workflows/e2e_sft_llm_tpu.yml) | `e2e_sft_llm.yml`, `e2e_sft_llm_ascend.yml` | `sft`: [`examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh`](../../examples/tpu/sft/run_qwen3_0_6b_torchtitan.sh) with `SMOKE_TEST=1` (8 steps, FSDP over 4 chips), once with torchrun (`verl.trainer.sft_trainer`, the default) and once with `SFT_TRAINER_MODE=ray` (`verl.trainer.sft_trainer_ray`); each run saves a checkpoint at its last step, from which a second run resumes for 2 more steps (`trainer.resume_mode=auto`), like the resume test of `e2e_sft_llm.yml` | 1 x v6e-4 | push / PR |
| [`e2e_v1_separate_async_tpu.yml`](../../.github/workflows/e2e_v1_separate_async_tpu.yml) | `e2e_v1_separate_async.yml`, `e2e_v1_separate_async_ascend.yml` | `grpo`: [`examples/tpu/grpo/run_qwen3_torchtitan.sh`](../../examples/tpu/grpo/run_qwen3_torchtitan.sh) with `MODEL_SIZE=0.6B` — the 5-step Qwen3-0.6B GRPO smoke test of the README: V1 separate-async trainer, TorchTitan trainer on one v6e-1 slice, vLLM rollout on the other, `tpu-sync` weight sync with `VERIFY_PARITY=all` | 2 x v6e-1 | push / PR |
| [`nightly_tpu.yml`](../../.github/workflows/nightly_tpu.yml) | `nightly_ascend.yml` | `grpo_4b`: the same script with `MODEL_SIZE=4B` for 20 steps (`GRPO_4B_STEPS`), validation before and after; and `grpo` against verl-hardware-plugin `main` | 2 x v6e-4, 2 x v6e-1 | schedule / manual / PRs that change it |
| [`docker-build-tpu.yml`](../../.github/workflows/docker-build-tpu.yml) | `docker-build-ascend-*.yml` | builds the image of [`examples/tpu/grpo/Dockerfile`](../../examples/tpu/grpo/Dockerfile) (see [Image](#image)) | — | manual |

The push and pull request triggers cover `main` and `v0.*` with path filters: the verl sources the TPU recipes use (not Megatron, MindSpeed, VeOmni, AutoModel, TensorRT-LLM or the GPU SGLang rollout), the trainer configs, the recipe and Dockerfile under test, `examples/tpu/grpo/README.md` (it pins the plugin), `tests/special_tpu`, and for `tpu_unit_tests.yml` the unit tests it runs. The data preprocessing scripts are not among them: the TPU CI reads data staged in the bucket (see [CI cluster](#ci-cluster)).

[`verify_tpu_e2e_log.py`](verify_tpu_e2e_log.py) checks every log: every unit test file ran and passed; all steps ran; losses, gradient norms and rewards are finite; SFT train and validation losses go down, the checkpoint of the last step was saved, and the resumed run continued at the next step; GRPO gradients flow, at least one rollout earns a reward, validation runs, the rollout (vLLM) and trainer log-probs of the sampled tokens agree (`rollout_corr/kl` < 0.02 and `rollout_corr/ppl_ratio` within 5% of 1 on every step, which catches a broken weight sync), and the `tpu-sync` parity checks ran and passed (the step-0 sync matches the checkpoint vLLM loaded bit for bit, the weight norms of every later sync are `VERIFIED`); the 4B run must raise GSM8K validation accuracy to at least 0.45.

## Unit tests

`gpu_unit_tests.yml` and `npu_unit_tests.yml` run all of `tests/` minus ignore globs. Most of those tests rely on GPU behaviour that TPU does not have, because a TPU chip belongs to one process at a time:

- Several WorkerGroups cannot share a resource pool. On TPU, `RayResourcePool` caps colocation at one WorkerGroup (`supports_colocated_worker_groups()`), so a second group on the same pool is never scheduled (e.g. `test_colocated_workers.py`, `test_high_level_scheduling_api.py`, `test_nested_worker.py`).
- The driver cannot load device tensors that workers return, because their chips are taken (`test_worker_group_torch.py`).
- Every rank of a WorkerGroup has to initialize the TPU runtime, so a test in which only rank 0 touches the device hangs (`test_worker_group_basics.py`).

So `tpu_unit_tests.yml` runs a list instead: `TPU_UNIT_TESTS` in [`run_tpu_e2e_ci.sh`](run_tpu_e2e_ci.sh), the verl unit tests that pass on one chip (single-controller data transfer, rollout correction on TPU tensors, FLOPs counter). [`run_tpu_unit_tests.sh`](run_tpu_unit_tests.sh) runs them in the TPU container (or on any TPU host), one pytest process per file, and each test that uses Ray gets its own local Ray instance. The TPU code itself lives in verl-hardware-plugin, whose unit tests run in that repository.

## How a TPU job runs

Like the GPU workflows, every TPU workflow starts with a `setup` job that selects the runners (here: reads the runner label, see [CI cluster](#ci-cluster)). The TPU jobs run on self-hosted runners: small pods on the CPU node pool of a GKE cluster that also has TPU v6e node pools. A job checks out the commit under test and calls [`run_tpu_e2e_ci.sh <test>`](run_tpu_e2e_ci.sh), which

1. finds the image of the checkout's `examples/tpu/grpo/Dockerfile` (see [Image](#image)) and fetches the plugin (see [Code under test](#code-under-test));
2. creates a RayCluster for this test from [`gke/raycluster-ci.yaml`](gke/raycluster-ci.yaml) (the pods of [`examples/tpu/grpo/ray-tpu-v6e-2slice.yaml`](../../examples/tpu/grpo/ray-tpu-v6e-2slice.yaml)), queued by Kueue: it stays suspended until its chip-count queue has TPU quota, then all of its pods start at once;
3. submits the test with `ray job submit --working-dir <checkout> --runtime-env-json <RUNTIME_ENV of the README>`, with the example's settings as environment variables of the entrypoint, streams and saves the log (the workflow uploads it as the artifact `tpu-logs-<test>-<attempt>`), and runs the verifier. The torchrun SFT run is the exception: it copies the checkout and the plugin into the TPU worker container and runs the script there with `kubectl exec` (a Ray driver that reserves all chips of a host gets Ray's host-default TPU environment, which does not describe a 2x2 subslice);
4. deletes the RayCluster on exit (pass, failure, timeout or cancellation). The workflow repeats the deletion in an `if: always()` step, and every run deletes CI RayClusters older than 3 hours.

When a test fails and one of the TPU nodes it ran on is gone or `NotReady` (a spot preemption), the test is re-run once on a new RayCluster (`PREEMPTION_RETRIES`).

| Shape | TPU workers | Kueue LocalQueue | Quota |
|---|---|---|---|
| 1 x v6e-1 | a whole `ct6e-standard-1t` (1x1) node | `verl-tpu-ci-1chip` | 1 chip, borrows from 2chip |
| 2 x v6e-1 | two such nodes, one Ray TPU slice each (`tpu-group-0` trainer, `tpu-group-1` rollout) | `verl-tpu-ci-2chip` | 2 chips, borrows from 1chip |
| 1 x v6e-4 | one host of the 2x4 (v6e-8) node pools, as a 2x2 subslice | `verl-tpu-ci-4chip` | 8 chips, borrows from 8chip |
| 2 x v6e-4 | two such hosts, one Ray TPU slice each | `verl-tpu-ci-8chip` | 8 chips, borrows from 4chip |

A host is the smallest multi-chip unit: GKE only admits whole-host TPU requests on 2x4 nodes, and libtpu cannot open an ICI session on 2 chips of a host, so the 2-chip test uses two 1-chip nodes.

## Code under test

- **verl**: the checked-out commit (for a pull request, its merge commit), as for every other verl workflow.
- **verl-hardware-plugin**: the repository and commit that step 1 of [`examples/tpu/grpo/README.md`](../../examples/tpu/grpo/README.md) clones and checks out, so CI tests exactly what the README tells users to run. A verl change that needs a newer plugin updates that line, and CI tests the change with it. To test another plugin commit (e.g. of a fork), set `VERL_HARDWARE_PLUGIN_REPO` / `VERL_HARDWARE_PLUGIN_COMMIT` (a full 40-character commit id, a branch or a tag), or use the `plugin_repo` / `plugin_commit` inputs of a manual run. The nightly run also tests the plugin's `main` branch.

## Image

The Ray pods run `TPU_CI_IMAGE_REPO:df-<first 16 hex digits of the sha256 of examples/tpu/grpo/Dockerfile>`, pinned by digest, so a Dockerfile change is tested together with the code that needs it. When that image does not exist yet, it is built in the cluster by a Kubernetes Job ([`gke/image-build-job.yaml`](gke/image-build-job.yaml): daemonless BuildKit, the builder of `docker buildx build`, with the Dockerfile as the whole build context; about 25 minutes) — but only for code that is already in this repository, because the build runs privileged and pushes to the registry:

- pushes, schedules, manual runs and pull requests from branches of this repository build the image themselves (runs that start together share one build);
- pull requests from forks never build. Their TPU jobs wait (`IMAGE_WAIT_MINS`) and print what to do: a maintainer reviews the Dockerfile change and runs [`docker-build-tpu.yml`](../../.github/workflows/docker-build-tpu.yml) with ref `refs/pull/<number>/head`, which builds that Dockerfile with the scripts of `main`.

Set `TPU_CI_IMAGE` to run a prebuilt image instead.

## Run a test by hand

From a machine with `gcloud` access to the cluster and the Ray CLI of the image (`pip install "ray[default]==<version in the Dockerfile's FROM line>"`), at the root of the checkout:

```bash
bash tests/special_tpu/run_tpu_e2e_ci.sh tpu_platform   # or sft / grpo / grpo_4b
# another plugin commit, e.g. of a fork
VERL_HARDWARE_PLUGIN_REPO=https://github.com/<user>/verl-hardware-plugin.git VERL_HARDWARE_PLUGIN_COMMIT=<commit> \
    bash tests/special_tpu/run_tpu_e2e_ci.sh grpo
# a local plugin checkout (e.g. uncommitted changes)
VERL_HARDWARE_PLUGIN_DIR=../verl-hardware-plugin bash tests/special_tpu/run_tpu_e2e_ci.sh grpo
```

The script fetches its own kubeconfig, so it does not change the current `kubectl` context. Logs go to `${TMPDIR:-/tmp}/verl-tpu-ci-<run id>/`.

## CI cluster

[`gke/tpu-ci.env`](gke/tpu-ci.env) holds the settings (cluster, namespace, queues, bucket, image repository); every value can be overridden by an environment variable. A cluster admin sets up:

| | |
|---|---|
| GKE cluster | Ray operator, Cloud Storage FUSE, Workload Identity, and Kueue with the `ray.io/raycluster` integration |
| TPU node pools | `ct6e-standard-1t` (v6e 1x1, autoscaling 0-3: one node for the 1-chip tier and two for the 2-chip tier) and two v6e-8 pools (`ct6e-standard-4t`, `--tpu-topology 2x4`, 2 nodes each) |
| Kueue queues | [`gke/kueue-tpu-ci.yaml`](gke/kueue-tpu-ci.yaml) (`kubectl apply -f`); CI never changes cluster-wide objects and fails if a queue is missing |
| Runners | [`gke/github-runner.yaml`](gke/github-runner.yaml): ephemeral runners (one job per container) whose service account may only manage RayClusters, exec into their pods and read nodes. The `setup` job of each TPU workflow (GitHub-hosted) reads their label from the repository variable `TPU_CI_RUNNER`, else from `TPU_CI_RUNNER` in [`gke/tpu-ci.env`](gke/tpu-ci.env) — GitHub does not pass variables to pull requests from forks — and the TPU jobs are skipped while it is empty. `TPU_CI_BUILD_RUNNER` (optional variable) selects other runners for `docker-build-tpu.yml`, e.g. the only ones allowed to push images. As for any self-hosted runner of a public repository, require approval for workflow runs of outside contributors |
| Models and data | GCS bucket `TPU_CI_BUCKET`, mounted at `/data` and read through the Ray pods' service account `TPU_CI_RAY_SERVICE_ACCOUNT`; models in `<TPU_CI_DATA_HOME>/assets/hf/Qwen3-{0.6B,4B}`, data in `<TPU_CI_DATA_HOME>/data/gsm8k` and `data/gsm8k_sft` (step 3 of the GRPO README). The GPU CI preprocesses its data in every job; the TPU data is staged once, so re-stage it when `examples/data_preprocess/gsm8k*.py` change their output |

> [!IMPORTANT]
> Whoever can create a RayCluster (or a Job) can run pods as any service account of its namespace, and pull requests run code on the runners. Keep the CI in a dedicated namespace (`TPU_CI_NAMESPACE`) whose service accounts are all safe for that: the runners' and the Ray pods' (read-only access to the bucket), nothing bound to cluster-wide roles.

> [!WARNING]
> Do not create long-lived RayClusters on the TPU node pools of the CI cluster: Kueue does not see them, so CI RayClusters get admitted but their TPU pods stay `Pending`.

## Maintenance

- **New test**: add a `test_<name>` function and its RayCluster shape in [`run_tpu_e2e_ci.sh`](run_tpu_e2e_ci.sh), a verifier in [`verify_tpu_e2e_log.py`](verify_tpu_e2e_log.py), and a workflow (or a step in an existing one).
- **New unit test**: once a verl unit test passes on TPU (run `tpu_platform` with it added), add it to `TPU_UNIT_TESTS` and to the path filters of `tpu_unit_tests.yml`.
- **New plugin commit**: update the checkout line of [`examples/tpu/grpo/README.md`](../../examples/tpu/grpo/README.md); its TPU jobs test the new commit. The nightly `main` run shows when that is possible.
- **New image**: change [`examples/tpu/grpo/Dockerfile`](../../examples/tpu/grpo/Dockerfile); see [Image](#image).

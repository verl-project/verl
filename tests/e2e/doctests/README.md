# Ascend documentation doctests

These tests execute commands from the public [source installation](../../../docs/ascend_tutorial/zh/get_start/container_source_install.rst),
[installation](../../../docs/ascend_tutorial/zh/get_start/install_guidance.rst),
[Docker](../../../docs/ascend_tutorial/zh/get_start/dockerfile_build_guidance.rst), and
[Quick Start](../../../docs/ascend_tutorial/zh/get_start/quick_start.rst) guides.

| CI job | What it verifies |
| --- | --- |
| CPU documentation checks | Extract the documented commands, check Bash syntax, and test the workers with pytest. |
| A3 vLLM Docker build (ARM64 CPU) | Build the PR checkout with its submodules and read package metadata inside the new image. The image stays on the build runner. |
| Published A3 vLLM image (8 NPUs) | Run one GRPO step from the installed source at `/verl` with FSDP2 and Megatron. |
| CANN source installation (8 NPUs) | Create the documented Conda environment, install the PR commit, check imports and run one GRPO step for each FSDP2/Megatron × vLLM/SGLang combination. |

CI reads the image tags from the guides. NPU cases run one at a time. A manual
workflow run can select `fsdp2_vllm`, `megatron_vllm`, `fsdp2_sglang`,
`megatron_sglang`, or `all`. A vLLM selection also runs the Docker build and
the corresponding published-image case.

## Run locally

Run these commands from the repository root. The read-only checks need Bash
and Python 3.10+. `DOCTEST_PYTHON` selects the extraction interpreter.
On Windows, `DOCTEST_TEST_BASH` selects Git Bash for the pytest suite.

```bash
runner=tests/e2e/doctests/scripts/run_doctests.sh
bash "$runner" installation check
DRY_RUN=1 bash "$runner" quickstart fsdp2_vllm
python3 -m pytest -q tests/special_sanity/test_doctest_helper.py
```

To build the documented A3 vLLM image on an ARM64 Docker host:

```bash
git submodule update --init --recursive
ALLOW_INSTALL=1 bash "$runner" installation docker
```

For Quick Start in an installed Ascend environment or source installation in
a disposable CANN container with Conda, point both workers to local inputs:

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B GSM8K_DATASET_PATH=/path/to/gsm8k
bash "$runner" quickstart fsdp2_vllm
ALLOW_INSTALL=1 INSTALL_BACKEND=sglang INSTALL_CASE=megatron_sglang \
  bash "$runner" installation source
```

- Source installation defaults to `fsdp2_vllm`. `INSTALL_CASE` must match
  `INSTALL_BACKEND`. The worker clones the current commit. Use `VERL_REPOSITORY`
  and `VERL_REVISION` to select another source. Set `KEEP_INSTALL_WORKDIR=1`
  to retain its temporary Conda environment, or supply an empty `INSTALL_WORKDIR`.
- Quick Start defaults to one training step and eight logical devices. Set
  `TOTAL_TRAINING_STEPS` and `NDEVICES_PER_NODE` to override them. Supply both
  `TRAIN_FILE` and `TEST_FILE` to reuse parquet files, or set `GSM8K_OUTPUT_DIR`
  to choose the preprocessing output directory. `CANN_ENV_SCRIPT` and
  `ATB_ENV_SCRIPT` select local environment scripts.
- Source and Docker modes support `ALLOW_INSTALL=1 DRY_RUN=1`. To inspect the
  planned images, run `python3 tests/e2e/doctests/scripts/doctest_helper.py matrix`
  with `--method source` or `--method prebuilt` and optionally `--case`.
- On the A3 runner, source installation uses the TUNA conda-forge mirror,
  the Huawei Cloud PyPI mirror, and a writable `HF_DATASETS_CACHE`. Documented
  Git clones and submodule updates get up to three attempts, with a 20-minute
  limit per attempt. Git aborts HTTP transfers below 10 KiB/s for 120 seconds.
  For local diagnosis set `DOCTEST_GIT_ATTEMPTS`, `DOCTEST_GIT_TIMEOUT`,
  `DOCTEST_GIT_RETRY_DELAY`, `GIT_HTTP_LOW_SPEED_LIMIT` or `GIT_HTTP_LOW_SPEED_TIME`.

# Tests layout

Each folder under tests/ corresponds to a test category for a sub-namespace in verl. For instance:
- `tests/trainer` for testing functionality related to `verl/trainer`
- `tests/models` for testing functionality related to `verl/models`
- ...

There are a few folders with `special_` prefix, created for special purposes:
- `special_distributed`: unit tests that must run with multiple GPUs
- `special_e2e`: end-to-end tests with training/generation scripts
- `special_npu`: tests for NPUs
- `special_sanity`: a suite of quick sanity tests
- `special_standalone`: a set of test that are designed to run in dedicated environments

Accelerators for tests 
- By default tests are run with GPU available, except for the ones under `special_npu`, and any test script whose name ends with `on_cpu.py`.
- For test scripts with `on_cpu.py` name suffix would be tested on CPU resources in linux environment.

# Workflow layout

All CI tests are configured by yaml files in `.github/workflows/`. Here's an overview of all test configs:
1. A list of always triggered CPU sanity tests: `check-pr-title.yml`, `secrets_scan.yml`, `check-pr-title,yml`, `pre-commit.yml`, `doc.yml`
2. Some heavy multi-GPU unit tests, such as `model.yml`, `vllm.yml`, `sgl.yml`
3. End-to-end tests: `e2e_*.yml`
4. Unit tests
  - `cpu_unit_tests.yml`, run pytest on all scripts with file name pattern `tests/**/test_*_on_cpu.py`
  - `gpu_unit_tests.yml`, run pytest on all scripts with file without the `on_cpu.py` suffix.
  - Since cpu/gpu unit tests by default runs all tests under `tests`, please make sure tests are manually excluded in them when
    - new workflow yaml is added to `.github/workflows`
    - new tests are added to workflow mentioned in 2.
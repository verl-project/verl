# verl Kernel Benchmark

## `linear_cross_entropy`

### 1. Unified Benchmark Settings

To ensure apples-to-apples comparisons across **CUDA (`SM90` / `SM100`)** and **Ascend (`910` / `950`)** for **Eager**, **Triton**, and **TileLang**, all runs follow the same benchmark protocol:

| Item | Unified Setting | Notes |
| :--- | :--- | :--- |
| **Shape (`M, V, H`)** | `M = 65536 (64k)`, `V = 129280`, `H = 5120` | `V_local = V / TP`: `129280` (`TP=1`), `32320` (`TP=4`) |
| **Query `topk`** | `1` (single-label), `129` (`1` target + `128` top-k) | `labels` dtype `int64`, `reduction = "none"` |
| **Parallelism (`TP`)** | `TP = 1`, `TP = 4` | Vocab-parallel sharding (`weight`: `[V / TP, H]`). For `TP = 4`, Backward latency includes `all_reduce(SUM)` on `d_hidden` and reports `max` rank latency |
| **Dtype & Hyperparams** | `hidden / weight`: `bfloat16`; outputs: `float32`; `temperature = 1.0` | Forward outputs `(logprobs, entropy)` in `fp32`; Backward returns `(d_hidden, d_weight)` in `bf16` |
| **Timing Protocol** | `10` warmup + `20` timed iterations | Device-event timing after JIT warmup; peak memory recorded via `max_memory_allocated()` |
| **Error Baseline** | Chunked Eager (`FP32` logits) | Avoids Eager OOM at `M = 64k` while measuring `MaxAbs = max \|x - x_ref\|` and `RelL2 = \|\|x - x_ref\|\|_2 / \|\|x_ref\|\|_2` |

---

### 2. Performance & Memory Benchmark (`Fwd` / `Bwd` / `Peak Mem`)

- **`Bwd CE`**: upstream `grad_logprobs` only (`entropy_coeff = 0`).
- **`Bwd CE+Ent`**: upstream `grad_logprobs` + `grad_entropy` (`entropy_coeff > 0`).
- **`Total`**: `Fwd + Bwd CE+Ent`; **`Speedup`**: `Eager Total / Kernel Total` (`OOM` if Eager exceeds device HBM).

#### 2.1 CUDA (`SM90` H100/H800, `SM100` B200)

| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend | Fwd (ms) | Bwd CE (ms) | Bwd CE+Ent (ms) | Total (ms) | Peak Mem (GiB) | Speedup vs Eager |
| :--- | :-: | :--- | :-: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | Eager | - | - | - | - | - | 1.00x |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | Eager | - | - | - | - | - | 1.00x |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | Eager (VocabParallel) | - | - | - | - | - | 1.00x |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | Eager (VocabParallel) | - | - | - | - | - | 1.00x |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | Eager | - | - | - | - | - | 1.00x |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | Eager | - | - | - | - | - | 1.00x |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | Eager (VocabParallel) | - | - | - | - | - | 1.00x |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | Eager (VocabParallel) | - | - | - | - | - | 1.00x |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - | - | - | - | - | - |

#### 2.2 Ascend (`910` 910B2C 64GiB, `950`)

| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend | Fwd (ms) | Bwd CE (ms) | Bwd CE+Ent (ms) | Total (ms) | Peak Mem (GiB) | Speedup vs Eager |
| :--- | :-: | :--- | :-: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | Eager | OOM | OOM | OOM | OOM | OOM (>61) | - |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | Triton | 963.90 | 1729.50 | 1735.26 | 2699.16 | 10.31 | Eager OOM |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | Eager | OOM | OOM | OOM | OOM | OOM (>61) | - |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | Triton | 1193.67 | 1795.62 | 1795.49 | 2989.16 | 10.54 | Eager OOM |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | Eager (VocabParallel) | 196.88 | 291.93 | 363.55 | 560.43 | 28.55 | 1.00x |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | Triton | 259.71 | 485.25 | 484.71 | 744.42 | 8.96 | 0.75x |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | Eager (VocabParallel) | OOM | OOM | OOM | OOM | OOM (>61) | - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | Triton | 328.86 | 499.56 | 499.92 | 828.77 | 9.18 | Eager OOM |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | Eager | - | - | - | - | - | 1.00x |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | Eager | - | - | - | - | - | 1.00x |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | Eager (VocabParallel) | - | - | - | - | - | 1.00x |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | Eager (VocabParallel) | - | - | - | - | - | 1.00x |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - | - | - | - | - | - |

---

### 3. Numerical Error vs Eager Reference (`MaxAbs / RelL2`)

Error is measured on the full `TP ∈ {1, 4}` × `topk ∈ {1, 129}` matrix against the **Chunked Eager (FP32 logits)** baseline (`MaxAbs = max |x - x_ref|`, `RelL2 = ||x - x_ref||_2 / ||x_ref||_2`), with `Bwd CE+Ent` enabled so both logprob and entropy gradient paths are verified:

#### 3.1 CUDA (`SM90` / `SM100`)

| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend | `logprobs` MaxAbs / RelL2 | `entropy` MaxAbs / RelL2 | `d_hidden` MaxAbs / RelL2 | `d_weight` MaxAbs / RelL2 |
| :--- | :-: | :--- | :-: | :--- | :---: | :---: | :---: | :---: |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | Eager BF16 | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | Eager BF16 | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | Eager BF16 | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | Eager BF16 | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |

#### 3.2 Ascend (`910` / `950`)

| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend | `logprobs` MaxAbs / RelL2 | `entropy` MaxAbs / RelL2 | `d_hidden` MaxAbs / RelL2 | `d_weight` MaxAbs / RelL2 |
| :--- | :-: | :--- | :-: | :--- | :---: | :---: | :---: | :---: |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | Eager BF16 | `7.76e-03 / 9.78e-05` | `4.48e-05 / 8.87e-07` | `8.59e-04 / 2.34e-03` | `8.25e-04 / 2.34e-03` |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | Triton | `5.72e-06 / 6.31e-08` | `2.86e-06 / 5.17e-08` | `1.26e-03 / 2.86e-03` | `8.25e-04 / 2.34e-03` |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | Eager BF16 | `7.82e-03 / 9.83e-05` | `4.48e-05 / 8.87e-07` | `2.56e-03 / 2.34e-03` | `1.93e-03 / 2.34e-03` |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | Triton | `7.63e-06 / 6.35e-08` | `2.86e-06 / 5.17e-08` | `3.37e-03 / 2.87e-03` | `1.93e-03 / 2.34e-03` |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | Triton | `3.81e-06 / 6.35e-08` | `2.86e-06 / 5.11e-08` | `1.54e-03 / 3.71e-03` | `8.64e-04 / 2.34e-03` |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | Triton | `7.63e-06 / 6.34e-08` | `2.86e-06 / 5.12e-08` | `5.87e-03 / 3.94e-03` | `2.26e-03 / 2.34e-03` |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | Eager BF16 | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | Eager BF16 | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |

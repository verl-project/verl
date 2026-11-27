# verl Kernel Benchmark

## `linear_cross_entropy`

### 1. Unified Benchmark Settings

To ensure apples-to-apples comparisons across **CUDA (`SM90` / `SM100`)** and **Ascend (`910` / `950`)** for **Eager**, **Triton**, and **TileLang**, all runs use [`tests/utils/kernel/linear_cross_entropy/benchmark_linear_cross_entropy.py`](../../../tests/utils/kernel/linear_cross_entropy/benchmark_linear_cross_entropy.py):

| Item | Unified Setting | Notes |
| :--- | :--- | :--- |
| **Shape (`M, V, H`)** | `M = 65536 (64k)`, `V = 129280`, `H = 5120` | `V_local = V / TP`: `129280` (`TP=1`), `32320` (`TP=4`) |
| **Query `topk`** | `1` (single-label), `129` (`1` target + `128` top-k) | `labels` dtype `int64`, `reduction = "none"` |
| **`TP = 1` Eager** | `Eager (FSDP)` | Production FSDP path in `verl/workers/engine/fsdp/transformer_impl.py` (`F.linear` + `entropy_from_logits` + `topk_log_probs_from_logits` + `logprobs_from_logits`) |
| **`TP = 4` Eager** | `Eager (Megatron)` | Production Megatron path in `verl/workers/engine/megatron/transformer_impl.py` (`ColumnParallelLinear` + `vocab_parallel_entropy` + `topk_log_probs_from_logits` + `vocab_parallel_log_probs_from_logits`) |
| **Dtype & Hyperparams** | `hidden / weight`: `bfloat16`; fused outputs: `float32`; `temperature = 1.0` | Backward latency for `TP = 4` includes `all_reduce(SUM)` on `d_hidden` and reports `max` rank latency |
| **Timing Protocol** | `10` warmup + `20` timed iterations | Device-event timing (`retain_graph=False` per backward step); peak memory recorded via `max_memory_allocated()` |
| **Error Baseline** | Chunked Eager (`FP32` logits) | Exact chunked `FP32` reference (`MaxAbs = max \|x - x_ref\|`, `RelL2 = \|\|x - x_ref\|\|_2 / \|\|x_ref\|\|_2`) |

```bash
# Run full TP={1,4}, topk={1,129} suite automatically:
python tests/utils/kernel/linear_cross_entropy/benchmark_linear_cross_entropy.py

# Or run TP=1 / TP=4 separately:
python tests/utils/kernel/linear_cross_entropy/benchmark_linear_cross_entropy.py --tp 1
torchrun --standalone --nproc_per_node=4 tests/utils/kernel/linear_cross_entropy/benchmark_linear_cross_entropy.py --tp 4
```

---

### 2. Performance & Memory Benchmark (`Fwd` / `Bwd` / `Peak Mem`)

- **`Bwd CE`**: upstream `grad_logprobs` only (`calculate_entropy = False`).
- **`Bwd CE+Ent`**: upstream `grad_logprobs` + `grad_entropy` (`calculate_entropy = True`).
- **`Total`**: `Fwd + Bwd CE+Ent`; **`Speedup`**: `Eager Total / Kernel Total` (`Eager OOM` if Eager exceeds device HBM).

#### 2.1 CUDA (`SM90` H100/H800, `SM100` B200)

| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend | Fwd (ms) | Bwd CE (ms) | Bwd CE+Ent (ms) | Total (ms) | Peak Mem (GiB) | Speedup vs Eager |
| :--- | :-: | :--- | :-: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | Eager (FSDP) | - | - | - | - | - | 1.00x |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | Eager (FSDP) | - | - | - | - | - | 1.00x |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | Eager (Megatron) | - | - | - | - | - | 1.00x |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | Eager (Megatron) | - | - | - | - | - | 1.00x |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | Eager (FSDP) | - | - | - | - | - | 1.00x |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | Eager (FSDP) | - | - | - | - | - | 1.00x |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | Eager (Megatron) | - | - | - | - | - | 1.00x |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | Eager (Megatron) | - | - | - | - | - | 1.00x |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - | - | - | - | - | - |

#### 2.2 Ascend (`910` 910B2C 64GiB, `950`)

| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend | Fwd (ms) | Bwd CE (ms) | Bwd CE+Ent (ms) | Total (ms) | Peak Mem (GiB) | Speedup vs Eager |
| :--- | :-: | :--- | :-: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | Eager (FSDP) | OOM | OOM | OOM | OOM | OOM | - |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | Triton | 952.29 | 1729.64 | 1736.60 | 2688.90 | 10.31 | Eager OOM |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | Eager (FSDP) | OOM | OOM | OOM | OOM | OOM | - |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | Triton | 1196.69 | 1796.92 | 1788.81 | 2985.50 | 10.54 | Eager OOM |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | Eager (Megatron) | 195.21 | 291.06 | 369.00 | 564.21 | 28.86 | 1.00x |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | Triton | 259.47 | 486.03 | 483.46 | 742.93 | 9.26 | 0.76x |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | Eager (Megatron) | 266.80 | 389.29 | 474.22 | 741.02 | 28.99 | 1.00x |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | Triton | 328.53 | 500.79 | 498.90 | 827.42 | 9.49 | 0.90x |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | Eager (FSDP) | - | - | - | - | - | 1.00x |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | Eager (FSDP) | - | - | - | - | - | 1.00x |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - | - | - | - | - | - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | Eager (Megatron) | - | - | - | - | - | 1.00x |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - | - | - | - | - | - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - | - | - | - | - | - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | Eager (Megatron) | - | - | - | - | - | 1.00x |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - | - | - | - | - | - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - | - | - | - | - | - |

---

### 3. Numerical Error vs Chunked FP32 Reference (`MaxAbs / RelL2`)

Error is measured on the full `TP ∈ {1, 4}` × `topk ∈ {1, 129}` matrix against the **Chunked Eager (`FP32` logits)** baseline (`MaxAbs = max |x - x_ref|`, `RelL2 = ||x - x_ref||_2 / ||x_ref||_2`), with `Bwd CE+Ent` enabled so both logprob and entropy gradient paths are verified:

#### 3.1 CUDA (`SM90` / `SM100`)

| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend | `logprobs` MaxAbs / RelL2 | `entropy` MaxAbs / RelL2 | `d_hidden` MaxAbs / RelL2 | `d_weight` MaxAbs / RelL2 |
| :--- | :-: | :--- | :-: | :--- | :---: | :---: | :---: | :---: |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | Eager (FSDP) | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | Eager (FSDP) | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **SM90** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | Eager (Megatron) | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | Eager (Megatron) | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **SM90** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | Eager (FSDP) | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | Eager (FSDP) | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **SM100** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | Eager (Megatron) | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | Eager (Megatron) | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **SM100** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |

#### 3.2 Ascend (`910` / `950`)

| Arch | TP | Shape (`M, V_local, H`) | `topk` | Backend | `logprobs` MaxAbs / RelL2 | `entropy` MaxAbs / RelL2 | `d_hidden` MaxAbs / RelL2 | `d_weight` MaxAbs / RelL2 |
| :--- | :-: | :--- | :-: | :--- | :---: | :---: | :---: | :---: |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | Eager (FSDP) | OOM | OOM | OOM | OOM |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | Triton | `4.77e-06 / 6.35e-08` | `2.86e-06 / 5.12e-08` | `1.15e-03 / 2.86e-03` | `8.08e-04 / 2.34e-03` |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | Eager (FSDP) | OOM | OOM | OOM | OOM |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | Triton | `6.68e-06 / 6.34e-08` | `2.86e-06 / 5.12e-08` | `3.20e-03 / 2.87e-03` | `2.14e-03 / 2.34e-03` |
| **Ascend 910** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | Eager (Megatron) | `7.82e-03 / 9.77e-05` | `6.95e-02 / 2.50e-03` | `1.34e-03 / 3.34e-03` | `8.09e-04 / 2.34e-03` |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | Triton | `3.81e-06 / 6.36e-08` | `2.86e-06 / 5.10e-08` | `1.34e-03 / 3.71e-03` | `8.09e-04 / 2.34e-03` |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | Eager (Megatron) | `7.82e-03 / 9.84e-05` | `7.00e-02 / 2.50e-03` | `4.88e-03 / 3.56e-03` | `2.28e-03 / 2.34e-03` |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | Triton | `6.68e-06 / 6.34e-08` | `2.86e-06 / 5.11e-08` | `5.54e-03 / 3.94e-03` | `2.28e-03 / 2.34e-03` |
| **Ascend 910** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | Eager (FSDP) | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | Eager (FSDP) | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **Ascend 950** | 1 | `(65536, 129280, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | Eager (Megatron) | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | Triton | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 1 | TileLang | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | Eager (Megatron) | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | Triton | - / - | - / - | - / - | - / - |
| **Ascend 950** | 4 | `(65536, 32320, 5120)` | 129 | TileLang | - / - | - / - | - / - | - / - |

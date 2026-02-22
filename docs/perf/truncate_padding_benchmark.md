# Truncate padding benchmark (Qwen 2.5 7B + GSM8K)

Short benchmark comparing **full padding**, **truncate_padding**, and **remove_padding** on the same setup to quantify speedup and memory reduction.

## Setup

- **Model**: Qwen 2.5 7B (bf16)
- **Data**: GSM8K (parquet), 32 train batch size, max prompt 4096, max response 8192
- **Hardware**: 1 node, 8× H100 80GB
- **Training**: 10 steps, GRPO, same hyperparameters across runs
- **Dynamic batching**: Full padding used fixed micro_batch_size=2; truncate and remove used `use_dynamic_bsz=True` with 16k tokens/GPU

## Results

| Config            | Total time (10 steps) | Time/step | Throughput (tok/s) | Actor MFU | Peak memory (GB) | update_actor/step |
|-------------------|------------------------|-----------|---------------------|-----------|------------------|--------------------|
| Full padding      | 13:45                  | 70–83 s   | ~128                | ~1.0%     | 36.4 / 45.9      | ~44 s              |
| Truncate padding  | 5:12                   | 19–31 s   | ~454                | ~11.6%    | 31.5 / 37.6      | ~5 s               |
| Remove padding    | 4:48                   | 13–29 s   | ~699                | ~20.4%    | 27.1 / 36.5      | ~3.5 s             |

## Summary

- **Truncate padding** is ~**2.6× faster** than full padding and uses ~**15% less** peak memory, with most of the benefit coming from shorter effective sequences in the actor forward/backward (update_actor time drops from ~44 s to ~5 s per step).
- **Remove padding** is faster and lighter still (~2.9× vs full, ~1.2× vs truncate) but requires sequence packing support; truncate_padding is intended for models that cannot use packing (e.g. Mamba/SSM).

*Single run per config, 10 steps; numbers are indicative.*

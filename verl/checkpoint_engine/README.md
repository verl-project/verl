Checkpoint Engine
---

### Overview

Checkpoint Engine is an unified abstract layer to synchronize weights between various training backends and inference backends. It provides three unified APIs:
- send_weights: get named tensors from generator and send them in streaming manner.
- receive_weights: return a tensor generator that yield named tensors in streaming manner.
- get_weights: return a tensor generator that yield named tensors in streaming manner, used for each inference instance update weight independently from local cache (e.g share memory, disk).

![checkpoint-engine](https://github.com/wuxibin89/verl/blob/wuxibin/doc_images/docs/_static/checkpoint_engine.png?raw=true)

### Supported Backends

||Comm Library|Topology|Hardware|Performance|Elastic|Use case|
|----|----|----|----|----|----|----|
|naive|torch.distributed|all_gather|NVIDIA/AMD/Ascend|Very High|NA|On-policy training<br>- Trainer/rollout colocated
|nccl|NCCL|all_gather+broadcast|NVIDIA GPU & NCCL|Very High|Low: rebuild nccl group|Off-policy training<br>- Trainer/rollout disaggregated<br>- Fixed clusters
|hccl|HCCL|all_gather+broadcast|Ascend NPU & HCCL| High|Low: rebuild hccl group|Off-policy training<br>- Trainer/rollout disaggregated<br>- Fixed clusters
|nixl|NIXL|all_gather+ring p2p|Various transport backends (D2D, H2H, H2D, etc)<br>- UCX<br>- UCCL<br>- Mooncacke|Medium/High|High: dynamic adjust ring topology|Off-policy training<br>- Trainer/rollout disaggregated<br>- Elastic rollout<br>- Rollout fault tolerance<br>- Heterogeneous hardware rollout

### Benchmark
1. benchmark setup
- model: Qwen/Qwen3-30B-A3B-Base
- trainer: fsdp world_size=2
- rollout: num_rollout=30 (only receive weight without cuda ipc to vllm/sglang)
```bash
python3 tests/checkpoint_engine/test_nixl_checkpoint_engine.py
python3 tests/checkpoint_engine/test_nccl_checkpoint_engine.py
python3 tests/checkpoint_engine/test_hccl_checkpoint_engine.py
```

2. benchmark result

| hardware | backend | time cost (s) | Bandwidth(GB/s) |
|----|----|----|----|
|4*8 H100, ConnectX-7 400 Gbps (InfiniBand)| NCCL | ~7 | 8.25|
|4*8 H100, ConnectX-7 400 Gbps (InfiniBand)| NIXL | ~7 | 8.25|
|2*16 Ascend 910C, inner suppernode| HCCL | ~11 | 5.3|
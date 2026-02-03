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
|kimi_ckpt_engine|MOONCAKE+NCCL/HCCL|p2p+broadcast|NVIDIA/Ascend|High|Low: rebuild communication group|Off-policy training<br>- Trainer/rollout disaggregated<br>- Save checkpoint each time

##### kimi_ckpt_engine detail:

In the kimi_ckpt_engine workflow, the trainer first offloads the weights to the CPU, and the rollout creates a sub communication group that includes all the cards for the rollout. Then, using Mooncake transfer engine, these weights are transmitted via P2P to a specific worker in the rollout, followed by a broadcast to all other rollout workers.

<img src="https://github.com/kip-cxj/verl/blob/cxj/doc_imgs/docs/_static/kimi_ckpt_engine.png?raw=true" alt="kimi-ckpt-engine" width="50%">

This mode requires the P2P feature of checkpoint_engine. Please ensure you have installed it via pip install 'checkpoint-engine[p2p]' and that your version is 0.4.0 or higher.

In addition, during the installation of checkpoint-engine[p2p], the transfer engine will be installed. However, This library has no prebuilt packages for Ascend devices and must be compiled from source. For detailed compilation instructions, see: [transfer-engine: ascend direct](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/ascend_direct_transport.md)

### Benchmark
1. benchmark setup
- model: Qwen/Qwen3-30B-A3B-Base
- trainer: fsdp world_size=2 (since Ascend 910C has 64GB of HBM, we set world_size=4)
- rollout: num_rollout=30 (only receive weight without cuda ipc to vllm/sglang)
```bash
pytest tests/checkpoint_engine/test_correctness_on_gpu.py
pytest tests/checkpoint_engine/test_correctness_on_npu.py
pytest tests/checkpoint_engine/test_special_server_adapter.py
```

2. benchmark result

| hardware | backend | time cost (s) | Bandwidth(GB/s) |
|----|----|----|----|
|4*8 H100, ConnectX-7 400 Gbps (InfiniBand)| NCCL | ~7 | 8.25|
|4*8 H100, ConnectX-7 400 Gbps (InfiniBand)| NIXL | ~7 | 8.25|
|2*16 Ascend 910C, inner suppernode| HCCL | ~11 | 5.3|
|2*16 Ascend 910C, inner suppernode| kimi_ckpt_engine | offload: 7 update: 3.5 | 16.5|

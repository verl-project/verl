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
|naive|torch.distributed|all_gather|NVIDIA/AMD/Ascend|Very High|NA|On-policy training<br>- Actor/rollout colocated
|nccl|NCCL|all_gather+broadcast|NVIDIA GPU & NCCL|Very High|Low: rebuild nccl group|Off-policy training<br>- Actor/rollout disaggregated<br>- Fixed clusters
|hccl|HCCL|all_gather+broadcast|Ascend NPU & HCCL| High|Low: rebuild hccl group|Off-policy training<br>- Actor/rollout disaggregated<br>- Fixed clusters
|nixl|NIXL|all_gather+ring p2p|Various transport backends (D2D, H2H, H2D, etc)<br>- UCX<br>- UCCL<br>- Mooncacke|Medium/High|High: dynamic adjust ring topology|Off-policy training<br>- Actor/rollout disaggregated<br>- Elastic rollout<br>- Rollout fault tolerance<br>- Heterogeneous hardware rollout
|kimi_ckpt_engine|MOONCAKE+NCCL/HCCL|p2p+broadcast|NVIDIA/Ascend|High|Low: rebuild communication group|Off-policy training<br>- Actor/rollout disaggregated<br>- Save checkpoint each time
|mooncake|Mooncake Transfer Engine|all_gather+ring p2p|NVIDIA/Ascend|High|High: dynamic adjust ring topology|Off-policy training<br>- Actor/rollout disaggregated<br>- Fixed clusters
|mori|MoRI-IO|all_gather (or per-rank shards)+p2p striped over all NICs, XGMI fan-out inside each rollout node|AMD GPU & RDMA NIC (Pollara AINIC tested)|High|High: topology rebuilt each sync|Off-policy training<br>- Actor/rollout disaggregated<br>- AMD clusters

##### mori detail:

`mori` pushes buckets with one-sided RDMA WRITE through [MoRI-IO](https://github.com/ROCm/mori) (`pip install amd-mori`). The model is split into stripes, one per trainer rank (up to `num_chains`, default 8), each sent by its own trainer GPU through its own NIC. On every rollout node one GPU holds each stripe (a different GPU per stripe, so all NICs of the node are used): it receives the stripe from outside and copies it to the other GPUs of the node over XGMI; the holders of a stripe on different rollout nodes form a chain. Each rollout node therefore takes one copy of the model, spread over its NICs.

With `export: shard`, stripe r carries trainer rank r's own shards from the training engine's `get_per_tensor_param_shard()` (bf16): the trainer does not all-gather, and rollout workers put full tensors together from the pieces of all stripes (and split packed MoE experts into per-expert names, as the full export does). This needs `num_chains` >= the trainer world size, a training engine whose shard export is in HF layout (FSDP; it declares `per_tensor_param_shard_is_hf`) and shards that are one flat range of the full tensor (FSDP2 `Shard(0)`); LoRA, QAT and parameters that need a conversion to HF layout are refused (use the default `export: full`). Rollout workers hold at most `num_slots * update_weights_bucket_megabytes` plus the largest tensor of partly assembled tensors. The trainer worker hands its training engine to `send_weights` (`consumes_training_engine`).

Each worker registers one region of `num_slots * update_weights_bucket_megabytes` (default 2 slots) once and keeps it, the QPs and the sockets across syncs; each stripe gets `update_weights_bucket_megabytes / num_chains` per slot, so larger buckets mean fewer, larger transfers. Options go in `engine_kwargs.mori` (`num_chains`, `num_slots`, `export`, `intra_node`, `qp_per_transfer`, `num_nics_per_transfer`, `timeout_s`). Buckets are allocated with cupy because MoRI cannot register PyTorch expandable-segment (VMM) memory.

**Failures are fail-stop.** MoRI 1.2.4 does not guarantee that a failed or abandoned transfer has stopped touching memory (a FAILED status can leave work requests posted), and a GPU consumer may still read a slot when its sync is aborted. So a worker whose sync fails, raises, is cancelled or is closed early marks its engine unusable before anything else and tells its peers, which then fail at once. It keeps its MoRI engine, region and transfer statuses alive for the rest of the process and refuses every further call; recreate the rollout and trainer workers to continue.
- A worker refuses `prepare`, `init_process_group` and a new sync while its previous sync is still running. `ray.get` in the manager returns on the first error, so a retry could otherwise reach a worker that is still unwinding.
- A receiver's sync starts when `receive_weights` is called, not at its first iteration; closing the stream before the first tensor fails the sync too.
- A receiver grants its slots to its previous hops only once it has entered the sync, so nothing is ever written into a worker that is not guarding its region.

**Rollout workers that join, leave or are replaced.** Each topology gets an id that tags its messages. The engine descriptors and sockets of peers that left are released at once. MoRI 1.2.4's `deregister_remote_engine` keeps their RDMA endpoints, sessions and XGMI IPC mappings until the MoRI engine is destroyed, so once more than max(16, 2 x current peers) peers have left, the next `prepare` replaces the worker's MoRI engine (same region, re-registered).

##### kimi_ckpt_engine detail:

In the kimi_ckpt_engine workflow, the actor first offloads the weights to the CPU, and the rollout creates a sub communication group that includes all the cards for the rollout. Then, using Mooncake transfer engine, these weights are transmitted via P2P to a specific worker in the rollout, followed by a broadcast to all other rollout workers.

<img src="https://github.com/kip-cxj/verl/blob/cxj/doc_imgs/docs/_static/kimi_ckpt_engine.png?raw=true" alt="kimi-ckpt-engine" width="50%">

This mode requires the P2P feature of checkpoint_engine. Please ensure you have installed it via pip install 'checkpoint-engine[p2p]' and that your version is 0.4.0 or higher.

In addition, during the installation of checkpoint-engine[p2p], the transfer engine will be installed. However, This library has no prebuilt packages for Ascend devices and must be compiled from source. For detailed compilation instructions, see: [transfer-engine: ascend direct](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/ascend_direct_transport.md)

Note: Important Configuration for Ascend Devices
1. In Ascend 950, you need enable whitelist to support kimi-checkpoint-engine. you should set the following environment variable and create
a json file to add your device ip addresses.

```bash
export HCCL_WHITELIST_DISABLE=0
export HCCL_WHITELIST_FILE=/path/to/whitelist.json
```

In whitelist json file, you need to add IP addresses using the following format.

```json
{ "host_ip": ["ip1"], ["ip2"] }
```

2. If you are using CANN version >= 8.5.0 on Ascend devices, you must set the following environment variable to enable intra-node ROCE:

```bash
export HCCL_INTRA_ROCE_ENABLE=1
```

### Benchmark
1. benchmark setup
- model: Qwen/Qwen3-30B-A3B-Base
- actor: fsdp world_size=2 (since Ascend 910C has 64GB of HBM, we set world_size=4)
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
|2*8 H100, ConnectX-7 400 Gbps (InfiniBand)| mooncake | 5.93 | 9.44|

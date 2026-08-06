# Training DeepSeek 671b

Last updated: 08/20/2025.

verl integrates Megatron to support large MoE models such as `Qwen3-235B-A22B` and `deepseek-ai/DeepSeek-V3`. This is an ongoing community effort.

In the journey the community added the following features and optimizations that enable verl with larger models:
- per tensor weight resharding between rollout and training
- context parallelism and expert parallelism enabled via megatron
- dynamic batch size (sequence balance) for megatron
- reduced ray-related serialization overhead
- optimizer offloading, recomputation, and efficient kernels
- various debugging metrics and utils
- hybrid optimizer

and the megatron backend now has a wider list of models supported:
- DeepSeek-V3
- Moonlight
- Qwen3
- Qwen2.5-VL (to be merged soon)
- Qwen2
- Mixtral

## Getting Started

### preparation
The recommended image with pre-built Megatron dependency is `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview`, which is built using the Dockerfile at [docker/verl0.4-cu124-torch2.6-fa2.7.4/Dockerfile.app.vllm.mcore0.13.preview](https://github.com/verl-project/verl/blob/main/docker/verl0.4-cu124-torch2.6-fa2.7.4/Dockerfile.app.vllm.mcore0.13.preview).

The image is build in Hopper GPUs with DeepEP. It does not support None-Hopper GPUs, such as A100. You may need to reinstall DeepEP to work with A100.

With `OFFLOAD_FRACTION=1`, the system's minimum requirements are lowered. It can run on as few as 96 H20 (96GB) GPUs for DeepSeek-V3, and on as few as 32 H20 (96GB) GPUs for Qwen3-235B-A22B. However, this configuration will use 1.6TB CPU memory per node. If you run out of CPU memory or require faster training speed, you can add more nodes.

### DeepSeek 671b

For DeepSeek-V3 671b, please refer to [examples/grpo_trainer/run_deepseek_v3_671b_megatron.sh](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_deepseek_v3_671b_megatron.sh).

MTP and quantilization is disabled during RL training.

To train your project, configure the following environment variables based on the number of available GPUs. These are recommended settings and can be adjusted based on your specific hardware.
| num gpus | NNODES | TP | PP | EP | OFFLOAD_FRACTION | OFFLOAD_OPTIM | LAST_LAYER |
| -- | -- | -- | -- | -- | -- | -- | -- |
| 96 | 12 | 8 | 12 | 8 | 1. | False | 6 |
| 128 | 16 | 8 | 16 | 8 | 0.5 | True | 1 |
| 256 | 32 | 8 | 16 | 8 | 0. | True | 1 |
| 512 | 64 | 1 | 16 | 32 | 0 | True | 1 |

### Qwen3 235b

For Qwen3-235b, please refer to [examples/grpo_trainer/run_qwen3_235b_a22b_megatron.sh](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_235b_a22b_megatron.sh).

To train your project, configure the following environment variables based on the number of available GPUs. These are recommended settings and can be adjusted based on your specific hardware.
| num gpus | NNODES | TP | PP | EP | OFFLOAD_FRACTION | OFFLOAD_OPTIM | LAST_LAYER |
| -- | -- | -- | -- | -- | -- | -- | -- |
| 32 | 4 | 4 | 8 | 4 | 1. | False | 6 |
| 64 | 8 | 4 | 8 | 4 | 0.5 | True | 6 |
| 128 | 16 | 4 | 8 | 4 | 0 | True | 6 |
| 256 | 32 | 4 | 8 | 4 | 0 | True | 6 |

### Benchmark
Here are some benchmark results for DeepSeek / Qwen3-235B. All configurations match the recommended settings based on the number of GPUs.

| model | num gpus | mean response length | rollout time(s) | GPU memory(GB) | CPU memory(GB) | MFU | step time(s) |
| -- | -- | -- | -- | -- | -- | -- | -- |
| DeepSeek 671b | 96 | 1960 | 1050 | 66 | 1500 | 0.19 | 1700 |

### Qwen3-30B-A3B MOE

For Qwen3-30b, please refer to [examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh).

To train your project, configure the following environment variables based on the number of available GPUs. These are recommended settings and can be adjusted based on your specific hardware.
| num gpus | NNODES | TP | PP | EP | OFFLOAD_FRACTION | OFFLOAD_OPTIM | MFU |
| -- | -- | -- | -- | -- | -- | -- | -- | 
| 8 | 1 | 1 | 1 | 8 | 1. | True | 0.4 |
| 16 | 2 | 1 | 1 | 8 | 1. | True | 0.37 |
| 32 | 4 | 1 | 1 | 8 | 1. | True | 0.31 |


## Upcoming Optimizations

The community continue to optimize large MoE models further, ongoing efforts include:
- further optimizing memory consumption, and provide recommended/tuned configurations with various machine types
- optimizing long context RL training performance
- performance improvement with SGLang x Megatron

We invite the community to try and improve verl together. Get connected with us on [slack](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)/[wechat](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG)/[Github issues](https://github.com/verl-project/verl/issues/708)!

## Acknowledgement
@vermouth1992 @ISEEKYAN @ETOgaosion @yzlnew @ShareLer @BearBiscuit05 @ccclyu @ann-qin-lu @SwordFaith @zzong2006 @zhaochenyang20 @ocss884 @eric-haibin-lin @chenhaiq @techkang

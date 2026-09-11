# Supported tags

> Last updated: 08/19/2026.

A full list of tags that are supported with Verl on ascend.

---

## Latest Images

| Device | CANN Base Image | Inference Backend | Image Tag | Dockerfile |
|--------|-----------------|-------------------|-----------|------------|
| 910b | 9.0.0 | vLLM | `latest-vllm-910b-ubuntu` | [Dockerfile.ascend_9.1.0_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a2) |
| A3 | 9.0.0 | vLLM | `latest-vllm-a3-ubuntu` | [Dockerfile.ascend_9.0.1_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.1_a3) |
| 910b | 8.5.0 | SGLang | `latest-sglang-910b-ubuntu` | [Dockerfile.ascend.sglang_8.5.0_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a2) |
| A3 | 8.5.0 | SGLang | `latest-sglang-a3-ubuntu` | [Dockerfile.ascend.sglang_8.5.0_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a3) |

---

## verl Release Images

| Device | CANN Base Image | Inference Backend | verl release version | Image Tag | Dockerfile |
|--------|-----------------|-------------------|----------------------|-----------|------------|
| 910b | 9.0.0 | vLLM | v0.8.0 | `v0.8.0-cann9.0.0-torch_npu2.9.0.post2-910b-ubuntu22.04-py3.11-vllm` | [Dockerfile.ascend_9.0.0_a2_v0.8.0](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2_v0.8.0) |
| A3 | 9.0.0 | vLLM | v0.8.0 | `v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm` | [Dockerfile.ascend_9.0.0_a3_v0.8.0](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3_v0.8.0) |
| 910b | 8.5.0 | vLLM | v0.7.1 | `verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1` | [Dockerfile.ascend_8.5.0_a2_v0.7.1](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2_v0.7.1) |
| A3 | 8.5.0 | vLLM | v0.7.1 | `verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1` | [Dockerfile.ascend_8.5.0_a3_v0.7.1](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3_v0.7.1) |
| 910b | 8.3.RC1 | vLLM | v0.7.0 | `verl-8.3.rc1-910b-ubuntu22.04-py3.11-v0.7.0` | [Dockerfile.ascend_8.3.rc1_a3](https://github.com/verl-project/verl/blob/release/v0.7.0/docker/ascend/Dockerfile.ascend_8.3.rc1_a3) |
| A3 | 8.3.RC1 | vLLM | v0.7.0 | `verl-8.3.rc1-a3-ubuntu22.04-py3.11-v0.7.0` | [Dockerfile.ascend_8.3.rc1_a2](https://github.com/verl-project/verl/blob/release/v0.7.0/docker/ascend/Dockerfile.ascend_8.3.rc1_a2) |
| 910b | 8.2.RC1 | vLLM | v0.6.1 | `verl-8.2.rc1-910b-ubuntu22.04-py3.11-v0.6.1` | [Dockerfile.ascend_8.2.rc1_a2](https://github.com/verl-project/verl/blob/release/v0.6.1/docker/ascend/Dockerfile.ascend_8.2.rc1_a2) |
| A3 | 8.2.RC1 | vLLM | v0.6.1 | `verl-8.2.rc1-a3-ubuntu22.04-py3.11-v0.6.1` | [Dockerfile.ascend_8.2.rc1_a3](https://github.com/verl-project/verl/blob/release/v0.6.1/docker/ascend/Dockerfile.ascend_8.2.rc1_a3) |

---

## Model-Specific Images

| Device | CANN Base Image | Inference Backend | Model | Image Tag | Dockerfile |
|--------|-----------------|-------------------|-------|-----------|------------|
| 910b | 8.5.2 | vLLM | Qwen3.5 | `verl-8.5.2-910b-ubuntu22.04-py3.11-qwen3-5` | [Dockerfile.ascend_8.5.2_a2_qwen3-5](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a2_qwen3-5) |
| A3 | 8.5.2 | vLLM | Qwen3.5 | `verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5` | [Dockerfile.ascend_8.5.2_a3_qwen3-5](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a3_qwen3-5) |

---

## History Images

| Device | CANN Base Image | Inference Backend | Image Tag | Dockerfile |
|--------|-----------------|-------------------|-----------|------------|
| 910b | 9.0.0 | vLLM | `verl-9.0.0-910b-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend_9.0.0_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2) |
| A3 | 9.0.0 | vLLM | `verl-9.0.0-a3-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend_9.0.0_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3) |
| 910b | 8.5.0 | SGLang | `verl-sglang-8.5.0-910b-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend.sglang_8.5.0_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a2) |
| A3 | 8.5.0 | SGLang | `verl-sglang-8.5.0-a3-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend.sglang_8.5.0_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a3) |
| 910b | 9.0.0 | vLLM | `verl-9.0.0-910b-ubuntu22.04-py3.11-v0.8.0` | [Dockerfile.ascend_9.0.0_a2_v0.8.0](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2_v0.8.0) |
| A3 | 9.0.0 | vLLM | `verl-9.0.0-a3-ubuntu22.04-py3.11-v0.8.0` | [Dockerfile.ascend_9.0.0_a3_v0.8.0](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3_v0.8.0) |
| 910b | 8.5.0 | vLLM | `verl-8.5.0-910b-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend_8.5.0_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2) |
| A3 | 8.5.0 | vLLM | `verl-8.5.0-a3-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend_8.5.0_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3) |
| 910b | 8.3.RC1 | vLLM | `verl-8.3.rc1-910b-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend_8.3.rc1_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a2) |
| A3 | 8.3.RC1 | vLLM | `verl-8.3.rc1-a3-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend_8.3.rc1_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a3) |
| 910b | 8.3.RC1 | SGLang | `verl-sglang-8.3.rc1-910b-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend.sglang_8.3.rc1_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a2) |
| A3 | 8.3.RC1 | SGLang | `verl-sglang-8.3.rc1-a3-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend.sglang_8.3.rc1_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a3) |
| 910b | 8.2.RC1 | vLLM | `verl-8.2.rc1-910b-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend_8.2.rc1_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2) |
| A3 | 8.2.RC1 | vLLM | `verl-8.2.rc1-a3-ubuntu22.04-py3.11-latest` | [Dockerfile.ascend_8.2.rc1_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3) |

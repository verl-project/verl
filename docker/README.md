# Dockerfiles of verl

We provide pre-built Docker images for quick setup. And from this version, we utilize a new image release hierarchy for productivity and stability.

Start from v0.6.0, we use vllm and sglang release image as our base image.

Start from v0.7.0, since vllm/vllm-openai:v0.12.0 is a minimal image without some essential libraries, we use nvidia/cuda:12.9.1-devel-ubuntu22.04 as our base image for vllm.

## Base Image

- vLLM: https://hub.docker.com/r/nvidia/cuda
- SGLang: https://hub.docker.com/r/lmsysorg/sglang

## Application Image

Upon base image, the following packages are added:
- flash_attn
- Megatron-LM
- Apex
- TransformerEngine
- DeepEP

Latest docker file:
- [Dockerfile.stable.vllm](https://github.com/volcengine/verl/blob/main/docker/Dockerfile.stable.vllm)
- [Dockerfile.stable.sglang](https://github.com/volcengine/verl/blob/main/docker/Dockerfile.stable.sglang)

All pre-built images are available in dockerhub: https://hub.docker.com/r/verlai/verl. For example, `verlai/verl:sgl056.latest`, `verlai/verl:vllm012.dev3`.

You can find the latest images used for development and ci in our github workflows:
- [.github/workflows/vllm.yml](https://github.com/volcengine/verl/blob/main/.github/workflows/vllm.yml)
- [.github/workflows/sgl.yml](https://github.com/volcengine/verl/blob/main/.github/workflows/sgl.yml)


## Installation from Docker

After pulling the desired Docker image and installing desired inference and training frameworks, you can run it with the following steps:

1. Launch the desired Docker image and attach into it:

```sh
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
docker start verl
docker exec -it verl bash
```

2. If you use the images provided, you only need to install verl itself without dependencies:

```sh
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
```

[Optional] If you hope to switch between different frameworks, you can install verl with the following command:

```sh
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
pip3 install -e .[vllm]
pip3 install -e .[sglang]
```

## Release History

- 2026/01/17: update vllm stable image to torch==2.9.1, cudnn==9.16, deepep==1.2.1
- 2025/12/23: update vllm stable image to vllm==0.12.0; update sglang stable image to sglang==0.5.6
- 2025/11/18: update vllm stable image to vllm==0.11.1; update sglang stable image to sglang==0.5.5


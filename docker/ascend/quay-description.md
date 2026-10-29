# verl for Ascend

## Quick Reference

- verl is maintained by the [verl community](https://github.com/verl-project/verl).

- Where to get help

    - [Ascend verl Image Registry](https://quay.io/repository/ascend/verl?tab=tags&tag=latest)
    - [verl Documentation](https://verl.readthedocs.io/en/latest/)
    - [Ascend Tutorial](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial)
    - [Issue Tracker](https://github.com/verl-project/verl/issues)

---

## verl

verl (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient, and production-ready reinforcement learning training library for large language models. The Ascend container images provide evaluation environments for running verl on Ascend A2 and A3 hardware.

A2/A3 images are hosted at [quay.io/ascend/verl](https://quay.io/repository/ascend/verl?tab=tags&tag=latest).

---

## Supported Images and Dockerfile Links

### Image Naming Convention

general image names follow this pattern:

```text
latest-{inference_backend}-{npu_device_type}-{os_version}-{other_fields}
```

Release image names follow this pattern:

```text
{verl_release_version}-{CANN_version}-{torch_npu_version}[-{npu_device_type}-{os_version}]-{Python_version}[-{inference_backend}-{other_fields}]
```

```text
| Field                 | Example Values        | Description                               |
|-----------------------|-----------------------|-------------------------------------------|
| CANN_version          | 9.0.0, 8.5.0, 8.3.RC1 | CANN version                              |
| torch_npu_version     | torch_npu2.9.0.post2  | torch_npu_version                         |
| npu_device_type       | 910b, a3              | Target Ascend device                      |
| os_version            | ubuntu22.04           | Base operating system                     |
| python_version        | py3.11                | Python version                            |
| verl_release_version  | v0.7.1                | verl release version; release images only |
```

### Latest Images

**Device / CANN Base Image / Inference Backend / Image Tag / Dockerfile**

- A2 — 9.0.0 — vLLM — `latest-vllm-910b-ubuntu` — [Dockerfile.ascend_9.1.0_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a2) 
- A3 — 9.0.0 — vLLM — `latest-vllm-a3-ubuntu` — [Dockerfile.ascend_9.1.0_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.1_a3) 
- A2 — 8.5.0 — SGLang — `latest-sglang-910b-ubuntu` — [Dockerfile.ascend.sglang_8.5.0_a2](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a2) 
- A3 — 8.5.0 — SGLang — `latest-sglang-a3-ubuntu` — [Dockerfile.ascend.sglang_8.5.0_a3](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a3) 

### verl Latest  Release Images

**Device / CANN Base Image / Inference Backend / verl release version / Image Tag / Dockerfile**
-  A2 — 9.0.0 — vLLM — v0.8.0 — `v0.8.0-cann9.0.0-torch_npu2.9.0.post2-910b-ubuntu22.04-py3.11-vllm` — [Dockerfile.ascend_9.0.0_a2_v0.8.0](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2_v0.8.0) 
-  A3 — 9.0.0 — vLLM — v0.8.0 — `v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm` — [Dockerfile.ascend_9.0.0_a3_v0.8.0](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3_v0.8.0) 

For tags associated with historical versions, please refer to [Supported Tags](https://github.com/verl-project/verl/blob/main/docker/ascend/supported_tags.md).

In vLLM images, vLLM, vLLM Ascend, MindSpeed, Megatron-LM, and verl are installed from source, with their source directories under `/`.

In SGLang images, SGLang, MindSpeed, and verl are installed from source, with their source directories also under `/`.

---

## Quick Start


### Run a Container

```bash
docker run -dit \
    --ipc=host \
    --network host \
    --name {your_docker_name} \
    --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/sbin:/usr/sbin \
    -v /home:/home \
    -v /data:/data \
    {image_name}:{tag} \
    /bin/bash
```

Add `-v <host_path>:<container_path>` for any additional mounts.

The `--privileged` option grants extended permissions to the container and should be evaluated against your security requirements.

### Start and Enter the Container

```bash
docker start {your_docker_name}
docker exec -it {your_docker_name} bash
```

---

## Supported Hardware

```text
| Chip Series | Product Examples                      | Architecture   |
|-------------|---------------------------------------|----------------|
| Ascend 910B | Atlas 200T A2 Box16, Atlas 900 A2 PoD | ARM64 / x86_64 |
| Ascend A3   | Atlas 800T A3                         | ARM64 / x86_64 |
```
---

## Security Risks
When running containers with these images, be aware of the following security risks:

- Running as the `root` user: Containers run as the `root` user by default, which can introduce security risks. In production environments, it is recommended to create a non-privileged user to run the application.

- Lack of CPU and memory resource limits: Not setting resource limits may cause a container to consume excessive system resources and affect host performance. It is recommended to use the `--cpus` and `--memory` parameters to limit resource usage.

- Device use with `rwm` permissions: The NPU device is assigned read, write, and `mknod` permissions. Although these are required for functionality, the scope of permissions should be carefully evaluated in security-sensitive environments.

---

## Disclaimer

The Ascend Dockerfiles and images provided by verl are reference examples intended for evaluation and early-access use. For production deployment, please contact the official support channel.

---

## License

verl is distributed under the [Apache License 2.0](https://github.com/verl-project/verl/blob/main/LICENSE). Pre-installed packages and software components in the container images may be subject to their own licenses.

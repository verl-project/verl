---
name: env-check
description: 检查verl训练环境是否就绪，包括容器状态、依赖工具安装
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: training_preparation
enforce_mode: STRICT
strict_rules:
  - NO_SKIP: 每个Step必须依次执行，禁止跳过任何Step
  - SEQUENCE: 必须按Step 1→2→3→4→5顺序执行
---

# ENV-CHECK Skill

检查verl训练环境是否就绪，包括容器状态和必要依赖工具。

## 环境变量定义

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VERL_CONTAINER` | `verl` | 容器名称 |
| `HTTP_PROXY` | - | HTTP代理（可选，用于安装工具） |
| `HTTPS_PROXY` | - | HTTPS代理（可选，用于安装工具） |

## 调用模式

### 模式1: 从主skill调用（传参）

传递参数时不交互：
```bash
skill env-check(
  VERL_CONTAINER="verl",
  HTTP_PROXY="",
  HTTPS_PROXY=""
)
```

### 模式2: 独立使用（无参数）

未传递参数时默认使用 `verl` 作为容器名。

## 命令规范

### Docker Exec 正确写法

当命令中包含以下情况时，必须使用 `bash -c` 包装：

| 情况 | 错误写法 | 正确写法 |
|------|----------|----------|
| Glob模式 | `docker exec c ls -la /path/*.sh` | `docker exec c bash -c "ls -la /path/*.sh"` |
| 变量展开 | `docker exec c ls $DIR/` | `docker exec c bash -c "ls $DIR/"` |
| 管道 | `docker exec c cmd \| grep` | `docker exec c bash -c "cmd \| grep"` |

## 执行步骤

### 步骤1: 检查容器状态

```bash
docker ps --filter "name=${VERL_CONTAINER}" --filter "status=running"
```

- 容器不存在 → 输出 `container_not_found`
- 容器未运行 → 输出 `container_not_running`

### 步骤2: 构建docker exec命令

根据代理配置构建命令：
```bash
# 有代理
DOCKER_CMD="docker exec -e HTTP_PROXY=${HTTP_PROXY} -e HTTPS_PROXY=${HTTPS_PROXY} ${VERL_CONTAINER}"

# 无代理
DOCKER_CMD="docker exec ${VERL_CONTAINER}"
```

### 步骤3: 安装modelscope（如需要）

```bash
${DOCKER_CMD} which modelscope || \
  ${DOCKER_CMD} pip install modelscope==1.20.1 -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host repo.huaweicloud.com
```

### 步骤4: 基础依赖检查

在容器内执行版本检查：

```bash
# OS版本
${DOCKER_CMD} cat /etc/os-release | grep PRETTY_NAME

# Python版本
${DOCKER_CMD} python3 --version

# CANN版本
${DOCKER_CMD} cat /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info |grep ^version= |cut -d'=' -f 2 2>/dev/null || echo "N/A"

# HDK版本 (torch_npu)
${DOCKER_CMD} python3 -c 'import torch_npu; print(torch_npu.__version__)'

# 芯片类型
${DOCKER_CMD} npu-smi info -t board -i 0 -c 0 2>/dev/null | grep 'Chip Name' | awk '{print $4}' || echo "N/A"

# torch版本
${DOCKER_CMD} python3 -c 'import torch; print(torch.__version__)'

# verl版本
${DOCKER_CMD} pip show verl 2>/dev/null | grep Version || echo "N/A"

# vllm版本
${DOCKER_CMD} pip show vllm 2>/dev/null | grep Version || echo "N/A"

# vllm-ascend版本
${DOCKER_CMD} pip show vllm-ascend 2>/dev/null | grep Version || echo "N/A"

# torchvision版本
${DOCKER_CMD} python3 -c 'import torchvision; print(torchvision.__version__)'

# triton-ascend版本
${DOCKER_CMD} pip show triton-ascend 2>/dev/null | grep Version || echo "N/A"

# transformers版本
${DOCKER_CMD} python3 -c 'import transformers; print(transformers.__version__)'

# mindspeed版本
${DOCKER_CMD} pip show mindspeed 2>/dev/null | grep Version || echo "N/A"

# megatron-core版本
${DOCKER_CMD} pip show megatron-core 2>/dev/null | grep Version || echo "N/A"
```

### 步骤5: 展示结果

将收集到的信息整理成表格：

| 检查项 | 版本 | 状态 |
|--------|------|------|
| OS版本 | <value> | ✓/✗ |
| Python版本 | <value> | ✓/✗ |
| CANN版本 | <value> | ✓/✗ |
| HDK版本 | <value> | ✓/✗ |
| 芯片类型 | <value> | ✓/✗ |
| torch版本 | <value> | ✓/✗ |
| verl版本 | <value> | ✓/✗ |
| vllm版本 | <value> | ✓/✗ |
| vllm-ascend版本 | <value> | ✓/✗ |
| torchvision版本 | <value> | ✓/✗ |
| triton-ascend版本 | <value> | ✓/✗ |
| transformers版本 | <value> | ✓/✗ |
| mindspeed版本 | <value> | (可选) |
| megatron-core版本 | <value> | (可选) |

> **核心依赖**（OS, Python, CANN, HDK, torch, verl, vllm）必须存在; 如果是用sglang，sglang必须存在

## 输出状态

| 状态 | 说明 |
|------|------|
| ready | 环境就绪 |
| container_not_found | 容器不存在 |
| container_not_running | 容器未运行 |
| missing_required | 缺少必要依赖 |

## 错误提示

```
⚠️ 缺少必要依赖: <缺失列表>
请确保所有核心依赖已安装。
```

## 使用示例

### 示例1: 从主skill调用（不交互）

```
[主skill调用env-check]
参数: VERL_CONTAINER=verl, HTTP_PROXY=
执行: 检查容器状态 -> ✓
结果: ready ✓
```

### 示例2: 独立使用

```
用户: 检查环境
执行: container_verification -> verl运行中
显示: 环境检查结果表格
结果: ready ✓
```
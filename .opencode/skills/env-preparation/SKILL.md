---
name: env-preparation
description: 准备verl训练环境，包括拉取镜像、创建启动脚本、启动容器
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: environment-setup
enforce_mode: STRICT
strict_rules:
  - NO_SKIP: 每个Step必须依次执行，禁止跳过任何Step
  - SEQUENCE: 必须按Step 1→2→3→4→5→6顺序执行
---

# ENV-PREPARATION Skill

准备VERL训练环境，包括镜像拉取、启动脚本创建和容器启动。

## 环境变量定义

| 变量 | 宿主机默认值 | 容器内路径 | 说明 |
|------|-------------|------------|------|
| `VERL_IMAGE` | - | - | Docker镜像标签（可选，如未设置则自动选择） |
| `ROLLOUT_NAME` | - | - | 推理后端（vllm/sglang），用于自动选择镜像 |
| `HTTP_PROXY` | - | - | HTTP代理（可选） |
| `HTTPS_PROXY` | - | - | HTTPS代理（可选） |
| `VERL_WORK_DIR` | `/home/agenta` | `/workspace` | 用户工作目录 |
| `VERL_DATA_DIR` | `/home/datasets` | `/datasets` | 数据集目录 |
| `VERL_MODEL_DIR` | `/home/model_weights` | `/models` | 模型权重目录 |
| `VERL_CONTAINER` | `verl-{月日-时分}` | - | 容器名称（如verl-0420-1430） |
| `REUSE_CONTAINER` | `false` | - | 是否复用现有容器（true/false），true是复用，false是不复用 |

## 调用模式

### 模式1: 从主skill调用（传参）

传递参数时不交互，直接使用：
```bash
skill env-preparation(
  VERL_IMAGE="",  # 可选，如未设置则自动选择
  ROLLOUT_NAME="vllm",  # 推理后端，用于自动选择镜像
  HTTP_PROXY="",
  HTTPS_PROXY="",
  VERL_WORK_DIR="/home/agenta",
  VERL_DATA_DIR="/home/datasets",
  VERL_MODEL_DIR="/home/model_weights",
  VERL_CONTAINER="verl-$(date +"%m%d-%H%M")",  # 默认带时间戳
  REUSE_CONTAINER=false  # 是否复用现有容器，默认false，表示不复用
)
```

### 模式2: 独立使用（无参数）

未传递参数时，交互获取配置：

| 问题 | 选项/示例 | 默认值 |
|------|-----------|--------|
| Docker镜像 | verl-8.5.0-910b-ubuntu22.04-py3.11-latest | - |
| 工作目录 | /home/agenta | /home/agenta |
| 数据集目录 | /home/datasets | /home/datasets |
| 模型权重目录 | /home/model_weights | /home/model_weights |
| 代理（可选） | http://127.0.0.1:58232 | 不使用 |

## 执行步骤

**STRICT_RULE**: 必须严格按照以下步骤执行，禁止自己揣测是否要复用容器

### 步骤1: 检查参数并自动选择镜像

如果用户提供了 `VERL_IMAGE`，直接使用，跳过自动选择。

如果未提供 `VERL_IMAGE`，则根据 `ROLLOUT_NAME`（推理后端）自动选择：

**a) 自动查询芯片类型：**
```bash
# 在宿主机上执行（不需要容器）
npu-smi info -t board -i 0 -c 0 2>/dev/null | grep 'Chip Name' | awk '{print $4}'
```

映射规则：
| 查询结果 | 芯片类型 | 镜像中的写法 |
|----------|---------|-------------|
| Ascend 910B / 910B | A2 | 910b |
| Ascend 910B Pro | A2 | 910b |
| Ascend 910B1 | A2 | 910b |
| Ascend 910B2 | A2 | 910b |
| Ascend 910B3 | A2 | 910b |
| Ascend 910c | A3 | a3 |
| Ascend A3 | A3 | a3 |

**b) 根据推理后端 + 芯片类型匹配镜像（选择最新CANN版本）：**

| 推理后端 | 芯片类型 | 镜像标签 |
|----------|---------|----------|
| vllm | A2 | verl-8.5.0-910b-ubuntu22.04-py3.11-latest |
| vllm | A3 | verl-8.5.0-a3-ubuntu22.04-py3.11-latest |
| sglang | A2 | verl-sglang-8.3.rc1-910b-ubuntu22.04-py3.11-latest |
| sglang | A3 | verl-sglang-8.3.rc1-a3-ubuntu22.04-py3.11-latest |

**c) 如果无法匹配（如未知芯片类型或查询失败），退回交互模式：**
```
提示信息：
"根据您的配置（推理后端: xxx, 芯片类型: xxx），未找到合适的镜像。
可选镜像列表：
- verl-8.5.0-910b-ubuntu22.04-py3.11-latest (vllm, A2)
- verl-8.5.0-a3-ubuntu22.04-py3.11-latest (vllm, A3)
- verl-sglang-8.3.rc1-910b-ubuntu22.04-py3.11-latest (sglang, A2)
- verl-sglang-8.3.rc1-a3-ubuntu22.04-py3.11-latest (sglang, A3)
请选择镜像："
```

### 步骤2: 检查本地镜像

```bash
docker images ${VERL_IMAGE} --format "{{.Repository}}:{{.Tag}}"
```

### 步骤3: 拉取镜像（如需要）

```bash
# 不使用代理
docker pull ${VERL_IMAGE}

# 使用代理
docker -e HTTP_PROXY=$HTTP_PROXY -e HTTPS_PROXY=$HTTPS_PROXY pull ${VERL_IMAGE}
```

### 步骤4: 检查宿主机目录

```bash
# 检查工作目录是否存在
if [ ! -d "${VERL_WORK_DIR}" ]; then
  echo "错误: 工作目录不存在: ${VERL_WORK_DIR}"
  exit 1
fi

# 检查数据集目录是否存在
if [ ! -d "${VERL_DATA_DIR}" ]; then
  echo "警告: 数据集目录不存在，将自动创建: ${VERL_DATA_DIR}"
  mkdir -p "${VERL_DATA_DIR}"
fi

# 检查模型目录是否存在
if [ ! -d "${VERL_MODEL_DIR}" ]; then
  echo "警告: 模型目录不存在，将自动创建: ${VERL_MODEL_DIR}"
  mkdir -p "${VERL_MODEL_DIR}"
fi
```

### 步骤5: 创建启动脚本

```bash
[ -f "${VERL_WORK_DIR}/start_verl_docker.sh" ] && rm "${VERL_WORK_DIR}/start_verl_docker.sh"

cat <<EOF > ${VERL_WORK_DIR}/start_verl_docker.sh
#!/bin/bash

: \${VERL_IMAGE:="quay.io/ascend/verl:verl-8.5.0-910b-ubuntu22.04-py3.11-latest"}
: \${VERL_CONTAINER:="verl"}
: \${VERL_MODEL_DIR:="/home/model_weights"}
: \${VERL_DATA_DIR:="/home/datasets"}
: \${VERL_WORK_DIR:="/home/agenta"}

docker run -dit --ipc=host --network host --name \${VERL_CONTAINER} --privileged \\
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \\
    -v /usr/local/sbin/:/usr/local/sbin/ \\
    -v \${VERL_MODEL_DIR}:/models \\
    -v \${VERL_DATA_DIR}:/datasets \\
    -v \${VERL_WORK_DIR}:/workspace \\
    \${VERL_IMAGE} /bin/bash
EOF
chmod +x ${VERL_WORK_DIR}/start_verl_docker.sh
```

### 步骤6: 容器冲突检测与创建

```bash
# 6.1 生成容器名（如果未设置）
if [ -z "$VERL_CONTAINER" ]; then
  VERL_CONTAINER="verl-$(date +"%m%d-%H%M")"
fi

# 6.2 检查是否有重名容器
EXISTING=$(docker ps -a --filter "name=${VERL_CONTAINER}" --format "{{.Names}}" 2>/dev/null || echo "")
EXISTING_STATUS=$(docker ps -a --filter "name=${VERL_CONTAINER}" --format "{{.Status}}" 2>/dev/null || echo "")

# 6.3 REUSE_CONTAINER 默认值为 false
if [ -z "$REUSE_CONTAINER" ]; then
  REUSE_CONTAINER="false"
fi

# 6.4 冲突处理
if [ -n "$EXISTING" ]; then
  if [ "$REUSE_CONTAINER" = "true" ]; then
    # 尝试复用现有容器
    if echo "$EXISTING_STATUS" | grep -q "Up"; then
      echo "复用现有容器: ${VERL_CONTAINER} (running)"
      exit 0
    else
      echo "启动现有容器: ${VERL_CONTAINER}"
      docker start ${VERL_CONTAINER}
      exit 0
    fi
  else
    # 强制报错，不覆盖
    echo "错误: 容器 ${VERL_CONTAINER} 已存在（状态: ${EXISTING_STATUS}）"
    echo "如需复用现有容器，请在调用时设置 REUSE_CONTAINER=true"
    exit 1
  fi
fi

# 6.4 无冲突，正常创建
# 更新启动脚本中的容器名
sed -i "s/:\${VERL_CONTAINER:=.*}/:\${VERL_CONTAINER:=\"${VERL_CONTAINER}\"}/" ${VERL_WORK_DIR}/start_verl_docker.sh

# 启动新容器
bash ${VERL_WORK_DIR}/start_verl_docker.sh

echo "创建新容器: ${VERL_CONTAINER} ✓"
```

## 默认镜像配置

| CANN版本 | 芯片 | 推理后端 | 镜像标签 |
|----------|------|----------|----------|
| 8.5.0 | A2 (910b) | vllm | verl-8.5.0-910b-ubuntu22.04-py3.11-latest |
| 8.5.0 | A3 | vllm | verl-8.5.0-a3-ubuntu22.04-py3.11-latest |
| 8.3.RC1 | A2 (910b) | sglang | verl-sglang-8.3.rc1-910b-ubuntu22.04-py3.11-latest |
| 8.3.RC1 | A3 | sglang | verl-sglang-8.3.rc1-a3-ubuntu22.04-py3.11-latest |

> 完整镜像版本列表参考: `references/image_versions.md`

## 输出状态

| 状态 | 说明 |
|------|------|
| ready | 环境就绪 |
| pulling | 拉取镜像中 |
| failed | 失败 |

## 错误处理

| 错误 | 处理方式 |
|------|----------|
| 镜像标签无效 | 检查镜像版本列表 |
| 拉取失败 | 检查网络/代理，重试 |
| 容器启动失败 | 检查Docker状态 |
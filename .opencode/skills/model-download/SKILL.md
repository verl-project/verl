---
name: model-download
description: 从ModelScope或HuggingFace下载模型权重
license: MIT
compatibility: opencode
metadata:
    audience: developers
    workflow: model_download
---

# MODEL-DOWNLOAD Skill

从ModelScope或HuggingFace下载模型权重。

## 环境变量定义

| 变量 | 宿主机默认值 | 容器内路径 | 说明 |
|------|-------------|------------|------|
| `MODEL_NAME` | - | - | 模型名称（必须，如未设置则交互获取） |
| `VERL_MODEL_DIR` | `/home/model_weights` | `/models` | 模型保存目录 |
| `HTTP_PROXY` | - | - | HTTP代理（可选） |
| `HTTPS_PROXY` | - | - | HTTPS代理（可选） |
| `VERL_CONTAINER` | `verl` | - | 容器名称 |
| `DOWNLOAD_SOURCE` | `modelscope` | - | 下载源：modelscope / huggingface |

## 调用模式

### 模式1: 从主skill调用（传参）

传递参数时不交互：
```bash
skill model-download(
  MODEL_NAME="Qwen2.5-0.5B-Instruct",
  VERL_MODEL_DIR="/home/model_weights",
  HTTP_PROXY="",
  HTTPS_PROXY="",
  VERL_CONTAINER="verl",
  DOWNLOAD_SOURCE="modelscope"
)
```

### 模式2: 独立使用（无参数）

未传递参数时，交互获取配置：

| 问题 | 选项/示例 | 默认值 |
|------|-----------|--------|
| 模型名称 | Qwen2.5-0.5B-Instruct 等 | - |
| 保存目录 | /home/model_weights | /home/model_weights |
| 下载源 | ModelScope / HuggingFace | ModelScope |
| 代理（可选） | http://127.0.0.1:58232 | 不使用 |

## 命令规范

### Docker Exec 正确写法

当命令中包含以下情况时，必须使用 `bash -c` 包装：

| 情况 | 错误写法 | 正确写法 |
|------|----------|----------|
| Glob模式 | `docker exec c ls -la /path/*.sh` | `docker exec c bash -c "ls -la /path/*.sh"` |
| 变量展开 | `docker exec c ls $DIR/` | `docker exec c bash -c "ls $DIR/"` |
| 管道 | `docker exec c cmd \| grep` | `docker exec c bash -c "cmd \| grep"` |

## 执行步骤

### 步骤1: 检查参数

如果 `MODEL_NAME` 未设置，则进入交互模式。

### 步骤2: 检查容器状态

```bash
docker ps --filter "name=${VERL_CONTAINER}" --filter "status=running" || {
  echo "错误: 容器 ${VERL_CONTAINER} 未运行，请先启动容器"
  exit 1
}
```

### 步骤4: 检查本地模型（如已有）

```bash
docker exec ${VERL_CONTAINER} bash -c "ls -la /models/${MODEL_NAME}/"
```

- 文件存在 → 验证后返回 `ready`
- 文件不存在 → 继续下载

### 步骤5: 下载模型

```bash
# 构建docker命令
if [ -n "$HTTP_PROXY" ]; then
  DOCKER_CMD="docker exec -e http_proxy=${HTTP_PROXY} -e https_proxy=${HTTPS_PROXY} ${VERL_CONTAINER}"
else
  DOCKER_CMD="docker exec ${VERL_CONTAINER}"
fi

# ModelScope下载
timeout 7200 $DOCKER_CMD modelscope download --model ${MODEL_ID} --local_dir /models/MODEL_NAME

# HuggingFace下载
timeout 7200 $DOCKER_CMD hf download ${MODEL_ID} --local-dir /models/MODEL_NAME
```

### 步骤6: 验证模型

必需文件：`config.json`, `model.safetensors`, `tokenizer.json`

## 常用模型ID

### ModelScope

| 模型 | model_id |
|------|--------|
| Qwen2.5-0.5B-Instruct | qwen/Qwen2.5-0.5B-Instruct |
| Qwen2.5-7B-Instruct | qwen/Qwen2.5-7B-Instruct |
| Qwen3-8B | qwen/Qwen3-8B |
| Qwen3.5-4B | qwen/Qwen3.5-4B |

### HuggingFace

| 模型 | model_id |
|------|--------|
| Qwen2.5-0.5B-Instruct | Qwen/Qwen2.5-0.5B-Instruct |
| Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct |
| Qwen3-8B | Qwen/Qwen3-8B |
| Qwen3.5-4B | Qwen/Qwen3.5-4B |

## 输出状态

| 状态 | 说明 |
|------|------|
| ready | 模型已就绪 |
| downloading | 下载中 |
| failed | 下载失败 |

## 错误处理

| 错误 | 处理方式 |
|------|----------|
| container_not_running | 容器未运行 |
| model_not_found | 模型不存在，检查model_id |
| download_failed | 下载失败，检查网络/代理 |
| verification_failed | 验证失败，重新下载 |

## 使用示例

### 示例1: 从主skill调用

```
[主skill调用model-download]
参数: MODEL_NAME=Qwen2.5-0.5B-Instruct, VERL_MODEL_DIR=/home/model_weights
执行: skill model-download(MODEL_NAME, VERL_MODEL_DIR, HTTP_PROXY, HTTPS_PROXY, VERL_CONTAINER)
结果: ready ✓
```

### 示例2: 独立使用

```
用户: 下载模型
助手: 请提供：
- 模型名称: Qwen2.5-0.5B-Instruct
- 保存目录: [/home/model_weights]
- 下载源: [ModelScope]
- 代理（可选）

用户: Qwen2.5-0.5B-Instruct
执行: modelscope download
结果: ready ✓
```
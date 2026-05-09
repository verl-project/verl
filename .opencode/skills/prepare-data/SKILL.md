---
name: prepare-data
description: 在verl容器中下载并处理数据集为parquet格式
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: data_preprocessing
enforce_mode: STRICT
strict_rules:
  - NO_SKIP: 每个Step必须依次执行，禁止跳过任何Step
  - SEQUENCE: 必须按Step 1→2→3→4→5顺序执行
---

# PREPARE-DATA Skill

在verl容器中下载并处理数据集为parquet格式。

## 环境变量定义

| 变量 | 宿主机默认值 | 容器内路径 | 说明 |
|------|-------------|------------|------|
| `DATASET_NAME` | - | - | 数据集名称（必须，如未设置则交互获取） |
| `VERL_DATA_DIR` | `/home/datasets` | `/datasets` | 数据集输出目录 |
| `HTTP_PROXY` | - | - | HTTP代理（可选） |
| `HTTPS_PROXY` | - | - | HTTPS代理（可选） |
| `VERL_CONTAINER` | `verl` | - | 容器名称 |

## 调用模式

### 模式1: 从主skill调用（传参）

传递参数时不交互：
```bash
skill prepare-data(
  DATASET_NAME="gsm8k",
  VERL_DATA_DIR="/home/datasets",
  HTTP_PROXY="",
  HTTPS_PROXY="",
  VERL_CONTAINER="verl"
)
```

### 模式2: 独立使用（无参数）

未传递参数时，交互获取配置：

| 问题 | 选项/示例 | 默认值 |
|------|-----------|--------|
| 数据集名称 | gsm8k / geo3k / math 等 | - |
| 输出目录 | /home/datasets | /home/datasets |
| 代理（可选） | http://127.0.0.1:58232 | 不使用 |

## 支持的数据集

| 数据集 | 说明 |
|--------|------|
| gsm8k | 数学推理数据集 |
| gsm8k multiturn sft| 数学推理数据集 |
| math | 数学竞赛数据集 |
| hellaswag | 常识推理数据集 |
| full_hh_rlhf | 有趣对话数据集 |
| geo3k | 地理知识数据集 |
| pokemon | 宝可梦知识数据集 |
| dapo_math_17k | 多轮数学推理数据集 |
| aime_2024 | AIME 2024数学竞赛数据集 |

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

如果 `DATASET_NAME` 未设置，则进入交互模式。

### 步骤2: 检查容器状态

```bash
docker ps --filter "name=${VERL_CONTAINER}" --filter "status=running" || {
  echo "错误: 容器 ${VERL_CONTAINER} 未运行，请先启动容器"
  exit 1
}
```

### 步骤3: 检查数据是否已存在

```bash
EXISTING_FILES=$(docker exec ${VERL_CONTAINER} bash -c "ls -1 /datasets/${DATASET_NAME}" 2>/dev/null | grep -E "\.parquet$" || true)
if [ -n "$EXISTING_FILES" ]; then
  echo "数据已存在，跳过下载"
  SKIP_DOWNLOAD=true
else
  echo "数据不存在，需要下载"
  SKIP_DOWNLOAD=false
fi
```

### 步骤4: 执行数据处理

```bash
#创建输出目录
docker exec ${VERL_CONTAINER} mkdir -p /datasets/${DATASET_NAME}

# 构建docker命令
if [ -n "$HTTP_PROXY" ]; then
  DOCKER_CMD="docker exec -e http_proxy=${HTTP_PROXY} -e https_proxy=${HTTPS_PROXY} ${VERL_CONTAINER}"
else
  DOCKER_CMD="docker exec ${VERL_CONTAINER}"
fi

if [ "$SKIP_DOWNLOAD" = "true" ]; then
  echo "跳过下载，保留现有数据"
else
  # 备份可能存在的旧数据
  if [ -n "$(docker exec ${VERL_CONTAINER} bash -c "ls -1 /datasets/${DATASET_NAME}/" 2>/dev/null | grep -E '\.parquet$')" ]; then
    BACKUP_DIR="/datasets/${DATASET_NAME}_backup_$(date +%Y%m%d_%H%M%S)"
    docker exec ${VERL_CONTAINER} mv /datasets/${DATASET_NAME} ${BACKUP_DIR}
    echo "已备份旧数据到: ${BACKUP_DIR}"
    docker exec ${VERL_CONTAINER} mkdir -p /datasets/${DATASET_NAME}
  fi
  # 确定脚本名称（部分数据集需要特殊映射）
  case "$DATASET_NAME" in
    dapo_math_17k) SCRIPT_NAME="dapo_multiturn_w_tool.py" ;;
    aime_2024) SCRIPT_NAME="aime2024_multiturn_w_tool.py" ;;
    *) SCRIPT_NAME="${DATASET_NAME}.py" ;;
  esac
  # 执行数据处理
  timeout 7200 $DOCKER_CMD python3 /verl/examples/data_preprocess/${SCRIPT_NAME} --local_save_dir /datasets/${DATASET_NAME}
fi
```

> `full_hh_rlhf` 需要额外指定 split：`--split <sft|rm|rl>`

### 步骤5: 验证输出

```bash
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  # 检查parquet文件是否存在
  FILE_COUNT=$(docker exec ${VERL_CONTAINER} bash -c "ls -1 /datasets/${DATASET_NAME}/" 2>/dev/null | grep -E "\.parquet$" | wc -l)

  # 确定需要的最少文件数（dapo_math_17k和aime_2024只生成train.parquet）
  case "$DATASET_NAME" in
    dapo_math_17k|aime_2024) MIN_FILES=1 ;;
    *) MIN_FILES=2 ;;
  esac

  if [ "$FILE_COUNT" -ge "$MIN_FILES" ]; then
    # 检查文件是否可读
    READABLE=true
    for f in train.parquet test.parquet; do
      if docker exec ${VERL_CONTAINER} test -e /datasets/${DATASET_NAME}/${f} 2>/dev/null; then
        if ! docker exec ${VERL_CONTAINER} test -r /datasets/${DATASET_NAME}/${f} 2>/dev/null; then
          READABLE=false
          break
        fi
      fi
    done

    if [ "$READABLE" = "true" ]; then
      echo "验证成功: ${FILE_COUNT} 个parquet文件"
      exit 0
    fi
  fi
  
  RETRY_COUNT=$((RETRY_COUNT + 1))
  if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
    echo "验证失败 (${RETRY_COUNT}/${MAX_RETRIES})，重新下载..."
    BACKUP_DIR="/datasets/${DATASET_NAME}_backup_retry_${RETRY_COUNT}_$(date +%Y%m%d_%H%M%S)"
    docker exec ${VERL_CONTAINER} mv /datasets/${DATASET_NAME} ${BACKUP_DIR} 2>/dev/null || true
    docker exec ${VERL_CONTAINER} mkdir -p /datasets/${DATASET_NAME}
    $DOCKER_CMD python3 /verl/examples/data_preprocess/${DATASET_NAME}.py --local_save_dir /datasets/${DATASET_NAME}
  fi
done

echo "验证失败: 超过最大重试次数 (${MAX_RETRIES})"
echo "数据目录: /datasets/${DATASET_NAME}"
exit 1
```

## 输出状态

| 状态 | 说明 |
|------|------|
| ready | 数据已就绪，验证通过 |
| processing | 处理中 |
| failed | 验证失败，超过最大重试次数 |

## 错误处理

| 错误 | 处理方式 |
|------|----------|
| container_not_running | 容器未运行，先启动容器 |
| dataset_not_found | 数据集名称错误，检查支持列表 |
| download_failed | 下载失败，检查网络/代理 |
| output_missing | 输出文件缺失，重新处理 |
| data_already_exists | 数据已存在，无需重新下载 |
| verification_failed | 验证失败（文件不存在或不可读） |

## 使用示例

### 示例1: 从主skill调用（传参）

```
[主skill调用prepare-data]
参数: DATASET_NAME=gsm8k, VERL_DATA_DIR=/home/datasets
执行: skill prepare-data(DATASET_NAME, VERL_DATA_DIR, HTTP_PROXY, HTTPS_PROXY, VERL_CONTAINER)
结果: ready ✓
```

### 示例2: 独立使用

```
用户: 下载数据集
助手: 请提供：
- 数据集名称: gsm8k
- 输出目录: [/home/datasets]

用户: gsm8k
执行: python3 /verl/examples/data_preprocess/gsm8k.py --local_save_dir /datasets/gsm8k
结果: train.parquet, test.parquet ✓
```
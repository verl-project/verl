---
name: verl-training
description: 拉起verl训练，包含环境准备、环境检查、数据集处理、模型下载和训练执行；触发关键词：verl训练、使用verl拉起训练
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: training
enforce_mode: STRICT
strict_rules:
  - NO_SKIP: 每个Step必须依次执行，禁止跳过任何Step
  - SEQUENCE: 必须按Step 1→2→3→4→5→6→7顺序执行
  - VERIFY: 每个Step执行前必须验证前序Step已完成
  - ALWAYS_USE_SKILL: 所有子操作必须通过skill命令调用，禁止直接执行bash
---

# VERL训练 skill (主skill)

模块化设计的主skill，调用其他subskill完成训练。所有配置在开始时一次性收集，后续子skill调用时传递参数不交互。

**STRICT_RULE**: 一定仔细阅读 环境变量定义 命令规范 模块结构

## 环境变量定义

| 变量 | 宿主机默认值 | 容器内路径 | 说明 |
|------|-------------|------------|------|
| `VERL_IMAGE` | - | - | Docker镜像标签 |
| `VERL_CONTAINER` | `verl-{月日-时分}` | - | 容器名称（如verl-0420-1430） |
| `REUSE_CONTAINER` | `false` | - | 是否复用现有容器（true/false），true是复用，false是不复用 |
| `VERL_WORK_DIR` | `/home/agenta` | `/workspace` | 工作目录 |
| `VERL_DATA_DIR` | `/home/datasets` | `/datasets` | 数据集目录 |
| `VERL_MODEL_DIR` | `/home/model_weights` | `/models` | 模型权重目录 |
| `VERL_LOG_DIR` | `$VERL_WORK_DIR/logs` | `/workspace/logs` | 日志输出目录 |
| `HTTP_PROXY` | - | - | HTTP代理（可选） |
| `HTTPS_PROXY` | - | - | HTTPS代理（可选） |
| `ASCEND_RT_VISIBLE_DEVICES` | 未设置 | - | GPU设备列表，如 `4,5,6,7`（可选） |


## 命令规范

### Docker Exec 正确写法

当命令中包含以下情况时，必须使用 `bash -c` 包装：

| 情况 | 错误写法 | 正确写法 |
|------|----------|----------|
| Glob模式 | `docker exec c ls -la /path/*.sh` | `docker exec c bash -c "ls -la /path/*.sh"` |
| 变量展开 | `docker exec c ls $DIR/` | `docker exec c bash -c "ls $DIR/"` |
| 管道 | `docker exec c cmd \| grep` | `docker exec c bash -c "cmd \| grep"` |

## 模块结构

```
verl-training (主skill)
├── env-preparation: 环境准备 (拉取镜像、创建启动脚本、启动容器)
├── env-check: 环境检查 (容器状态、依赖工具)
├── prepare-data: 数据集处理
├── model-download: 模型下载
└── train-runner: 执行训练
```

## 强制约束（必须遵守）

1. **禁止创建新脚本**：必须使用Step 6.2查找到的原始脚本路径，禁止创建新脚本文件
2. **参数必须通过命令行传递**：所有参数必须组装为OVERRIDE_PARAMS，通过命令行$OVERRIDE_PARAMS传递，禁止写死在脚本里
3. **日志重定向**：按Step 6.8模板执行，禁止在脚本末尾添加重定向（原脚本无重定向则不添加）
4. **修改仅限原始脚本**：Step 6.4的修改是指用sed修改原始脚本文件，不是创建新脚本
5. **执行前必须朗读**：每个子步骤执行前必须朗读当前步骤定义，确认后再执行

## 交互流程（只在开始时交互一次）、

**STRICT_RULE**: 所有子操作必须通过skill命令调用，禁止直接执行bash

### Step 1: 收集训练配置（一次性交互）

**STRICT_RULE**: 必须先告诉用户当前算法支持现状，然后展示参数表格 + 示例说明，禁止跳过表格直接给简化示例或只给示例

接到用户做verl训练的需求后：
1. 首先必须告诉用户："当前算法支持现状请参考如下表格（暂时仅支持单机拉起训练）：https://verl.readthedocs.io/en/latest/ascend_tutorial/quick_start/ascend_quick_start.html#id11"
2. 接着必须展示"需要收集的参数"表格，供用户做参数选择参考
3. 接着展示"代理使用时机"表格，说明各步骤是否使用代理
4. 然后提供用户配置示例作为参考
5. 用户只需要参考示例格式，一次性说明所有配置（使用列表形式），不需要从选项中选择：

**需要收集的参数（共12项）：**

| 参数 | 英文名 | 说明 | 示例 | 默认值 |
|------|--------|------|------|--------|
| 算法 | ALGORITHM | GRPO/DAPO/PPO/SFT-PEFT/ReTool-SFT | GRPO | GRPO |
| 模型名称 | MODEL_NAME | 模型名称 | Qwen2.5-0.5B-Instruct | - |
| 数据集名称 | DATASET_NAME | 数据集名称 | gsm8k | - |
| 训练后端 | actor.strategy | FSDP/MindSpeed_LLM/Megatron/FSDP2 | FSDP | FSDP |
| 推理后端 | INFER_ENGINE | vllm/sglang | vllm | vllm |
| 工作目录 | VERL_WORK_DIR | 宿主机工作目录 | /home/agenta | /home/agenta |
| 数据集目录 | VERL_DATA_DIR | 宿主机数据集目录 | /home/datasets | /home/datasets |
| 模型权重目录 | VERL_MODEL_DIR | 宿主机模型权重目录 | /home/model_weights | /home/model_weights |
| GPU设备 | ASCEND_RT_VISIBLE_DEVICES | 可选，如4,5,6,7，留空使用所有卡 | 4,5,6,7 | 留空 |
| 代理 | HTTP_PROXY/HTTPS_PROXY | 可选，如http://127.0.0.1:58232 | http://127.0.0.1:58232 | 留空 |
| 代理使用时机 | PROXY_USAGE | 见下方步骤说明 | 数据集下载时使用 | 仅数据集处理使用 |
| 复用容器 | REUSE_CONTAINER | 用户说"复用"时为true，不说就是不复用，为false | 复用现有容器 | false |

**代理使用时机（步骤）：**

| 步骤 | 默认 | 说明 |
|------|------|------|
| 环境准备 | NO | 拉取镜像不使用代理 |
| 环境检查 | NO | 检查依赖不使用代理 |
| 数据集处理 | YES | 下载数据集使用代理 |
| 模型下载 | NO | 下载模型权重不使用代理 |
| 执行训练 | NO | 训练时不使用代理 |

**示例（请按此格式提供您的配置）：**

```
- 算法: GRPO
- 模型名称: Qwen2.5-0.5B-Instruct
- 数据集名称: gsm8k
- 训练后端: FSDP
- 推理后端: vllm
- 工作目录: /home/agenta
- 数据集目录: /home/datasets
- 模型权重目录: /home/model_weights
- GPU设备: 4,5,6,7
- 代理: http://127.0.0.1:58232
- 代理使用时机: 数据集下载时使用，其他步骤不使用
- 复用容器
```

请输入您的训练配置：

### Step 2: 环境准备

ENFORCE_CHECK:
  - PRECONDITION: Step 1 已收集完整配置（ALGORITHM, MODEL_NAME, DATASET_NAME, actor.strategy, INFER_ENGINE, VERL_WORK_DIR, VERL_DATA_DIR, VERL_MODEL_DIR, ASCEND_RT_VISIBLE_DEVICES, HTTP_PROXY, PROXY_USAGE, REUSE_CONTAINER）
  - FAIL_ACTION: 配置不完整时禁止进入Step 2
  - USE_SKILL: 必须使用 "skill env-preparation(...)" 调用，禁止直接执行bash

在收集用户配置时，检测用户是否提到"复用"关键词（如"复用"、"使用之前的容器"等），设置 `REUSE_CONTAINER=true`，否则为 `false`。

调用 `env-preparation` skill，传递参数：
```bash
skill env-preparation(VERL_IMAGE, HTTP_PROXY, HTTPS_PROXY, VERL_WORK_DIR, VERL_DATA_DIR, VERL_MODEL_DIR, VERL_CONTAINER, ROLLOUT_NAME, REUSE_CONTAINER)
```

其中：
- `ROLLOUT_NAME` 为推理后端（vllm/sglang），用于自动选择合适的镜像
- `REUSE_CONTAINER` 为是否复用现有容器（用户说"复用"时为true，否则为false）

### Step 3: 环境检查

ENFORCE_CHECK:
  - PRECONDITION: Step 2 已完成，VERL_CONTAINER 容器已启动
  - VERIFY: docker ps | grep $VERL_CONTAINER 返回running状态
  - FAIL_ACTION: 容器未就绪时禁止进入Step 3
  - USE_SKILL: 必须使用 "skill env-check(...)" 调用，禁止直接执行bash

调用 `env-check` skill，传递参数：
```bash
skill env-check(VERL_CONTAINER, HTTP_PROXY, HTTPS_PROXY)
```

### Step 4: 数据集处理

ENFORCE_CHECK:
  - PRECONDITION: Step 3 环境检查通过
  - FAIL_ACTION: 环境检查未通过时禁止进入Step 4
  - USE_SKILL: 必须使用 "skill prepare-data(...)" 调用，禁止直接执行bash

调用 `prepare-data` skill，传递参数：
```bash
skill prepare-data(DATASET_NAME, VERL_DATA_DIR, HTTP_PROXY, HTTPS_PROXY, VERL_CONTAINER)
```

### Step 5: 模型处理

ENFORCE_CHECK:
  - PRECONDITION: Step 4 数据集处理完成
  - FAIL_ACTION: 数据集处理未完成时禁止进入Step 5
  - USE_SKILL: 必须使用 "skill model-download(...)" 调用，禁止直接执行bash

调用 `model-download` skill，传递参数：
```bash
skill model-download(MODEL_NAME, VERL_MODEL_DIR, HTTP_PROXY, HTTPS_PROXY, VERL_CONTAINER)
```

### Step 6: 执行训练

**执行模式：SEQUENCE + TODOWRITE**
- 每个子步骤执行前必须使用todowrite更新状态
- 每个子步骤执行前必须朗读当前步骤定义
- 禁止跳过任何子步骤

ENFORCE_CHECK:
  - PRECONDITION: Step 5 模型处理完成
  - FAIL_ACTION: 模型处理未完成时禁止进入Step 6
  - NO_SKIP: 每个Step 6.x 必须依次执行
  - SEQUENCE: 必须按Step 6.1→6.2→6.3→6.4→6.5→6.6→6.7→6.8顺序执行

**STRICT_RULE**: 根据收集的配置执行训练，全部在**容器内**完成

#### step 6.1: 获取GPU卡数

[执行前状态]
todowrite: content="Step 6.1: 获取GPU卡数" status="in_progress"

[ENFORCE_CHECK]
- PRECONDITION: Step 5 模型处理完成
- FAIL_ACTION: step 6.1 未完成时禁止进入6.2

[执行前朗读]
获取GPU卡数 - 使用npu-smi info -l获取可用GPU数量
命令: GPU_COUNT=$(docker exec $VERL_CONTAINER bash -c "npu-smi info -l | grep 'Total Count' | awk -F: '{print \$2}' | tr -d ' '")

如果用户指定了用哪些卡，使用指定的卡数，否则使用下面命令查看
```bash
GPU_COUNT=$(docker exec $VERL_CONTAINER bash -c "npu-smi info -l | grep 'Total Count' | awk -F: '{print \$2}' | tr -d ' '")
```

[执行后状态]
todowrite: content="Step 6.1: 获取GPU卡数" status="completed"

#### step 6.2: 查找训练脚本

[执行前状态]
todowrite: content="Step 6.2: 查找训练脚本" status="in_progress"

[ENFORCE_CHECK]
- PRECONDITION: step 6.1 已完成
- VERIFY: GPU_COUNT 已设置
- FAIL_ACTION: step 6.1未完成时禁止进入6.2

[执行前朗读]
查找训练脚本 - 必须使用references/training-config.md中查到的原始脚本路径，**禁止创建新脚本**，**禁止复制到宿主机**

- 根据从用户收集的算法信息，在references/training-config.md中查找训练脚本路径（SCRIPT_PATH）
- 如果references/training-config.md中找不到脚本路径，可以参考 代码路径说明，到容器内代码路径下找

[执行后状态]
todowrite: content="Step 6.2: 查找训练脚本" status="completed"

#### step 6.3: 创建ckpts目录（如需要）

[执行前状态]
todowrite: content="Step 6.3: 创建ckpts目录" status="in_progress"

[ENFORCE_CHECK]
- PRECONDITION: step 6.2 已完成
- VERIFY: SCRIPT_PATH已找到
- FAIL_ACTION: step 6.2未完成时禁止进入6.3

[执行前朗读]
创建ckpts目录 - 从训练脚本提取project_name和exp_name，创建检查点保存目录

- 1. 提取project_name（如有）
```bash
if docker exec $VERL_CONTAINER cat $SCRIPT_PATH | grep -q "project_name="; then
    PROJECT_NAME=$(docker exec $VERL_CONTAINER cat $SCRIPT_PATH | grep "project_name=" | head -1 | cut -d"'" -f2)
    EXP_NAME=$(docker exec $VERL_CONTAINER cat $SCRIPT_PATH | grep "exp_name=" | head -1 | cut -d"'" -f2)
fi
```
- 2.根据 PROJECT_NAME、EXP_NAME创建ckpts目录（如需要）
```bash
if [ -n "$PROJECT_NAME" ]; then
    CKPT_PATH="/workspace/ckpts/${PROJECT_NAME}/${EXP_NAME}"
    docker exec $VERL_CONTAINER bash -c "mkdir -p $CKPT_PATH"
fi
```

[执行后状态]
todowrite: content="Step 6.3: 创建ckpts目录" status="completed"

#### step 6.4: 统一训练脚本格式

[执行前状态]
todowrite: content="Step 6.4: 统一训练脚本格式" status="in_progress"

[ENFORCE_CHECK]
- PRECONDITION: step 6.3 已完成
- VERIFY: PROJECT_NAME/CKPT_PATH已获取
- FAIL_ACTION: step 6.3未完成时禁止进入6.4

[执行前朗读]
统一训练脚本格式 - 检查并修改训练脚本中的5项格式问题

---

##### 6.4.1 检查阶段

**CHECK_LIST** 逐一检查并给出结论：

| 检查项 | 检查命令 | 需要修改的pattern |
|--------|----------|-------------------|
| 1. ${number}传参检查 | `grep -qF '\${1:-'` | 存在 `${1:-` 需改为 `${VAR:-默认值}` |
| 2. $@结尾检查 | `tail -1 $SCRIPT_PATH \| grep -q '\$@'` | 结尾无 `$@` 需添加 |
| 3. wandb检查 | `grep -q 'wandb'` | 存在 wandb 需移除 |
| 4. device_name检查 | `grep -q 'device_name.*cuda'` | cuda 需改为 npu |
| 5. 日志重定向检查 | `tail -1 $SCRIPT_PATH \| grep -q '2>&1 \| tee'` | 存在日志重定向时需确保$@在重定向之前，**不存在日志重定向时无需修改** |

执行检查命令：
```bash
# 读取脚本内容
docker exec $VERL_CONTAINER cat $SCRIPT_PATH

# 检查1: ${number}传参
docker exec $VERL_CONTAINER bash -c "grep -qF '\${1:-' $SCRIPT_PATH && echo '需要修改' || echo '无需修改'"

# 检查2: $@结尾
docker exec $VERL_CONTAINER bash -c "tail -1 $SCRIPT_PATH | grep -q '\$@' && echo '无需修改' || echo '需要修改'"

# 检查3: wandb
docker exec $VERL_CONTAINER bash -c "grep -q 'wandb' $SCRIPT_PATH && echo '需要修改' || echo '无需修改'"

# 检查4: device_name
docker exec $VERL_CONTAINER bash -c "grep -q 'device_name.*cuda' $SCRIPT_PATH && echo '需要修改' || echo '无需修改'"

# 检查5: 日志重定向
docker exec $VERL_CONTAINER bash -c "tail -1 $SCRIPT_PATH | grep -q '2>&1.*tee' && echo '需要检查$@' || echo '无需修改'"
```

---

##### 6.4.2 修改阶段

根据检查结果，训练脚本中有如下格式时需修改：

| 场景 | 原内容 | 修改为 |
|------|--------|--------|
| 场景1 | `ENGINE=${1:-vllm}` | `ENGINE=${ENGINE:-vllm}` |
| 场景2 | `trainer.val_before_train=False 2>&1 \| tee ${LOG_PATH}` | `trainer.val_before_train=False $@ 2>&1 \| tee ${LOG_PATH}` |
| 场景3 | `actor_rollout_ref...=True` (训练脚本结尾参数无$@) | `actor_rollout_ref...=True $@` |
| 场景4 | `trainer.logger='["console","wandb"]'` | `trainer.logger='["console"]'` |
| 场景5 | `device_name=${device_name:-cuda}` | `device_name=${device_name:-npu}` |

修改命令示例（根据检查结果选择性执行）：
```bash
# 修改1: ${number}传参
docker exec $VERL_CONTAINER bash -c "sed -i 's/\${1:-vllm}/\${ENGINE:-vllm}/g' $SCRIPT_PATH"

# 修改2: 结尾行有重定向，参数后面添加 $@
docker exec "$VERL_CONTAINER" bash -c \
  "sed -i 's#trainer\.val_before_train=False 2>&1 | tee \\\${LOG_PATH}#trainer.val_before_train=False \\\$@ 2>&1 | tee \\\${LOG_PATH}#g' '$SCRIPT_PATH'"

# 修改3: 结尾添加 $@
docker exec "$VERL_CONTAINER" bash -c \
  "sed -i 's#trainer.total_epochs=15#trainer.total_epochs=15 \\\$@#g' '$SCRIPT_PATH'"

# 修改4: wandb移除
docker exec $VERL_CONTAINER bash -c "sed -i 's/\"wandb\"//g; s/\[\"console\",\]/[\"console\"]/g' $SCRIPT_PATH"

# 修改5: device_name cuda改为npu
docker exec $VERL_CONTAINER bash -c "sed -i 's/device_name=\${device_name:-cuda}/device_name=\${device_name:-npu}/g' $SCRIPT_PATH"
```

[执行后状态]
todowrite: content="Step 6.4: 统一训练脚本格式" status="completed"

#### step 6.5: 命令行覆盖

[执行前状态]
todowrite: content="Step 6.5: 命令行覆盖参数" status="in_progress"

[ENFORCE_CHECK]
- PRECONDITION: step 6.4 已完成
- VERIFY: 5项CHECK_LIST已完成
- FAIL_ACTION: step 6.4未完成时禁止进入6.5

[执行前朗读]
命令行覆盖 - **必须通过OVERRIDE_PARAMS传递参数，禁止把参数写死在脚本里**

对如下参数进行命令行覆盖：
**常用命令行参数表**：

|  命令行覆盖参数 | 默认值 |
|---------|--------|
| actor_rollout_ref.model.path | - |
| critic.model.path | - |
| data.train_files | - |
| data.val_files | - |
| trainer.nnodes | 1 |
| trainer.n_gpus_per_node | 动态获取 |
| trainer.default_local_dir | 动态组装|
 

```bash
OVERRIDE_PARAMS=""
OVERRIDE_PARAMS="$OVERRIDE_PARAMS trainer.nnodes=1"
OVERRIDE_PARAMS="$OVERRIDE_PARAMS trainer.n_gpus_per_node=$GPU_COUNT"
OVERRIDE_PARAMS="$OVERRIDE_PARAMS actor_rollout_ref.model.path=/models/${MODEL_NAME}"
OVERRIDE_PARAMS="$OVERRIDE_PARAMS data.train_files=['/datasets/${DATASET_NAME}/train.parquet']"
OVERRIDE_PARAMS="$OVERRIDE_PARAMS data.val_files=['/datasets/${DATASET_NAME}/test.parquet']"
if [ -n "$CRITIC_MODEL_NAME" ]; then
    OVERRIDE_PARAMS="$OVERRIDE_PARAMS critic.model.path=${CRITIC_MODEL_NAME}"
fi
if [ -n "$CKPT_PATH" ]; then
    OVERRIDE_PARAMS="$OVERRIDE_PARAMS trainer.default_local_dir=${CKPT_PATH}"
fi

echo ${OVERRIDE_PARAMS}
```

[执行后状态]
todowrite: content="Step 6.5: 命令行覆盖参数" status="completed"

#### step 6.6: 构建Docker命令

[执行前状态]
todowrite: content="Step 6.6: 构建Docker命令" status="in_progress"

[ENFORCE_CHECK]
- PRECONDITION: step 6.5 已完成
- VERIFY: OVERRIDE_PARAMS已组装,非空
- FAIL_ACTION: step 6.5未完成时禁止进入6.6

[执行前朗读]
构建Docker命令 - 构造docker exec命令，设置日志路径和环境变量

```bash
docker exec $VERL_CONTAINER bash -c "mkdir -p /workspace/logs"
LOG_FILE=/workspace/logs/${MODEL_NAME}_${ALGORITHM}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log

DOCKER_CMD="docker exec $VERL_CONTAINER bash -c"

if [ -n "$ASCEND_RT_VISIBLE_DEVICES" ]; then
    DOCKER_CMD="$DOCKER_CMD -e ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
fi

echo ${DOCKER_CMD}
```

[执行后状态]
todowrite: content="Step 6.6: 构建Docker命令" status="completed"

#### step 6.7: 启动Ray（如需要）

[执行前状态]
todowrite: content="Step 6.7: 启动Ray" status="in_progress"

[ENFORCE_CHECK]
- PRECONDITION: step 6.6 已完成
- VERIFY: DOCKER_CMD已构建
- FAIL_ACTION: step 6.6未完成时禁止进入6.7

[执行前朗读]
启动Ray - 如果训练脚本需要ray，则启动ray cluster

**STRICT_RULE**: 严格按照下面命令执行，如果需要ray则启用，否则禁止启用
```bash
if docker exec $VERL_CONTAINER cat $SCRIPT_PATH | grep -q "ray job submit\|ray start"; then
    $DOCKER_CMD "ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --dashboard-agent-listen-port 51365"
    $DOCKER_CMD "for i in {1..30}; do ray status > /dev/null 2>&1 && break || sleep 2; done"
fi
```

[执行后状态]
todowrite: content="Step 6.7: 启动Ray" status="completed"

#### step 6.8: 执行训练

[执行前状态]
todowrite: content="Step 6.8: 执行训练" status="in_progress"

[ENFORCE_CHECK]
- PRECONDITION: step 6.7 已完成（或跳过6.7）
- VERIFY: DOCKER_CMD、OVERRIDE_PARAMS和LOG_FILE已就绪
- FAIL_ACTION: step 6.7未完成时禁止进入6.8

[执行前朗读]
执行训练 - 使用nohup后台运行训练脚本，日志输出到指定文件

**执行后状态**
todowrite: content="Step 6.8: 执行训练" status="completed"

**STRICT_RULE**: 必须用bash执行训练脚本，禁止直接用python命令执行训练命令

**禁止事项**：禁止在脚本末尾添加日志重定向，原脚本无重定向则不添加

```bash
$DOCKER_CMD "cd /verl && nohup bash $SCRIPT_PATH $OVERRIDE_PARAMS > $LOG_FILE 2>&1 &"
```
> **重要** 训练在Docker容器内执行时，**不设置** `http_proxy`/`https_proxy` 环境变量


### Step 7: 监控训练

ENFORCE_CHECK:
  - PRECONDITION: Step 6 训练已启动
  - FAIL_ACTION: 训练未启动时无法监控

训练过程中提供以下命令查看进度：
```bash
# 查看最新日志
tail -f $LOG_FILE

# 过滤关键指标
tail -f $LOG_FILE | grep -E "(Epoch|Step|Loss|Reward|accuracy)"
```

## 代码路径说明

| 类型 | 容器内路径 |
|------|-----------|
| 数据处理脚本 | `/verl/examples/data_preprocess/*.py` |
| Qwen2.5-0.5B-Instruct GRPO训练脚本 | `/workspace/.opencode/skills/verl-training/scripts/run_*.sh` |
| GRPO训练脚本 | `/verl/examples/grpo_trainer/*.sh` |
| PPO训练脚本 | `/verl/examples/ppo_trainer/*.sh` |
| DAPO训练脚本 | `/verl/recipe/dapo/*.sh` |
| One_Step_Off_Policy训练脚本 | `/verl/verl/experimental/one_step_off_policy/shell/*.sh` |
| CISPO训练脚本 | `/verl/examples/cispo_trainer/*.sh` |
| RLOO训练脚本 | `/verl/examples/rloo_trainer/*.sh` |
| SFT-PEFT训练脚本 | `/verl/examples/sft/gsm8k/*.sh` |
| ReTool-SFT训练脚本 | `/verl/recipe/retool/retool.py` |
| 训练入口 | `/verl/verl/trainer/main_ppo.py` |

## 错误处理

| 错误 | 处理方式 |
|------|----------|
| 镜像拉取失败 | 检查网络/代理设置，重试 |
| 容器启动失败 | 检查Docker状态 |
| 容器不存在 | 调用 env-preparation |
| 依赖缺失 | 调用 env-check 查看详情 |
| 数据集处理失败 | 查看日志，常见原因为代理或网络问题 |
| 模型下载失败 | 查看日志，检查代理配置 |
| 训练脚本不存在 | 根据上面 代码路径说明，或者references/training-config.md, 从容器 `/verl` 下实时获取脚本列表 |
| 训练执行失败 | 查看日志排查: `tail -f $LOG_FILE` |
| 模型不在支持矩阵 | 提示暂不支持，已支持模型列表见references |

## 使用示例

### 错误输出（❌ 禁止）

```
用户: 开始训练
助手: 请提供您的训练配置（按以下格式一次性说明）：

- 算法: GRPO
- 模型名称: Qwen2.5-0.5B-Instruct
- 数据集名称: gsm8k
...
请输入您的训练配置：
```

跳过表格直接给简化示例是**错误的输出** ❌

### 正确输出（✅ 允许）

```
用户: 开始训练
助手: 请提供您的训练配置：

**需要收集的参数（共12项）：**

| 参数 | 英文名 | 说明 | 示例 | 默认值 |
|------|--------|------|------|--------|
| 算法 | ALGORITHM | GRPO/DAPO/PPO/SFT-PEFT/ReTool-SFT | GRPO | GRPO |
| 模型名称 | MODEL_NAME | 模型名称 | Qwen2.5-0.5B-Instruct | - |
| 数据集名称 | DATASET_NAME | 数据集名称 | gsm8k | - |
| 训练后端 | actor.strategy | FSDP/MindSpeed_LLM/Megatron/FSDP2 | FSDP | FSDP |
| 推理后端 | INFER_ENGINE | vllm/sglang | vllm | vllm |
| 工作目录 | VERL_WORK_DIR | 宿主机工作目录 | /home/agenta | /home/agenta |
| 数据集目录 | VERL_DATA_DIR | 宿主机数据集目录 | /home/datasets | /home/datasets |
| 模型权重目录 | VERL_MODEL_DIR | 宿主机模型权重目录 | /home/model_weights | /home/model_weights |
| GPU设备 | ASCEND_RT_VISIBLE_DEVICES | 可选，如4,5,6,7，留空使用所有卡 | 4,5,6,7 | 留空 |
| 代理 | HTTP_PROXY/HTTPS_PROXY | 可选，如http://127.0.0.1:58232 | http://127.0.0.1:58232 | 留空 |
| 代理使用时机 | PROXY_USAGE | 见下方步骤说明 | 数据集下载时使用 | 仅数据集处理使用 |
| 复用容器 | REUSE_CONTAINER | 用户说"复用"时为true，不说就是不复用，为false | 复用现有容器 | false |

**代理使用时机（步骤）：**

| 步骤 | 默认 | 说明 |
|------|------|------|
| 环境准备 | NO | 拉取镜像不使用代理 |
| 环境检查 | NO | 检查依赖不使用代理 |
| 数据集处理 | YES | 下载数据集使用代理 |
| 模型下载 | NO | 下载模型权重不使用代理 |
| 执行训练 | NO | 训练时不使用代理 |

示例：
- 算法: GRPO
- 模型名称: Qwen2.5-0.5B-Instruct
- 数据集名称: gsm8k
- 训练后端: FSDP
- 推理后端: vllm
- 工作目录: /home/agenta
- 数据集目录: /home/datasets
- 模型权重目录: /home/model_weights
- GPU设备: 4,5,6,7
- 代理: http://127.0.0.1:58232
- 代理使用时机: 数据集下载时使用
- 复用容器

请输入您的训练配置：

用户: 
- 算法: GRPO
- 模型名称: Qwen2.5-0.5B-Instruct
- 数据集名称: gsm8k
- 训练后端: FSDP
- 推理后端: vllm
- 工作目录: /home/agenta
- 数据集目录: /home/datasets
- 模型权重目录: /home/model_weights
- GPU设备: 4,5,6,7
- 代理: http://127.0.0.1:58232
- 代理使用时机: 数据集下载时使用
- 复用容器

助手: 开始执行训练流程...
[env-preparation] ✓
[env-check] ✓
[prepare-data] ✓ (使用代理)
[model-download] ✓
[训练执行中] GPU=4,5,6,7
日志: /home/agenta/logs/Qwen2.5-0.5B-Instruct_GRPO_gsm8k_20260416_100000.log
```

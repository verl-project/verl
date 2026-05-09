# verl skills

用OpenCode Skills 在昇腾 NPU 上拉起 VERL 训练。

- VERL 官网：https://github.com/verl-project/verl
- VERL 文档：https://verl.readthedocs.io

## 目录

- [Skills 概览](#skills-概览)
- [安装配置](#安装配置)
- [快速开始](#快速开始)
- [使用教程](#使用教程)
- [路径映射](#路径映射)
- [注意事项](#注意事项)
- [算法支持情况](#算法支持情况)
- [常见问题](#常见问题)

## Skills 概览

| Skill | 功能 | 说明 |
|-------|------|------|
| **verl-training** | 主 Skill | 编排整个训练流程，一次性收集配置后自动执行 |
| **env-preparation** | 环境准备 | 拉取镜像、创建启动脚本、启动容器 |
| **env-check** | 环境检查 | 检查容器状态、依赖工具的版本信息 |
| **image-list** | 镜像查询 | 查询 quay.io 上可用的 VERL 镜像 |
| **model-download** | 模型下载 | 从 ModelScope 或 HuggingFace 下载模型权重 |
| **prepare-data** | 数据集处理 | 下载并将数据集处理为 parquet 格式 |

## 安装配置

```bash
# 创建工作目录
mkdir -p /home/agenta
cd /home/agenta

# 拉取代码并链接
git clone https://github.com/verl-project/verl.git
ln -s verl/.opencode .opencode

# 安装 opencode
curl -fsSL https://opencode.ai/install | bash
source ~/.bashrc

# 启动 opencode
opencode
```

## 快速开始

使用主 Skill 一键拉起训练：

```bash
# 启动 opencode 后，告诉 AI 您要使用verl做训练， ai会给出可选参数以及配置示例，想AI提供配置示例，即可开始训练流程：
开发者：您好，我想使用verl做训练

agent：下面是需要收集的参数配置，以及配置示例：XXXX

开发者：下面是具体的配置：
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
```

详细配置参数见 [使用教程](#使用教程) 部分。

## 使用教程

使用 `verl-training` 主 Skill，一次性收集配置后自动执行完整训练流程。

**需要收集的参数（共 11 项）：**

| 参数 | 说明 | 示例 |
|-----|------|------|
| 算法 | GRPO/DAPO/PPO/CISPO/RLOO/One_Step_Off_Policy | GRPO |
| 模型名称 | 模型名称 | Qwen2.5-0.5B-Instruct |
| 数据集名称 | 数据集名称 | gsm8k |
| 训练后端 | FSDP/MindSpeed_LLM/Megatron/FSDP2 | FSDP |
| 推理后端 | vllm/sglang | vllm |
| 工作目录 | 宿主机工作目录 | /home/agenta |
| 数据集目录 | 宿主机数据集目录 | /home/datasets |
| 模型权重目录 | 宿主机模型权重目录 | /home/model_weights |
| GPU设备 | 可选，如 4,5,6,7 | 4,5,6,7 |
| 代理 | 可选，如 http://127.0.0.1:58232 | （留空） |
| 代理使用时机 | 见下方代理使用说明 | 仅数据集处理使用 |
| 复用容器 | 是否复用容器 | 复用容器verl-xxx |

**代理使用时机：**

| 步骤 | 是否使用代理 | 说明 |
|------|-------------|------|
| 环境准备 | 否 | 拉取镜像不使用代理 |
| 环境检查 | 否 | 检查依赖不使用代理 |
| 数据集处理 | 是 | 下载数据集使用代理 |
| 模型下载 | 否 | 下载模型权重不使用代理 |
| 执行训练 | 否 | 训练时不使用代理 |

**配置示例：**

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
- 代理使用时机: 数据集下载时使用
```

## 路径映射

| 类型 | 宿主机变量 | 容器内目录 |
|------|-----------|-----------|
| 工作目录 | VERL_WORK_DIR | /workspace |
| 数据集目录 | VERL_DATA_DIR | /datasets |
| 模型目录 | VERL_MODEL_DIR | /models |

> 说明：这些变量在调用时由用户指定具体路径，默认值为表格中的示例路径。

## 注意事项

1. **代理配置**：支持分任务阶段配置代理，请根据自己网络情况决定是否需要设置代理
2. **硬件要求**：Atlas 200T A2 Box16、Atlas 900 A2 PODc、Atlas 800T A3
3. **容器命名**：默认名为 `verl`，可自定义
4. **GPU 配置**：通过 `ASCEND_RT_VISIBLE_DEVICES` 环境变量指定，如 `4,5,6,7`
5. **容器内执行**：训练在 Docker 容器内执行

## 算法支持情况

目前仅支持**单机训练**。

详细支持情况（支持的模型、算法、硬件平台等）见：[VERL 官方文档 - 算法支持现状](https://verl.readthedocs.io/en/latest/ascend_tutorial/quick_start/ascend_quick_start.html#id11)

## 常见问题

**Q: 容器启动失败？**
A: 检查 Docker 是否运行，检查镜像是否拉取成功。

**Q: 数据集下载失败？**
A: 检查网络连接，如需代理确认代理配置正确（只在数据集处理步骤使用代理）。

**Q: 训练过程中报错？**
A: 查看日志排查问题确保硬件平台和模型配置匹配。

**Q: 如何查看训练日志？**
A: 训练日志默认保存在工作目录下的 `logs` 目录。

---

更多问题可查看 VERL 官方文档：https://verl.readthedocs.io

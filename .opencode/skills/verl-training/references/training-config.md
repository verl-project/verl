# 训练配置映射表

基于文档 https://verl.readthedocs.io/en/latest/ascend_tutorial/quick_start/ascend_quick_start.html

## RL类算法
### GRPO

| 模型 | 训练脚本路径 | actor.strategy | rollout.name | 默认数据集 | hardware | 默认GPU |
|------|---------|--------------|-------------|-----------|----------|---------|
| Qwen2.5-0.5B-Instruct | /workspace/.opencode/skills/verl-training/scripts/run_*.sh | fsdp | vllm-ascend | gsm8k | Atlas 200T A2 Box16 | 8 |
| Qwen2.5-7B-instruct | /verl/examples/grpo_trainer/run_qwen2_5_7b_grpo_npu.sh | fsdp | vllm-ascend | gsm8k | Atlas 200T A2 Box16 | 16 |
| Qwen2.5-32B-instruct | /verl/examples/grpo_trainer/run_qwen2_5_32b_grpo_npu.sh | fsdp | vllm-ascend | gsm8k | Atlas 200T A2 Box16 | 16 |
| Qwen2.5-VL-3B-instruct | /verl/examples/grpo_trainer/run_qwen2_5_vl_3b_npu.sh | fsdp | vllm-ascend | geo3k | Atlas 200T A2 Box16 | 16 |
| Qwen2.5-VL-7B-instruct | /verl/examples/grpo_trainer/run_qwen2_5_vl_7b_npu.sh | fsdp | vllm-ascend | geo3k | Atlas 200T A2 Box16 | 16 |
| Qwen2.5-VL-32B-instruct | /verl/examples/grpo_trainer/run_qwen2_5_vl_32b_npu.sh | fsdp | vllm-ascend | geo3k | Atlas 200T A2 Box16 | 16 |
| Qwen3-4B | /verl/examples/grpo_trainer/run_qwen3_4b_grpo_vllm_1k_npu.sh | fsdp | vllm-ascend | gsm8k | Atlas 800T A3 | 8 |
| Qwen3-8B | /verl/examples/grpo_trainer/run_qwen3-8b_npu.sh | fsdp | vllm-ascend | dapo-math-17k | Atlas 200T A2 Box16 | 8 |
| Qwen3-8B | /verl/examples/grpo_trainer/run_qwen3_8b_grpo_sglang_32k_spmd_npu.sh | fsdp | sglang | - | Atlas 200T A2 Box16 | 8 |
| Qwen3-32B | /verl/examples/grpo_trainer/run_qwen3-32b_npu.sh | fsdp | vllm-ascend | gsm8k | Atlas 200T A2 Box16 | 16 |
| Qwen3-30B-A3B | /verl/examples/grpo_trainer/run_qwen3moe-30b_sglang_mindspeedllm_npu.sh | mindspeed | sglang | - | Atlas 800T A3 | 8 |
| Qwen3-32B | /verl/examples/grpo_trainer/run_qwen3-32b_sglang_mindspeedllm_npu.sh | mindspeed | sglang | - | Atlas 800T A3 | 8 |
| DeepSeekv3-671B | /verl/examples/grpo_trainer/run_deepseek671b_math_megatron_80gb.sh | megatron | vllm-ascend | - | Atlas 200T A2 Box16 | 16 |

### DAPO

| 模型 | 训练脚本路径 | actor.strategy | rollout.name | 默认数据集 | hardware | 默认GPU |
|------|---------|--------------|-------------|-----------|----------|---------|
| Qwen2.5-7B-instruct | /verl/recipe/dapo/run_dapo_qwen2.5_7b_npu.sh | fsdp | vllm-ascend | dapo-math-17k | Atlas 200T A2 Box16 | 16 |
| Qwen2.5-32B | /verl/recipe/dapo/run_dapo_qwen2.5_32b_npu.sh | fsdp | vllm-ascend | dapo-math-17k | Atlas 200T A2 Box16 | 16 |
| Qwen3-8B-base | /verl/recipe/dapo/run_dapo_qwen3_8b_base_npu.sh | fsdp | vllm-ascend | dapo-math-17k | Atlas 200T A2 Box16 | 8 |
| Qwen3-14B-base | /verl/recipe/dapo/run_dapo_qwen3_14b_base_npu.sh | fsdp | vllm-ascend | dapo-math-17k | Atlas 200T A2 Box16 | 8 |
| Qwen3-30B-A3B-base | /verl/recipe/dapo/run_dapo_qwen3_moe_30b_base_fsdp_npu.sh | fsdp | vllm-ascend | dapo-math-17k | Atlas 200T A2 Box16 | 8 |
| Qwen3-30B-A3B-base | /verl/recipe/dapo/run_dapo_qwen3_30b_fsdp_6k_npu.sh | megatron | vllm-ascend | - | Atlas 200T A2 Box16 | 16 |

### PPO

| 模型 | 训练脚本路径路径 | actor.strategy | rollout.name | 默认数据集 | hardware | 默认GPU |
|------|---------|--------------|-------------|-----------|----------|---------|
| Qwen3-8B | /verl/examples/ppo_trainer/run_qwen3-8b_npu.sh | fsdp | vllm-ascend | dapo-math-17k | Atlas 900 A2 PODc | 8 |

### One_Step_Off_Policy

| 模型 | 训练脚本路径路径 | actor.strategy | rollout.name | 默认数据集 | hardware | 默认GPU |
|------|---------|--------------|-------------|-----------|----------|---------|
| Qwen3-8B | /verl/verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_2_6.sh | fsdp2 | vllm | gsm8k | Atlas 800T A3 | 8 |

### CISPO

| 模型 | 训练脚本路径 | actor.strategy | rollout.name | 默认数据集 | hardware | 默认GPU |
|------|---------|--------------|-------------|-----------|----------|---------|
| Qwen2.5-0.5B | /verl/examples/cispo_trainer/run_cispo_qwen2_5_0_5b_gsm8k.sh | fsdp | vllm | gsm8k | - | 8 |

### RLOO

| 模型 | 训练脚本路径 | actor.strategy | rollout.name | 默认数据集 | hardware | 默认GPU |
|------|---------|--------------|-------------|-----------|----------|---------|
| Qwen2.5-7B | /verl/examples/rloo_trainer/run_qwen2-7b.sh | fsdp | vllm | gsm8k | - | 8 |

### SFT-PEFT

| 模型 | 训练脚本路径 | actor.strategy | 默认数据集 | hardware | 默认GPU |
|------|---------|----------------|-----------|----------|---------|
| Qwen3-8B | /verl/examples/sft/gsm8k/run_qwen3_8b_sft_peft_sp2_npu.sh | fsdp | gsm8k | Atlas 900 A2 PODc | 8 |

### ReTool-SFT

| 模型 | 训练脚本路径 | 说明 |
|------|---------|------|
| Qwen2.7B-instruct | /verl/recipe/retool/retool.py | 使用recipe脚本 |

## 算法说明

| 算法 | adv_estimator | 说明 |
|------|--------------|------|
| GRPO | grpo | Group Relative Policy Optimization |
| DAPO | grpo | Decoupled Clip Loss Dynamic Sampling |
| PPO | gae | Proximal Policy Optimization |
| One_Step_Off_Policy | grpo | 离策略单步RL |
| SFT-PEFT | - | 监督微调 |

## 策略说明

| 策略 | 说明 |
|------|------|
| fsdp | Fully Sharded Data Parallel |
| fsdp2 | FSDP2 (新版本) |
| megatron | Megatron-LM后端 |
| mindspeed | MindSpeed-LLM后端 |

## 硬件平台

| 平台 | GPU数 |
|------|-------|
| Atlas 200T A2 Box16 | 16 |
| Atlas 900 A2 PODc | 8 |
| Atlas 800T A3 | 8 |

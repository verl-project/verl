# 模型下载配置

## 模型下载链接

### Qwen系列

| 模型名 | 模型标识 | ModelScope ID | HuggingFace ID | 说明 |
|--------|---------|---------------|----------------|------|
| Qwen2.5-0.5B-Instruct | qwen2.5-0.5b | qwen/Qwen2.5-0.5B-Instruct | Qwen/Qwen2.5-0.5B-Instruct | 0.5B指令模型 |
| Qwen2.5-1.5B-Instruct | qwen2.5-1.5b | qwen/Qwen2.5-1.5B-Instruct | Qwen/Qwen2.5-1.5B-Instruct | 1.5B指令模型 |
| Qwen2.5-7B-Instruct | qwen2.5-7b | qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | 7B指令模型 |
| Qwen2.5-32B-Instruct | qwen2.5-32b | qwen/Qwen2.5-32B-Instruct | Qwen/Qwen2.5-32B-Instruct | 32B指令模型 |
| Qwen2.5-VL-3B-Instruct | qwen2.5-vl-3b | qwen/Qwen2.5-VL-3B-Instruct | Qwen/Qwen2.5-VL-3B-Instruct | 3B视觉模型 |
| Qwen2.5-VL-7B-Instruct | qwen2.5-vl-7b | qwen/Qwen2.5-VL-7B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct | 7B视觉模型 |
| Qwen2.5-VL-32B-Instruct | qwen2.5-vl-32b | qwen/Qwen2.5-VL-32B-Instruct | Qwen/Qwen2.5-VL-32B-Instruct | 32B视觉模型 |
| Qwen3-4B | qwen3-4b | qwen/Qwen3-4B | Qwen/Qwen3-4B | 4B基础模型 |
| Qwen3-8B | qwen3-8b | qwen/Qwen3-8B | Qwen/Qwen3-8B | 8B基础模型 |
| Qwen3-14B | qwen3-14b | qwen/Qwen3-14B | Qwen/Qwen3-14B | 14B基础模型 |
| Qwen3-32B | qwen3-32b | qwen/Qwen3-32B | Qwen/Qwen3-32B | 32B基础模型 |
| Qwen3-30B-A3B | qwen3-30b-a3b | qwen/Qwen3-30B-A3B | Qwen/Qwen3-30B-A3B | 30B MoE模型 |
| Qwen3-8B-Base | qwen3-8b-base | qwen/Qwen3-8B | Qwen/Qwen3-8B | 8B基础模型 |
| Qwen3-14B-Base | qwen3-14b-base | qwen/Qwen3-14B | Qwen/Qwen3-14B | 14B基础模型 |
| Qwen3-30B-A3B-Base | qwen3-30b-a3b-base | qwen/Qwen3-30B-A3B | Qwen/Qwen3-30B-A3B | 30B MoE基础模型 |

### DeepSeek系列

| 模型名 | 模型标识 | ModelScope ID | HuggingFace ID | 说明 |
|--------|---------|---------------|----------------|------|
| DeepSeek-V3 | deepseek-v3-671b | deepseek-ai/DeepSeek-V3 | deepseek-ai/DeepSeek-V3 | 671B MoE模型 |

### 视觉语言模型

| 模型名 | 模型标识 | ModelScope ID | HuggingFace ID | 说明 |
|--------|---------|---------------|----------------|------|
| Qwen2.5-VL-3B-Instruct | qwen2.5-vl-3b | qwen/Qwen2.5-VL-3B-Instruct | Qwen/Qwen2.5-VL-3B-Instruct | 3B视觉模型 |
| Qwen2.5-VL-7B-Instruct | qwen2.5-vl-7b | qwen/Qwen2.5-VL-7B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct | 7B视觉模型 |
| Qwen2.5-VL-32B-Instruct | qwen2.5-vl-32b | qwen/Qwen2.5-VL-32B-Instruct | Qwen/Qwen2.5-VL-32B-Instruct | 32B视觉模型 |
| Qwen2.5-VL-3B | qwen2.5-vl-3b | qwen/Qwen2.5-VL-3B | Qwen/Qwen2.5-VL-3B | 3B VLM |
| Qwen2.5-VL-7B | qwen2.5-vl-7b | qwen/Qwen2.5-VL-7B | Qwen/Qwen2.5-VL-7B | 7B VLM |
| Qwen2.5-VL-32B | qwen2.5-vl-32b | qwen/Qwen2.5-VL-32B | Qwen/Qwen2.5-VL-32B | 32B VLM |

## 下载目录

推荐下载到以下目录:
- 默认: `/verl/models`
- 也可指定: `/home/model_weights/`

## 使用示例

```bash
# 使用ModelScope下载
modelscope download --model_id qwen/Qwen2.5-7B-Instruct --local_dir /verl/models

# 使用HuggingFace下载
huggingface-cli download --local_dir /verl/models Qwen/Qwen2.5-7B-Instruct
```
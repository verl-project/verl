# merge_verl_lora.py
import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.distributed as dist

# 设置离线环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 路径配置
base_model_path = "/home/hjw/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/0000000000000000000000000000000000000000"
lora_path = "/home/hjw/CoT-Data-verl/outputs/Qwen2.5-0.5B-gsm8k-sft/global_step_116"
save_path = "/home/hjw/CoT-Data-verl/outputs/Qwen2.5-0.5B-gsm8k-sft/checkpoint-last"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

print("Loading LoRA config...")
# 读取 verl 的 LoRA 配置
with open(os.path.join(lora_path, "lora_train_meta.json"), "r") as f:
    lora_meta = json.load(f)

print(f"LoRA config: {lora_meta}")

# 从 lora_train_meta.json 提取参数构建 LoraConfig
# 注意：需要根据实际 json 内容调整字段名
lora_config = LoraConfig(
    r=lora_meta.get("lora_rank", 8),
    lora_alpha=lora_meta.get("lora_alpha", 16),
    target_modules=lora_meta.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    lora_dropout=lora_meta.get("lora_dropout", 0.0),
    bias="none",
    task_type="CAUSAL_LM"
)

print("Applying LoRA config to model...")
model = get_peft_model(model, lora_config)

print("Loading LoRA weights...")
# 加载 FSDP/DeepSpeed 格式的权重
lora_state_dict = torch.load(
    os.path.join(lora_path, "model_world_size_1_rank_0.pt"), 
    map_location="cpu"
)

# 加载权重到模型
model.load_state_dict(lora_state_dict, strict=False)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print(f"Saving to {save_path}...")
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Done!")
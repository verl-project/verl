#!/usr/bin/env python3
"""
合并 verl LoRA 权重到 Qwen2.5 基础模型
适用于 Qwen2.5 0.5B 及其他尺寸
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import shutil
import argparse

# 路径配置
base_model_path = "/home/hjw/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/0000000000000000000000000000000000000000"
tokenizer_path = "/home/hjw/CoT-Data-verl/outputs/Qwen2.5-0.5B-gsm8k-sft/global_step_116"
lora_path = "/home/hjw/CoT-Data-verl/outputs/Qwen2.5-0.5B-gsm8k-sft/global_step_116"
save_path = "/home/hjw/CoT-Data-verl/outputs/Qwen2.5-0.5B-gsm8k-sft/checkpoint-last"

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--base", default=base_model_path, help="基础模型路径")
parser.add_argument("--lora", default=lora_path, help="LoRA 权重路径")
parser.add_argument("--tokenizer", default=tokenizer_path, help="Tokenizer 路径")
parser.add_argument("--output", default=save_path, help="输出路径")
parser.add_argument("--verify", action="store_true", help="合并后验证")
args = parser.parse_args()

def merge_lora_weights():
    print(f"正在加载基础模型: {args.base}")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float32,  # 可以改为 float16 或 bfloat16 节省显存
        device_map="auto",           # 自动分配设备，单卡可改为 "cuda:0" 或 "cpu"
        trust_remote_code=True
    )
    
    # 加载 Tokenizer（从 LoRA 目录，保留训练时的 special tokens）
    print(f"正在加载 Tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        padding_side="left"  # Qwen 系列通常使用 left padding
    )
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载 LoRA 适配器
    print(f"正在加载 LoRA 权重: {args.lora}")
    model = PeftModel.from_pretrained(model, args.lora)
    
    # 合并权重并卸载 LoRA 适配器
    print("正在合并 LoRA 权重到基础模型...")
    model = model.merge_and_unload()
    
    # 确保保存目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 保存合并后的模型
    print(f"正在保存合并后的模型到: {args.output}")
    model.save_pretrained(args.output, safe_serialization=True)
    
    # 保存 Tokenizer（保留 added_tokens 等训练配置）
    tokenizer.save_pretrained(args.output)
    
    # 复制其他必要的配置文件（如果 LoRA 目录中有而基础模型没有的）
    config_files = [
        "chat_template.jinja",
        "added_tokens.json", 
        "special_tokens_map.json"
    ]
    
    for filename in config_files:
        src = os.path.join(args.lora, filename)
        dst = os.path.join(args.output, filename)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"复制配置文件: {filename}")
            shutil.copy2(src, dst)
    
    print("✅ 合并完成！")
    print(f"模型已保存至: {args.output}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

def verify_model():
    """简单验证合并后的模型可以正常加载和推理"""
    print("\n正在验证合并后的模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.output, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.output, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 简单测试
    test_input = "Solve: 1+1="
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"测试输入: {test_input}")
    print(f"模型输出: {result}")
    print("✅ 验证通过")

if __name__ == "__main__":
    # 执行合并
    merge_lora_weights()
    
    # 可选：验证（需要 GPU 资源，如果显存不足可注释掉）
    if args.verify:
        verify_model()
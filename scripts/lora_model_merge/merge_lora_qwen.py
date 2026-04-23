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
default_base_model_path = "/data/pretrain_models/Qwen2.5-3B-Instruct/"
default_tokenizer_path = "/data/hjw/outputs/GSM8K/training/split_0/global_step_30"
default_lora_path = "/data/hjw/outputs/GSM8K/training/split_0/global_step_30"
default_save_path = "/data/hjw/outputs/GSM8K/training/split_0/checkpoint-last"

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--base", default=default_base_model_path, help="基础模型路径")
parser.add_argument("--lora", default=default_lora_path, help="LoRA 权重路径")
parser.add_argument("--tokenizer", default=default_tokenizer_path, help="Tokenizer 路径")
parser.add_argument("--output", default=default_save_path, help="输出路径")
parser.add_argument("--verify", action="store_true", help="合并后验证")
args = parser.parse_args()

def merge_lora_weights(base_path, lora_path, tokenizer_path, output_path):
    original_cuda_home = os.environ.get('CUDA_HOME', None)

    try:
        # 临时指向正确的 CUDA 路径，供 DeepSpeed 检查使用
        os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'

        print(f"正在加载基础模型: {base_path}")

        # 加载基础模型，使用逻辑 GPU ID 0（CUDA_VISIBLE_DEVICES 已设置）
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float32,
            device_map="cuda:0",
            trust_remote_code=True
        )
        
        # 加载 Tokenizer
        print(f"正在加载 Tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载 LoRA 适配器
        print(f"正在加载 LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        
        # 合并权重
        print("正在合并 LoRA 权重到基础模型...")
        model = model.merge_and_unload()
        
        # 确保保存目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 保存合并后的模型（触发点：transformers 会在这里 import deepspeed）
        print(f"正在保存合并后的模型到: {output_path}")
        model.save_pretrained(output_path, safe_serialization=True)
        
        # 保存 Tokenizer
        tokenizer.save_pretrained(output_path)
        
        # 复制其他配置文件
        config_files = [
            "chat_template.jinja",
            "added_tokens.json", 
            "special_tokens_map.json"
        ]
        
        for filename in config_files:
            src = os.path.join(lora_path, filename)
            dst = os.path.join(output_path, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                print(f"复制配置文件: {filename}")
                shutil.copy2(src, dst)
        
        print("✅ 合并完成！")
        print(f"模型已保存至: {output_path}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
    finally:
        # 恢复原始 CUDA_HOME
        if original_cuda_home is None:
            os.environ.pop('CUDA_HOME', None)  # 原来没有就删除
        else:
            os.environ['CUDA_HOME'] = original_cuda_home

def verify_model():
    """简单验证合并后的模型可以正常加载和推理"""
    print("\n正在验证合并后的模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.output, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.output, 
        torch_dtype=torch.float16,
        device_map="cuda:6",
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
    merge_lora_weights(args.base, args.lora, args.tokenizer, args.output)

    # 可选：验证（需要 GPU 资源，如果显存不足可注释掉）
    if args.verify:
        verify_model()
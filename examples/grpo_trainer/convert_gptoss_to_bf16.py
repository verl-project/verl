"""
Convert openai/gpt-oss-20b from mxfp4 to bf16 so verl can load it with FSDP.

This is the same step that run_gptoss_20b.sh does inline.  Run once before
starting any gpt-oss training:

    python examples/grpo_trainer/convert_gptoss_to_bf16.py

Output: ~/models/gpt-oss-20b-bf16
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

model_id = "openai/gpt-oss-20b"
output_dir = os.path.expanduser("~/models/gpt-oss-20b-bf16")

print(f"Loading {model_id} and dequantizing to bf16 ...")
quantization_config = Mxfp4Config(dequantize=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)
model.config.attn_implementation = "eager"

print(f"Saving to {output_dir} ...")
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_dir)

print("Done.")

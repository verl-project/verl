# verl image with verl v0.4.x

## Important packages version

```txt
cuda==12.4
cudnn==9.8.0
torch==2.6.0
flash_attn=2.7.4
sglang==0.4.6.post5
vllm==0.8.5.post1
nvidia-cudnn-cu12==9.8.0.87
transformer_engine==2.3
megatron.core==core_v0.12.2
# Preview
transformer_engine==2.5
megatron.core==core_r0.13.0
```

## Target

- Base image: 
    - `verlai/verl:base-verl0.4-cu124-cudnn9.8-torch2.6-fa2.7.4`
- App image:
    - `verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-te2.2`: SGLang requires vLLM in 0.4.6.post5 version, vLLM can have some package conflicts with SGLang
    - `verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-te2.2-deepep`: Built with deepep
    - `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2`
    - `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2-deepep`: Built with deepep
- Preview image:
    - `verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.13.0-te2.2-preview`
    - `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-te2.2-preview`
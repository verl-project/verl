# verl image with verl v0.5

## Important packages version

```txt
cuda==12.8
cudnn==9.8.0
torch==2.7.1
flash_attn=2.8.0    ##
sglang==0.4.8
transformer_engine==2.5
megatron.core==core_r0.13.0
nvidia-cudnn-cu12==9.8.0.87
```

## Target

- Base image:
    - `verlai/verl:base-verl0.5-preview-cu128-cudnn9.8-torch2.7.1-fa2.8.0`: We offer a base image with flash infer 0.2.6.post1 built in
- App image:
    - `verlai/verl:app-verl0.5-preview-sglang0.4.8-mcore0.13.0-preview`
- vllm temporarily not support latest version

## !!!Notice!!!

- pyext is lack of maintainace and cannot work with python 3.12, consider using replacement and deprecating this package.
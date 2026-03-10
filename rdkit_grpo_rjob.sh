#!/bin/bash

rjob delete rdkit-grpo-uspto-50k
rjob submit \
    --name=rdkit-grpo-uspto-50k \
    --gpu=8 \
    --memory=1280000 \
    --cpu=96 \
    --namespace=ailab-mineru4sh \
    --charged-group=mineru4sh_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
    --image=registry.h.pjlab.org.cn/ailab-mineru4sh/dingruiyi-vllm-verl-megatron-stable:vllm0.16_verl_with_rdkit \
    --host-network=true \
    -e DISTRIBUTED_JOB=true \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    -P 2 \
    -- bash -exc /mnt/shared-storage-user/mineru4s/dingruiyi/verl_wanjuan/train_forward_rdkit_grpo.sh
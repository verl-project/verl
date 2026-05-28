# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from verl.utils.logger import print_rank_0


def init_megatron_optim_config(
    optim_config: dict,
    use_distributed_optimizer: bool = True,
    fp16: bool = False,
    bf16: bool = True,
) -> OptimizerConfig:
    optim_args = {
        "optimizer": optim_config.optimizer,
        "lr": optim_config.lr,
        "min_lr": optim_config.min_lr,
        "clip_grad": optim_config.clip_grad,
        "weight_decay": optim_config.weight_decay,
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if fp16:
        optim_args.update(
            {
                "bf16": False,
                "fp16": True,
                "params_dtype": torch.float16,
                "initial_loss_scale": 32768,
                "min_loss_scale": 1,
                "use_precision_aware_optimizer": True,
                "store_param_remainders": False,
            }
        )
    elif bf16:
        # Match precision: keep the grad-accumulation buffer and Adam
        # moments (m, v) in bf16 so optimizer-state memory tracks the
        # model dtype. Master parameters stay fp32 (Megatron default
        # `main_params_dtype`) because TE FusedAdam currently rejects
        # bf16 master weights at init (only fp32/fp16 accepted). The
        # int16 "store_param_remainders" path (Megatron default True
        # in bf16 mode) already eliminates the fp32 master buffer in
        # favor of bf16 working + int16 remainders, achieving the same
        # ~50% master-memory reduction.
        # Requires TransformerEngine's FusedAdam (already needed by
        # the precision-aware optimizer path). Override any of these
        # via `override_optimizer_config` to opt back into fp32.
        optim_args.update(
            {
                "bf16": True,
                "params_dtype": torch.bfloat16,
                "use_precision_aware_optimizer": True,
                "main_grads_dtype": torch.bfloat16,
                "exp_avg_dtype": torch.bfloat16,
                "exp_avg_sq_dtype": torch.bfloat16,
            }
        )
    else:
        # fp32 mode: leave grad-accumulation buffer and Adam moments at
        # Megatron's default torch.float32. Do not enable the precision-aware
        # optimizer — it's only beneficial when a moment/grad dtype is below
        # fp32, and Megatron asserts the dtype fields equal fp32 whenever the
        # precision-aware optimizer is off (optimizer_config.py:258-268).
        optim_args.update(
            {
                "bf16": False,
                "fp16": False,
                "params_dtype": torch.float32,
            }
        )
    override_config = optim_config.get("override_optimizer_config", {})
    if override_config:
        for k, v in override_config.items():
            optim_args[k] = v

    print_rank_0(f"optimizer config after override: {optim_args}")

    config = OptimizerConfig(**optim_args)
    return config


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
):
    # Base optimizer.
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    lr_decay_steps = config.lr_decay_steps
    lr_warmup_steps = config.lr_warmup_steps
    if config.get("lr_decay_steps", None) is None:
        lr_decay_steps = config.total_training_steps
    wsd_decay_steps = None
    if config.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (
        config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.lr_warmup_init,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=config.total_training_steps,
        wd_incr_style=config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]

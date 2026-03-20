# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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


import argparse

import torch

from verl.workers.config import HFModelConfig, McoreOptimizerConfig, MindSpeedEngineConfig


def get_base_mcore_config_from_model_config(model_config: HFModelConfig) -> dict:
    """
    Create a base TransformerConfig with common parameters across different model architectures.

    Args:
        model_config: HuggingFace model configuration

    Returns:
        TransformerConfig with common parameters
    """

    hf_config = model_config.hf_config
    base_config = {
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),
        "norm_topk_prob": getattr(hf_config, "norm_topk_prob", False),
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "max_position_embeddings": hf_config.max_position_embeddings,
        "tie_word_embeddings": hf_config.tie_word_embeddings,
        "torch_dtype": hf_config.torch_dtype,
        "bf16": hf_config.dtype is torch.bfloat16,
        "rotary_base": int(hf_config.rope_theta),
        "num_experts": getattr(hf_config, "num_experts", None),
        "moe_router_topk": getattr(hf_config, "num_experts_per_tok", None),
        "moe_ffn_hidden_size": getattr(hf_config, "moe_intermediate_size", None),
        "padded_vocab_size": hf_config.vocab_size,
        "make_vocab_size_divisible_by": 1,
        "untie_embeddings_and_output_weights": True,
    }

    tokenizer_config = {
        "tokenizer_name_or_path": model_config.tokenizer_path,
        "tokenizer_type": "PretrainedFromHF",
    }
    base_config.update(tokenizer_config)
    return base_config


def get_base_mcore_config_from_engine_config(engine_config: MindSpeedEngineConfig) -> dict:
    """
    Create a base TransformerConfig with common parameters across different model architectures.

    Args:
        engine_config: mindspeed engine configuration

    Returns:
        TransformerConfig with common parameters
    """

    base_config = {
        "tensor_model_parallel_size": engine_config.tensor_model_parallel_size,
        "expert_model_parallel_size": engine_config.expert_model_parallel_size,
        "expert_tensor_parallel_size": engine_config.expert_tensor_parallel_size,
        "pipeline_model_parallel_size": engine_config.pipeline_model_parallel_size,
        "virtual_pipeline_model_parallel_size": engine_config.virtual_pipeline_model_parallel_size,
        "context_parallel_size": engine_config.context_parallel_size,
        "sequence_parallel": engine_config.sequence_parallel,
        "use_distributed_optimizer": engine_config.use_distributed_optimizer,
        "seed": engine_config.seed,
    }
    if engine_config.strategy == "mindspeed_llm":
        base_config.update(engine_config.llm_kwargs)
    elif engine_config.strategy == "mindspeed_mm":
        base_config.update(engine_config.mm_kwargs)
    return base_config


def get_base_mcore_config_from_optim_config(optim_config: McoreOptimizerConfig) -> dict:
    """
    Create a base TransformerConfig with common parameters across different model architectures.

    Args:
        optim_config: megatron optimizer configuration

    Returns:
        TransformerConfig with common parameters
    """

    base_config = {
        "lr": optim_config.lr,
        "lr_decay_style": optim_config.lr_decay_style,
        "min_lr": optim_config.min_lr,
        "weight_decay": optim_config.weight_decay,
        "lr_warmup_fraction": optim_config.lr_warmup_steps_ratio,
        "clip_grad": optim_config.clip_grad,
        "adam_beta1": optim_config.betas[0],
        "adam_beta2": optim_config.betas[1],
    }

    base_config.update(optim_config.override_optimizer_config)
    return base_config


def set_global_config(config):
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_global_variables

    args = parse_args(ignore_unknown_args=True)
    for key, value in config.items():
        setattr(args, key, value)

    validate_args(args)
    try:
        set_global_variables(args)
    except AssertionError:
        print("megatron args already set")


def add_mcore_arguments(all_config: dict) -> dict:
    mcore_config_dict = {}
    mcore_config_list = []
    for key, value in all_config.items():
        if value is None:
            continue
        mcore_config_dict[key] = value
        if isinstance(value, bool):
            mcore_config_list.append(f"--{key.replace('_', '-')}")

    from megatron.training.arguments import add_megatron_arguments

    parser = argparse.ArgumentParser(description="Megatron-LM Arguments", allow_abbrev=False)
    parser = add_megatron_arguments(parser)
    args, _ = parser.parse_known_args(mcore_config_list)
    return {**vars(args), **mcore_config_dict}


def apply_patch(model_config, engine_config, optimizer_config):
    model_config = get_base_mcore_config_from_model_config(model_config)
    optimizer_config = get_base_mcore_config_from_optim_config(optimizer_config)
    engine_config = get_base_mcore_config_from_engine_config(engine_config)
    all_config = {**model_config, **optimizer_config, **engine_config}
    mcore_config = add_mcore_arguments(all_config)
    from mindspeed_llm.tasks.megatron_adaptor_v2 import repatch

    repatch(mcore_config)
    set_global_config(mcore_config)


def gpt_model_provider(pre_process=True, post_process=True):
    """
    Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    # Experimental loading arguments from configs
    config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts, args.moe_grouped_gemm, qk_layernorm=args.qk_layernorm
            )
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                args.num_experts, args.moe_grouped_gemm, qk_layernorm=args.qk_layernorm
            )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    return model

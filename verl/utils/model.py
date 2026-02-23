# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Utilities to create common models from huggingface
"""

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tensordict.tensorclass import NonTensorData
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    MistralForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from verl.models.registry import ModelRegistry
from verl.utils.import_utils import is_trl_available


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def update_model_config(module_config, override_config_kwargs):
    """Update the module config with the override_config_kwargs.
    Args:
        module_config: The module config from Huggingface Transformers.
        override_config_kwargs: The kwargs to override the module config.
    """
    for key, val in override_config_kwargs.items():
        if isinstance(val, dict):
            update_model_config(getattr(module_config, key), val)
        else:
            setattr(module_config, key, val)


def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> dict:
    if override_config_kwargs is None:
        override_config_kwargs = {}
    assert isinstance(override_config_kwargs, dict), (
        f"override_config_kwargs must be a dict, got {type(override_config_kwargs)}"
    )
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    update_model_config(module_config, override_config_kwargs)

    return module_config


def get_generation_config(
    model: str,
    trust_remote_code: bool = False,
) -> Optional[GenerationConfig]:
    try:
        return GenerationConfig.from_pretrained(model)
    except OSError:  # Not found
        try:
            config = get_huggingface_actor_config(
                model,
                trust_remote_code=trust_remote_code,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def create_huggingface_actor(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    """

    Args:
        model_name:
        override_config_kwargs:

    Returns:

    """
    if override_config_kwargs is None:
        override_config_kwargs = {}
    if automodel_kwargs is None:
        automodel_kwargs = {}
    assert isinstance(override_config_kwargs, dict), (
        f"override_config_kwargs must be a dict, got {type(override_config_kwargs)}"
    )
    module_config = get_huggingface_actor_config(
        model_name, override_config_kwargs, trust_remote_code=automodel_kwargs.get("trust_remote_code", False)
    )
    module: nn.Module = AutoModelForCausalLM.from_config(module_config, **automodel_kwargs)
    return module


def create_huggingface_critic(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    """

    Args:
        model_name:
        override_config_kwargs:

    Returns:

    """
    critic_module: nn.Module = create_huggingface_actor(
        model_name, override_config_kwargs=override_config_kwargs, automodel_kwargs=automodel_kwargs
    )
    if automodel_kwargs is None:
        automodel_kwargs = {}
    torch_dtype = automodel_kwargs.get("torch_dtype", torch.float32)
    critic_module.lm_head = nn.Sequential(
        nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype), LambdaLayer(fn=squeeze)
    )
    return critic_module


def get_model_size(model: nn.Module, scale="auto"):
    n_params = sum(p.numel() for p in model.parameters())

    if scale == "auto":
        if n_params > 1e9:
            scale = "B"
        elif n_params > 1e6:
            scale = "M"
        elif n_params > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        n_params = n_params / 1e9
    elif scale == "M":
        n_params = n_params / 1e6
    elif scale == "K":
        n_params = n_params / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}")

    return n_params, scale


def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale="auto")
    if name is None:
        name = model.__class__.__name__
    print(f"{name} contains {n_params:.2f}{scale} parameters")


def create_random_mask(
    input_ids: torch.Tensor,
    max_ratio_of_valid_token: float,
    max_ratio_of_left_padding: float,
    min_ratio_of_valid_token: float = 0,
):
    """Create a random mask given input_ids. Support left padding and right padding.
    Process:
    - Sample valid token length
    - Sample left_padding length
    - Generate padding

    Args:
        input_ids:
            shape (batch_size, seq_len)

    Returns:

    """
    assert max_ratio_of_valid_token > 0 and max_ratio_of_valid_token <= 1.0
    assert max_ratio_of_left_padding >= 0 and max_ratio_of_left_padding < 1.0
    assert min_ratio_of_valid_token <= max_ratio_of_valid_token

    batch_size, sequence_length = input_ids.shape
    max_num_valid_tokens = int(sequence_length * max_ratio_of_valid_token)
    min_num_valid_tokens = max(1, int(sequence_length * min_ratio_of_valid_token))
    max_left_padding = int(sequence_length * max_ratio_of_left_padding)
    assert max_num_valid_tokens + max_left_padding <= sequence_length
    assert max_num_valid_tokens > 0 and max_ratio_of_valid_token <= sequence_length
    masks = torch.ones_like(input_ids, dtype=torch.int64)
    # TODO: we can make this faster
    for i in range(batch_size):
        num_left_padding = np.random.randint(low=0, high=max_left_padding + 1, dtype=np.int64)
        num_valid = np.random.randint(low=min_num_valid_tokens, high=max_num_valid_tokens + 1, dtype=np.int64)

        for index in range(num_left_padding):
            masks[i, index] = 0

        for index in range(num_left_padding + num_valid, sequence_length):
            masks[i, index] = 0
    return masks


def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def convert_weight_keys(state_dict: dict[str, torch.Tensor], model: PreTrainedModel):
    # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return state_dict

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    original_weights = {}
    for key, value in state_dict.items():
        for pattern, replacement in reverse_key_mapping.items():
            replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            # Early exit of the loop
            if n_replace > 0:
                break

        original_weights[key] = value

    return original_weights


def check_exclude_modules(config, key: str) -> bool:
    """
    A helper method to check if the passed module's key name matches any of the exclude modules in the adapter_config.
    Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/tuners_utils.py

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match exclude modules from
        key (`str`): A key to search any matches in config

    Returns:
        True of match object if key matches any exclude modules from config, False if no match found
    """
    if hasattr(config, "exclude_modules") and config.exclude_modules:
        if isinstance(config.exclude_modules, str):
            if re.fullmatch(config.exclude_modules, key):
                return True
        elif key in config.exclude_modules:
            return True
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):
            return True
    return False


def check_target_modules(config, key: str) -> bool:
    """
    A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.
    Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/tuners_utils.py

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        True of match object if key matches any target modules from config, False if no match found
    """
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    elif key in config.target_modules:
        # this module is specified directly in target_modules
        target_module_found = True
    else:
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        layer_indexes = getattr(config, "layers_to_transform", None)
        layers_pattern = getattr(config, "layers_pattern", None)

        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave
            # For now, empty layers_pattern means any layer pattern is ok
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes

    return target_module_found


def normalize_model_name(name, pp_rank, vpp_rank, transformer_config, layer_name="layers"):
    """
    Transform the model name in each model_chunk in each pp stage into the name in inference engine
    """
    from verl.utils.megatron_utils import get_transformer_layer_offset

    layer_offset = get_transformer_layer_offset(pp_rank, vpp_rank, transformer_config)

    if layer_name in name:  # belong to an intermediate layer
        split_name = name.split(".")
        # find the num next to split_name
        for i, name in enumerate(split_name):
            if name == layer_name:
                break
        layer_num_idx = i + 1
        # check the name
        assert len(split_name) >= layer_num_idx + 1, f"split_name = {split_name}"
        assert split_name[layer_num_idx].isdigit(), f"split_name = {split_name}"
        # increment layer_num_idx by layer_offset
        split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)
        name = ".".join(split_name)  # weight name in inference_tp_model
    return name


def normalize_pp_vpp_params(params, num_hidden_layers, layer_name="layers"):
    """
    Normalize the pp vpp params into a complete named parameters.
    This is useful when gather parameters from pp ranks and passed to a model without pp

    params: Iterable[List[Dict[str, param]]]
        params contains a list of pp, with a list of vpp named_parameters in each vpp chunk.
    output: Dict[str, param]

    """
    pp_size = len(params)
    for pp_rank in range(len(params)):
        vpp_size = len(params[pp_rank])
        for vpp_rank in range(vpp_size):
            for name, param in params[pp_rank][vpp_rank].items():
                normalized_name = normalize_model_name(
                    name, pp_rank, vpp_rank, pp_size, vpp_size, num_hidden_layers, layer_name=layer_name
                )
                yield normalized_name, param


def get_parallel_model_from_config(
    config, megatron_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    from megatron.core import ModelParallelConfig

    assert isinstance(megatron_config, ModelParallelConfig)
    model_class = _get_parallel_model_architecture_from_config(config, value)

    model = model_class(
        config,
        megatron_config,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
    )
    return model


def _get_parallel_model_architecture_from_config(config: PretrainedConfig, value=False) -> type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch, value)
        print("after load model cls")
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. Supported architectures: "
        f"{ModelRegistry.get_supported_archs()}"
    )


def _load_hf_model(config, model_config, is_value_model):
    """Helper function containing the loading hf model logic"""
    from accelerate import init_empty_weights
    from megatron.core import parallel_state as mpu

    from verl.models.mcore.saver import _megatron_calc_global_rank

    assert hasattr(model_config, "architectures"), "architectures cannot be empty when load weight!"
    architectures = getattr(model_config, "architectures", [])

    # get auto class
    auto_cls = get_hf_auto_model_class(model_config)

    if config.model.path.startswith("hdfs:"):
        from verl.utils.fs import copy_to_local

        print(f"start download from {config.model.path}")
        local_model_path = copy_to_local(src=config.model.path, use_shm=config.model.get("use_shm", False))
        print("finish download")
    else:
        local_model_path = config.model.path
        print(f"load from local dir {local_model_path}")

    src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=0, cp_rank=mpu.get_context_parallel_rank())
    cpu_init_weights = lambda: torch.device("cpu")
    init_context = init_empty_weights if torch.distributed.get_rank() != src_rank else cpu_init_weights
    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: to find a better way to load mistral7b-rm lm_head
        if "mistral7b-rm" in config.model.path:
            model = MistralForSequenceClassification.from_pretrained(
                local_model_path,
                torch_dtype="auto",
                # device_map="auto",  # disable auto device_map, the HF weight is only loaded to CPU in src_rank
                # low_cpu_mem_usage=True
            )  # use score head instead of lm_head
            state_dict = model.state_dict()
            state_dict["lm_head.weight"] = state_dict["score.weight"]
            state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"][
                :32000
            ]  # workaround, 32001 -> 32000
            is_value_model = True
        else:
            model = auto_cls.from_pretrained(
                local_model_path,
                torch_dtype="auto",
                # device_map="auto", # disable auto device_map, the HF weight is only loaded to CPU in src_rank
                # low_cpu_mem_usage=True
            )
            state_dict = model.state_dict()

    return architectures, model, state_dict, is_value_model


def get_hf_model_path(config):
    if config.model.path.startswith("hdfs:"):
        from verl.utils.fs import copy_to_local

        local_model_path = copy_to_local(src=config.model.path, use_shm=config.model.get("use_shm", False))
    else:
        local_model_path = config.model.path
    return local_model_path


def load_megatron_model_weights(config, model_config, parallel_model, params_dtype, is_value_model=False):
    """Load weights for verl customized model."""
    architectures, model, state_dict, is_value_model = _load_hf_model(config, model_config, is_value_model)

    from verl.models.weight_loader_registry import get_weight_loader

    print(f"before weight loader: architectures = {architectures}...")
    for arch in architectures:
        print(f"call weight loader arch = {arch}, model config = {model.config}")
        weight_loader = get_weight_loader(arch)
        weight_loader(
            state_dict=state_dict,
            wrapped_models=parallel_model,
            config=model.config,
            params_dtype=params_dtype,
            is_value_model=is_value_model,
            tie_word_embeddings=model_config.tie_word_embeddings,
        )
    return model.config


def load_megatron_gptmodel_weights(config, model_config, parallel_model, params_dtype, is_value_model=False):
    """Load weights for mcore GPT model."""
    _, model, state_dict, is_value_model = _load_hf_model(config, model_config, is_value_model)

    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    load_state_dict_to_megatron_gptmodel(
        state_dict=state_dict,
        wrapped_models=parallel_model,
        config=model.config,
        params_dtype=params_dtype,
        is_value_model=is_value_model,
    )
    del state_dict, model


# pad input_ids_rmpad, cu_seqlens and max_seqlen_in_batch to be divisible by tp
def pad_packed_inputs(unpad_tokens: torch.Tensor, cu_seqlens, max_seqlen_in_batch, size):
    """pad the tokens such that the total length is a multiple of size.
    This function is useful when applying sequence parallel and context parallel

    Args:
        unpad_tokens: (total_nnz, ...). Tokens after removing padding
        cu_seqlens: (total_nnz + 1,)
        max_seqlen_in_batch: int

    Returns:

    """
    F = nn.functional

    total_nnz = unpad_tokens.shape[0]

    pad_size = 0 if total_nnz % size == 0 else size - total_nnz % size

    # we assume adding a new data in the batch with seqlen pad_size
    if pad_size > 0:
        if unpad_tokens.ndim == 1:
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        elif unpad_tokens.ndim == 2:
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        else:
            raise NotImplementedError(f"Padding dim {unpad_tokens.ndim()} is not supported")

        cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_size + cu_seqlens[-1])
        max_seqlen_in_batch = max(max_seqlen_in_batch, pad_size)

    return unpad_tokens, cu_seqlens, max_seqlen_in_batch


def load_mcore_dist_weights(parallel_model, dist_weight_path, is_value_model=False, prefix=""):
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.serialization import StrictHandling

    from verl.utils.megatron_utils import unwrap_model

    # strict = StrictHandling.IGNORE_ALL if is_value_model else StrictHandling.ASSUME_OK_UNEXPECTED
    strict = StrictHandling.ASSUME_OK_UNEXPECTED
    for model in parallel_model:
        ssd = unwrap_model(model).sharded_state_dict(prefix=prefix)
        if is_value_model:
            for k in list(ssd.keys()):
                if "output_layer" in k:
                    ssd.pop(k)
        dist_checkpointing.load(ssd, dist_weight_path, strict=strict)

    return


def get_parallel_gptmodel_from_config(
    tfconfig, hf_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    use_te = True
    assert tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
    transformer_layer_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=use_te)
    rope_scaling_args = {}
    if hf_config.rope_scaling is not None:
        assert hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
        rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling["factor"]
    parallel_model = GPTModel(
        config=tfconfig,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=hf_config.vocab_size,
        max_sequence_length=hf_config.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        position_embedding_type="rope",
        rotary_base=hf_config.rope_theta,
        **rope_scaling_args,
    )
    # # for layer in parallel_model.decoder.layers:
    # layer.self_attention.core_attention.flash_attention.softmax_scale = None
    if post_process and value:
        from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

        parallel_model.output_layer = LinearForLastLayer(
            input_size=tfconfig.hidden_size, output_size=1, config=tfconfig
        )
    return parallel_model


def patch_valuehead_model(model) -> None:
    from types import MethodType

    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead

    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def can_generate(self):
        return False

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    model._keys_to_ignore_on_save = ignore_modules
    model.tie_weights = MethodType(tie_weights, model)
    model.get_input_embeddings = MethodType(get_input_embeddings, model)
    model.get_output_embeddings = MethodType(get_output_embeddings, model)
    model.can_generate = MethodType(can_generate, model)
    model._no_split_modules = getattr(model.pretrained_model, "_no_split_modules", [])


def load_valuehead_model(local_path, torch_dtype, model_config, trust_remote_code):
    from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=model_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
        )
        return model
    except BaseException as e:
        if not is_trl_available():
            raise RuntimeError(
                f"model({local_path}) is not a value head model, please install trl to make it valid"
            ) from e

    assert is_trl_available()

    from trl import AutoModelForCausalLMWithValueHead

    if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
        module_class = AutoModelForVision2Seq
    else:
        module_class = AutoModelForCausalLM
    ori_model = module_class.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(ori_model)
    patch_valuehead_model(model)
    return model


_architecture_to_auto_class = {
    "ForCausalLM": AutoModelForCausalLM,
    "ForVision2Seq": AutoModelForVision2Seq,
    "ForTokenClassification": AutoModelForTokenClassification,
    "ForSequenceClassification": AutoModelForSequenceClassification,
}


def get_hf_auto_model_class(hf_config):
    has_remote_code = hasattr(hf_config, "auto_map") and any(
        hf_config.architectures[0] in val for val in hf_config.auto_map.values()
    )
    if has_remote_code:
        auto_class = next(k for k, v in hf_config.auto_map.items() if hf_config.architectures[0] in v)
        match auto_class:
            case "AutoModelForVision2Seq":
                actor_module_class = AutoModelForVision2Seq
            case "AutoModelForCausalLM":
                actor_module_class = AutoModelForCausalLM
            case "AutoModelForImageTextToText":
                actor_module_class = AutoModelForImageTextToText
            case _:
                actor_module_class = AutoModel
    else:
        actor_module_class = AutoModel
        # For VLM models, we use type to check instead of architecture
        if type(hf_config) in AutoModelForImageTextToText._model_mapping.keys():
            actor_module_class = AutoModelForImageTextToText
        else:
            for key, cls in _architecture_to_auto_class.items():
                if key in hf_config.architectures[0]:
                    actor_module_class = cls
                    break

    return actor_module_class


def extract_multi_modal_inputs(
    batch_data: list[dict[str, torch.Tensor]],
    indices: Optional[list[int]] = None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """
    Extract and process multi-modal inputs from a batch.

    Args:
        batch_data (list[dict[str, torch.Tensor]]): The batch containing potential multi-modal inputs
        indices (Optional[list[int]]): If provided, only extract inputs at these indices

    Returns:
        dict[str, torch.Tensor | list[torch.Tensor]]: Processed multi-modal inputs ready for model consumption

    """
    multi_modal_inputs = {}
    multi_modal_inputs_collected = {}
    has_image_bound = False

    selected_batch_data = batch_data
    if indices is not None:
        selected_batch_data = [batch_data[i] for i in indices if i < len(batch_data)]

    for inputs in selected_batch_data:
        inputs = inputs.data if isinstance(inputs, NonTensorData) else inputs
        # Mixed pure text and multi-modal dataset.
        if inputs is None:
            continue
        if "image_bound" in inputs:
            has_image_bound = True
        for key, value in inputs.items():
            if value is not None:
                if key not in multi_modal_inputs_collected:
                    multi_modal_inputs_collected[key] = []
                multi_modal_inputs_collected[key].append(value)

    for key, values in multi_modal_inputs_collected.items():
        if has_image_bound:  # minicpm-o logic
            multi_modal_inputs[key] = values
        else:
            multi_modal_inputs[key] = torch.cat(values, dim=0)

    return multi_modal_inputs


def get_lora_rank_from_adapter(adapter_path: str | os.PathLike) -> int:
    """
    Extract LoRA rank from adapter configuration file.

    Args:
        adapter_path: Path to LoRA adapter directory

    Returns:
        LoRA rank value from adapter_config.json

    Raises:
        FileNotFoundError: If adapter path or config file doesn't exist
        ValueError: If config file is invalid or missing rank
    """
    adapter_path = os.path.abspath(os.path.expanduser(str(adapter_path)))

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            if "r" not in config:
                raise ValueError(f"LoRA rank 'r' not found in {config_path}")
            return int(config["r"])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}") from e
    except (KeyError, ValueError) as e:
        raise ValueError(f"Cannot parse LoRA rank from {config_path}: {e}") from e


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None

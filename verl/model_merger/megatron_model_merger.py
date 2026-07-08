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

import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, ContextManager

import numpy as np
import torch
import torch.distributed as dist

try:
    # NPU patch
    import mindspeed.megatron_adaptor  # noqa: F401
except ImportError:
    pass

from accelerate import init_empty_weights
from megatron.core import mpu
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    PretrainedConfig,
)

from verl.models.mcore import hf_to_mcore_config
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.distributed import set_numa_affinity
from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing
from verl.utils.megatron_utils import get_model
from verl.utils.tokenizer import hf_processor, hf_tokenizer

from .base_model_merger import BaseModelMerger, ModelMergerConfig


@contextmanager
def noop_context() -> Any:
    yield


def get_dynamic_pipeline_shards(layer_num: int, pp_size: int) -> list[int]:
    """Calculate the pipeline sharding configuration for Megatron-LM.

    Args:
        layer_num: Total number of layers in the model.
        pp_size: Number of pipeline parallel ranks.

    Returns:
        layer number of each pp rank. Make the sharding of the pipeline as uniform as possible.
    """
    if layer_num < pp_size:
        raise ValueError(f"layer_num {layer_num} must be greater than pp_size {pp_size}.")

    if pp_size < 1:
        raise ValueError(f"pp_size must be at least 1, got {pp_size}.")
    if pp_size == 1:
        return [layer_num]

    if pp_size == 2:
        return [
            layer_num // 2,
            layer_num - layer_num // 2,
        ]

    middle_size = pp_size - 2
    shards_strategy = []
    for middle_layer_num in range(layer_num):
        first_last_layer_num = layer_num - middle_layer_num * middle_size
        first_layer_num = first_last_layer_num // 2
        last_layer_num = first_last_layer_num - first_last_layer_num // 2
        if 0 < first_layer_num <= middle_layer_num and 0 < last_layer_num <= middle_layer_num:
            shards_strategy.append(
                (
                    [first_layer_num] + [middle_layer_num] * middle_size + [last_layer_num],
                    abs(first_layer_num - middle_layer_num),
                )
            )

    # sort by diff of layer_num, to make it as uniform as possible
    res = sorted(shards_strategy, key=lambda x: x[1])[0][0]
    assert sum(res) == layer_num, f"sum(res)={sum(res)} != layer_num={layer_num}, pp_size={pp_size}"
    return res


class MegatronModelMerger(BaseModelMerger):
    """
    Model merger for Megatron-LM distributed checkpoints.

    This class handles the conversion of Megatron-LM distributed checkpoints into HuggingFace format.
    Megatron-LM uses tensor parallelism, pipeline parallelism, and data parallelism to distribute
    large language models across multiple GPUs. This merger reconstructs the full model by
    loading distributed checkpoints and applying the necessary transformations.

    Key features:
    - Support for tensor parallel, pipeline parallel, and data parallel configurations
    - Automatic parameter name mapping from Megatron to HuggingFace conventions
    - Handling of QKV and gate-up tensor splitting/merging
    - Support for tied word embeddings and value models
    - Integration with Megatron's distributed checkpointing system

    The merger handles various model architectures and configurations:
    - Standard transformer models (GPT-style)
    - Models with tied word embeddings
    - Value models for reinforcement learning
    - Multi-layer attention (MLA) architectures
    - Mixture of Experts (MoE) models

    Args:
        config (ModelMergerConfig): Configuration object with Megatron-specific settings
            including tie_word_embedding and is_value_model flags.

    Example:
        To merge Megatron checkpoints:
        ```python
        config = ModelMergerConfig(
            operation="merge",
            backend="megatron",
            local_dir="path/to/megatron/checkpoints",
            target_dir="path/to/output",
            tie_word_embedding=True
        )
        merger = MegatronModelMerger(config)
        merger.merge_and_save()
        ```
    """

    def __init__(self, config: ModelMergerConfig):
        super().__init__(config)
        # Currently we use only 1 rank to merge the dist_ckpt, we will move to multi-process save shortly afterwards
        if "WORLD_SIZE" not in os.environ:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

        set_numa_affinity()
        torch.distributed.init_process_group(get_nccl_backend())

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        local_rank = os.environ.get("LOCAL_RANK", 0)
        get_torch_device().set_device(f"{get_device_name()}:{local_rank}")

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=self.world_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(0)
        self.hf_config = AutoConfig.from_pretrained(
            self.config.hf_model_config_path, trust_remote_code=self.config.trust_remote_code
        )
        print(self.hf_config, flush=True)

        self.params_mapping = {
            # megatron core gpt model name, huggingface model name
            # NOTICE: It's a little bit tricky, when 2 keys have the same prefix, we need to make sure the
            # longer key within the containing relationship is processed first.
            "embedding.word_embeddings": "model.embed_tokens",
            # input layer norm for dpskv3
            "input_layernorm.weight": "input_layernorm.weight",
            "input_layernorm.bias": "input_layernorm.bias",
            # attn
            "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
            "self_attention.linear_qkv.layer_norm_bias": "input_layernorm.bias",
            "self_attention.linear_qkv": "self_attn.qkv_proj",
            "self_attention.q_layernorm": "self_attn.q_norm",
            "self_attention.k_layernorm": "self_attn.k_norm",
            "self_attention.linear_proj": "self_attn.o_proj",
            # mla
            "self_attention.linear_q_proj": "self_attn.q_proj",
            "self_attention.linear_q_down_proj": "self_attn.q_a_proj",
            "self_attention.linear_q_up_proj.layer_norm_weight": "self_attn.q_a_layernorm.weight",
            "self_attention.linear_q_up_proj": "self_attn.q_b_proj",
            "self_attention.linear_kv_down_proj": "self_attn.kv_a_proj_with_mqa",
            "self_attention.linear_kv_up_proj.layer_norm_weight": "self_attn.kv_a_layernorm.weight",
            "self_attention.linear_kv_up_proj": "self_attn.kv_b_proj",
            # mlp
            "pre_mlp_layernorm": "post_attention_layernorm",
            "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
            "mlp.linear_fc1.layer_norm_bias": "post_attention_layernorm.bias",
            "mlp.linear_fc1": "mlp.gate_up_proj",
            "mlp.linear_fc2": "mlp.down_proj",
            # moe
            "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
            "mlp.router": "mlp.gate",
            "mlp.shared_experts.linear_fc1": "mlp.shared_experts.gate_up_proj",
            "mlp.shared_experts.linear_fc2": "mlp.shared_experts.down_proj",
            "linear_fc1": "gate_up_proj",
            "linear_fc2": "down_proj",
            # output
            "final_layernorm": "norm",
            "output_layer": "lm_head",
        }

        if "Qwen2MoeForCausalLM" in self.hf_config.architectures:
            self.params_mapping["mlp.shared_experts.linear_fc1"] = "mlp.shared_expert.gate_up_proj"
            self.params_mapping["mlp.shared_experts.linear_fc2"] = "mlp.shared_expert.down_proj"
            self.params_mapping["mlp.shared_experts.gate_weight"] = "mlp.shared_expert_gate.weight"

    def _load_state_dicts(self, model_ckpt_path: str) -> dict[str, Any]:
        """_summary_
        Use Megatron dist_checkpointing to load the model state dicts from the checkpoint directory.

        Args:
            model_ckpt_path (str): Path to the model checkpoint directory.

        Returns:
            State dict containing the model parameters.
        """

        # init hf config
        self.pipeline_shards = get_dynamic_pipeline_shards(self.hf_config.num_hidden_layers, self.world_size)
        print(f"Pipeline shards: {self.pipeline_shards}, total layers: {sum(self.pipeline_shards)}")

        tf_config = hf_to_mcore_config(
            self.hf_config,
            torch.bfloat16,
            num_layers_in_first_pipeline_stage=self.pipeline_shards[0] if len(self.pipeline_shards) > 1 else None,
            num_layers_in_last_pipeline_stage=self.pipeline_shards[-1] if len(self.pipeline_shards) > 2 else None,
        )
        tf_config.use_cpu_initialization = self.config.use_cpu_initialization
        tie_word_embeddings = getattr(self.hf_config, "tie_word_embeddings", False)

        # init megatron model
        def megatron_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(
                tf_config,
                self.hf_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=tie_word_embeddings,
                value=False,
            )
            return parallel_model

        context: Callable[..., ContextManager] = (
            init_empty_weights if self.config.use_cpu_initialization else noop_context
        )
        with context():
            whole_model = get_model(
                model_provider_func=megatron_model_provider,
                model_type=ModelType.encoder_or_decoder,
                wrap_with_ddp=False,
                transformer_config=tf_config,
            )

        if self.config.use_cpu_initialization:
            # convert meta device to empty tensor so it can use `copy_` function
            whole_model[0].module = whole_model[0].module.to_empty(device="cpu")

        # load state dicts
        sharded_state_dict = {}
        for vpp_rank, model in enumerate(whole_model):
            key = f"model{vpp_rank}" if len(whole_model) > 1 else "model"
            mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            sharded_state_dict[key] = model.sharded_state_dict()
        model_state_dict = load_dist_checkpointing(sharded_state_dict, model_ckpt_path)
        model_state_dict_list = []
        for vpp_rank, model in enumerate(whole_model):
            key = f"model{vpp_rank}" if len(whole_model) > 1 else "model"
            mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            model_state_dict_list.append(model_state_dict[key])

        return model_state_dict_list

    def _check_megatron_state_key(self, key: str) -> bool:
        """
        Checks if the key is a valid Megatron state key.

        Now the model merger only supports keys that start with "decoder/embedding/output_layer" in TransformerLayer.
        Shall not use key starts with "model."
        """
        if key.startswith("model."):
            raise ValueError(
                f"Invalid key {key} in Megatron state_dict. Expected keys to start with "
                f"'decoder/embedding/output_layer' in TransformerLayer."
            )

        skip_checking_keys = ["embedding.word_embeddings", "output_layer"]
        for skip_key in skip_checking_keys:
            if skip_key in key:
                print(f"skip checking key {key}")
                return

        # Exclude extra state keys
        if not key.startswith("decoder"):
            raise ValueError(
                f"Invalid key {key} in Megatron state_dict. Expected keys to start with 'decoder' in TransformerLayer."
            )

    def _split_tensors(
        self, key: str, tensor: torch.Tensor, config: PretrainedConfig, is_value_model: bool = False
    ) -> list[torch.Tensor]:
        """
        Splits a tensor into multiple tensors based on the name.
        This is used to handle qkv and gate_up tensors.
        """
        if "linear_fc1.weight" in key:
            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            gate, up = tensor.chunk(2)
            gate_lst.append(gate)
            up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            return [gate, up]
        elif "self_attention.linear_qkv." in key and "layer_norm" not in key:
            # if the tensor is qkv, for each param on tp, split into q, k, v
            # concat q, k, v separately.
            q_lst, k_lst, v_lst = [], [], []
            assert config.num_attention_heads % config.num_key_value_heads == 0
            num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
            assert tensor.shape[0] % (num_q_per_kv + 2) == 0, (
                f"Tensor shape {tensor.shape} is not divisible by {num_q_per_kv + 2}"
            )
            kv_size = tensor.shape[0] // (num_q_per_kv + 2)
            split_size = [kv_size * num_q_per_kv, kv_size, kv_size]

            num_query_groups_per_partition = config.num_key_value_heads
            for chunk in tensor.chunk(num_query_groups_per_partition):
                split_size = [
                    kv_size * num_q_per_kv // num_query_groups_per_partition,
                    kv_size // num_query_groups_per_partition,
                    kv_size // num_query_groups_per_partition,
                ]
                q, k, v = chunk.split(split_size)
                q_lst.append(q)
                k_lst.append(k)
                v_lst.append(v)

            return [torch.cat(q_lst, dim=0), torch.cat(k_lst, dim=0), torch.cat(v_lst, dim=0)]
        else:
            return [tensor]

    def _merge_state_dicts(self, model_state_dict_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        state_dict = {}
        layers_cum = 0
        if self.world_size > 1:
            pipeline_cumsum = np.cumsum(self.pipeline_shards)
            layers_cum = 0 if self.rank == 0 else pipeline_cumsum[self.rank - 1]

        print(f"{layers_cum=}")
        for model_state_dict in model_state_dict_list:
            layers_handled = 0
            keys = model_state_dict.keys()
            for key in keys:
                if "extra_state" in key:
                    continue
                if self.config.tie_word_embedding and ("output_layer" in key):
                    print("skip lm_head and reward_head loading because of tie_word_embeddings")
                    continue

                self._check_megatron_state_key(key)
                hf_name = self._replace_name(key, self.params_mapping)
                assert hf_name is not None, f"Failed to convert layer name [{key}] from megatron to huggingface."
                if "model.layers." in hf_name:
                    local_layer_no = int(hf_name.split(".")[2])
                    layers_handled = max(local_layer_no, layers_handled)
                    global_layer_no = local_layer_no + layers_cum
                    new_key_list = hf_name.split(".")
                    new_key_list[2] = str(global_layer_no)
                    hf_name = ".".join(new_key_list)
                else:
                    warnings.warn(f"hf_name {hf_name} will not be fixed with layer number", stacklevel=2)

                if "mlp.experts." in hf_name and ".weight" in hf_name:
                    name_prefix, expert_id = hf_name.split(".weight")
                    for proj in ["gate_up", "down"]:
                        if f"{proj}_proj" in hf_name:
                            hf_name = hf_name.replace(
                                f"mlp.experts.{proj}_proj.weight{expert_id}",
                                f"mlp.experts.{expert_id}.{proj}_proj.weight",
                            )

                tensor = model_state_dict[key]
                split_tensor = self._split_tensors(
                    key, tensor, self.hf_config, is_value_model=self.config.is_value_model
                )

                if len(split_tensor) == 1:
                    state_dict[hf_name] = split_tensor[0]
                elif len(split_tensor) == 3:
                    # split qkv
                    for n, d in zip(["q", "k", "v"], split_tensor, strict=True):
                        state_dict[hf_name.replace("qkv", n)] = d
                elif len(split_tensor) == 2:
                    # split gate up
                    state_dict[hf_name.replace("gate_up", "gate")] = split_tensor[0]
                    state_dict[hf_name.replace("gate_up", "up")] = split_tensor[1]
                shape_info = (
                    split_tensor.shape if isinstance(split_tensor, torch.Tensor) else [t.shape for t in split_tensor]
                )
                print(f"converted {key} to {hf_name} with shape {shape_info}")

            layers_cum += layers_handled + 1  # zero based

        return state_dict

    def save_hf_model_and_tokenizer(self, merged_state_dict):
        if self.world_size == 1:
            return super().save_hf_model_and_tokenizer(merged_state_dict)

        from safetensors.torch import save_file

        layer_num = self.hf_config.num_hidden_layers

        # FIXME: make configurable
        saves_per_layer = 1 if layer_num < 30 else 2
        saves_total = saves_per_layer * layer_num
        saves_indexes = {}

        # calculate the layer start index and key chunks
        layer_this_rank = self.pipeline_shards[self.rank]
        pipeline_cumsum = np.cumsum(self.pipeline_shards)
        layer_start = 0 if self.rank == 0 else pipeline_cumsum[self.rank - 1]
        keys = list(merged_state_dict.keys())
        keys_chunk = np.array_split(np.array(keys), layer_this_rank * saves_per_layer)
        numel = 0

        assert len(keys_chunk) == layer_this_rank * saves_per_layer, (
            f"Expected {len(keys_chunk)} chunks, but got {layer_this_rank * saves_per_layer} for rank {self.rank}."
        )

        # save to model shards manually
        target_dir = Path(self.config.target_dir)
        for i, keys in enumerate(keys_chunk):
            sd_to_save = {k: merged_state_dict[k] for k in keys}
            numel += sum([sd_to_save[i].numel() for i in sd_to_save])
            save_idx = layer_start * saves_per_layer + i
            save_path = target_dir / f"model-{save_idx + 1:05d}-of-{saves_total:05d}.safetensors"

            save_file(sd_to_save, save_path)
            for k in keys:
                saves_indexes[k] = str(save_path.name)

        tensor = torch.tensor([numel]).to(get_device_name())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        numel = tensor.cpu().item()

        all_save_indexes = [{} for _ in range(self.world_size)]
        dist.all_gather_object(all_save_indexes, saves_indexes)
        saves_indexes = {k: v for i in all_save_indexes for k, v in i.items()}
        if self.rank == 0:
            with open(target_dir / "model.safetensors.index.json", "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "total_size": numel,
                        },
                        "weight_map": saves_indexes,
                    },
                    f,
                    indent=4,
                )
            print(f"model saved to {target_dir} with {numel=}")

            self.model_config.save_pretrained(self.config.target_dir)

            processor = hf_processor(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
            tokenizer = hf_tokenizer(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
            if processor is not None:
                print(f"Saving processor to {self.config.target_dir}")
                processor.save_pretrained(self.config.target_dir)
            if tokenizer is not None:
                print(f"Saving tokenizer to {self.config.target_dir}")
                tokenizer.save_pretrained(self.config.target_dir)

    def merge_and_save(self):
        from verl.utils.megatron_utils import get_dist_checkpoint_path

        model_ckpt_path = get_dist_checkpoint_path(self.config.local_dir)

        model_state_dict = self._load_state_dicts(model_ckpt_path)
        merged_state_dict = self._merge_state_dicts(model_state_dict)
        del model_state_dict

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self._validate_state_dict(merged_state_dict)
        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            if self.config.hf_upload:
                self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

    def _validate_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """
        Compares the merged Megatron state_dict against a reference safetensors model.
        Applies necessary name mappings from Megatron to Hugging Face conventions using _replace_name.
        """
        ref_state_dict = load_file(Path(self.config.test_hf_dir) / "model.safetensors")

        for name, loaded_weight in state_dict.items():
            # name = self._replace_name(original_name, self.params_mapping)
            if not name or name.endswith(".bias") and name not in ref_state_dict:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "lm_head.weight" in name:
                if self.config.is_value_model or self.config.tie_word_embedding:
                    continue
            if name not in ref_state_dict:
                raise RuntimeError(f"key: {name} not exist in state_dict")
            param = ref_state_dict[name]
            assert loaded_weight.dtype == param.dtype
            torch.testing.assert_close(loaded_weight.to("cpu"), param, atol=1e-2, rtol=5e-2)

    def _replace_name(self, megatron_name: str, name_mapping: dict[str, str]) -> str:
        for m_name, v_name in name_mapping.items():
            if m_name not in megatron_name:
                continue

            megatron_name = megatron_name.replace("decoder", "model")
            param_name = megatron_name.replace(m_name, v_name)

            return param_name

        return None  # Return None if no mapping found

    def cleanup(self):
        torch.distributed.destroy_process_group()

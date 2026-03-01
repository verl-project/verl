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
This script is used to merge huggingface model and test verl checkpoints from FSDP and Megatron backends.

To merge FSDP checkpoints:
```sh
python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

To merge Megatron checkpoints:
```sh
python scripts/legacy_model_merger.py merge \
    --backend megatron \
    --tie-word-embedding \
    --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

For more details, please refer to documentation:
https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model
"""

import argparse
import os
import re
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch.distributed._tensor import Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PretrainedConfig,
)

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from tqdm import tqdm

from verl.utils import hf_processor, hf_tokenizer


@dataclass
class ModelMergerConfig:
    operation: str  # 'merge' or 'test'
    backend: str
    local_dir: str
    hf_model_config_path: str
    target_dir: Optional[str] = "tmp"
    hf_upload_path: Optional[str] = None
    private: bool = False
    test_hf_dir: Optional[str] = None
    tie_word_embedding: bool = False
    is_value_model: bool = False
    hf_model_path: Optional[str] = None
    hf_upload: bool = field(init=False)

    def __post_init__(self):
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)
        if self.operation == "test":
            self.target_dir = None
            self.hf_upload_path = None
            self.private = False


class BaseModelMerger(ABC):
    def __init__(self, config: ModelMergerConfig):
        self.config = config
        self.hf_model_config_path = config.hf_model_config_path

        if config.hf_model_path:
            print(
                "Warning: --hf_model_path is deprecated and will be removed in a future version. Currently verl will save huggingface model configuration files into checkpoint directories. Therefore, there is no need to provide --hf_model_path. "
            )
            self.hf_model_config_path = config.hf_model_path

        # Auto-detect huggingface subdirectory if it exists
        huggingface_subdir = os.path.join(self.hf_model_config_path, "huggingface")
        if os.path.isdir(huggingface_subdir):
            self.hf_model_config_path = huggingface_subdir

        self.model_config = AutoConfig.from_pretrained(self.hf_model_config_path)

    def get_transformers_auto_model_class(self):
        # Handle case where architectures might be None or empty
        if self.model_config.architectures is None or len(self.model_config.architectures) == 0:
            # Try to infer from model_type if architectures is missing
            model_type = getattr(self.model_config, 'model_type', '').lower()
            if 'vision' in model_type or 'vl' in model_type:
                return AutoModelForVision2Seq
            elif 'causal' in model_type or 'gpt' in model_type or 'llama' in model_type or 'qwen' in model_type:
                return AutoModelForCausalLM
            else:
                raise NotImplementedError(
                    f"Cannot determine model class: architectures is None and model_type '{model_type}' is not recognized"
                )
        
        architecture = self.model_config.architectures[0]
        if "ForTokenClassification" in architecture:
            return AutoModelForTokenClassification
        elif "ForCausalLM" in architecture:
            return AutoModelForCausalLM
        elif "ForConditionalGeneration" in architecture:
            return AutoModelForVision2Seq

        raise NotImplementedError(f"Unknown architecture {self.model_config.architectures}")

    def patch_model_generation_config(self, model):
        """
        The generation_config created from model config may be different to the pretrained model,
        this may lead to error when generating: https://github.com/volcengine/verl/issues/1246

        This function patch the generation_config created from model config to the pretrained model.
        """
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(self.hf_model_config_path)
            except OSError:
                print(
                    f"Warning: Generation config file not found in {self.hf_model_config_path}, using a generation config created from the model config."
                )
        return model

    def save_lora_adapter(self, state_dict: dict[str, torch.Tensor]):
        """
        Save lora adapter to safetensors.

        Returns:
            lora_path: str, the path to the lora adapter. None if no lora adapter found.

        Note:
            This function change the 'state_dict' in place.
        """
        lora_params_names = [name for name in state_dict.keys() if "lora_" in name]

        if len(lora_params_names) == 0:
            return None

        import json
        from typing import OrderedDict

        import peft
        from safetensors.torch import save_file

        lora_params = OrderedDict()
        target_modules = set()
        lora_key = None

        for name in lora_params_names:
            lora_key = name.replace(".default.weight", ".weight")
            target_modules.add(lora_key.split(".")[-3])
            lora_params[lora_key] = state_dict.pop(name)

        lora_rank = min(lora_params[lora_key].shape[0], lora_params[lora_key].shape[1])
        peft_dict = {
            "r": lora_rank,
            "lora_alpha": 0,  # lora_alpha is not set. An error should be raised to inform the user to set it manually.
            "target_modules": list(target_modules),
        }
        peft_config = peft.LoraConfig(**peft_dict).to_dict()
        peft_config["task_type"] = peft_config["task_type"].value if peft_config["task_type"] else None
        peft_config["peft_type"] = peft_config["peft_type"].value if peft_config["peft_type"] else None
        peft_config["target_modules"] = list(peft_config["target_modules"])

        lora_path = os.path.join(self.config.target_dir, "lora_adapter")
        os.makedirs(lora_path, exist_ok=True)
        with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(peft_config, f, ensure_ascii=False, indent=4)
        save_file(lora_params, os.path.join(lora_path, "adapter_model.safetensors"))

        for name in list(state_dict.keys()):
            key = (
                name.replace("base_model.model.", "")
                .replace(".base_layer.weight", ".weight")
                .replace(".base_layer.bias", ".bias")
            )
            state_dict[key] = state_dict.pop(name)

        return lora_path

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()
        with init_empty_weights():
            model = auto_model_class.from_config(self.model_config, torch_dtype=torch.bfloat16)
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        lora_path = self.save_lora_adapter(state_dict)
        if lora_path:
            print(f"Saving lora adapter to {lora_path}")

        print(f"Saving model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        del state_dict
        del model

        processor = hf_processor(self.hf_model_config_path)
        try:
            tokenizer = hf_tokenizer(self.hf_model_config_path)
        except Exception as e:
            warnings.warn(f"Failed to create tokenizer: {e}. This may affect tokenizer saving", stacklevel=1)
            tokenizer = None
        if processor is not None:
            print(f"Saving processor to {self.config.target_dir}")
            processor.save_pretrained(self.config.target_dir)
        if tokenizer is not None:
            print(f"Saving tokenizer to {self.config.target_dir}")
            tokenizer.save_pretrained(self.config.target_dir)

    def upload_to_huggingface(self):
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=self.config.hf_upload_path, private=self.config.private, exist_ok=True)
        api.upload_folder(folder_path=self.config.target_dir, repo_id=self.config.hf_upload_path, repo_type="model")

    @abstractmethod
    def merge_and_save(self):
        raise NotImplementedError("Subclasses should implement this method")


class FSDPModelMerger(BaseModelMerger):
    def _get_world_size(self) -> int:
        """Extracts the FSDP world_size from checkpoint filenames (e.g., 'model_world_size_8_rank_0.pt')."""
        for filename in os.listdir(self.config.local_dir):
            match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
            if match:
                return int(match.group(1))
        raise FileNotFoundError(
            f"Could not determine world size. No file matching 'model_world_size_(\\d+)_rank_0.pt' found in {self.config.local_dir}"
        )

    def _load_rank_zero_state_dict(self, world_size: int) -> dict:
        return torch.load(
            Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_0.pt",
            map_location="cpu",
            weights_only=False,
        )

    def _extract_device_mesh_info(self, state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
        """
        Retrieves sharding information (device_mesh, mesh_dim_names) from a DTensor in the state_dict.
        If no DTensor is found, infers a simple FSDP mesh based on world_size.
        """
        pivot_key = sorted(list(state_dict.keys()))[0]
        weight = state_dict[pivot_key]

        if isinstance(weight, DTensor):
            # get sharding info
            device_mesh = weight.device_mesh
            mesh = device_mesh.mesh
            mesh_dim_names = device_mesh.mesh_dim_names
        else:
            # for non-DTensor
            mesh = np.array([world_size], dtype=np.int64)
            mesh_dim_names = ("fsdp",)

        return mesh, mesh_dim_names

    def _calculate_shard_configuration(
        self, mesh: np.ndarray, mesh_dim_names: tuple[str, ...]
    ) -> tuple[int, tuple[int, ...]]:
        """Calculates the total number of shards and the shape of the device mesh."""
        assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

        if "tp" in mesh_dim_names:
            # TODO: "tp" is not supported yet due to the above assert
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        else:
            total_shards = mesh.shape[-1]
            mesh_shape = (mesh.shape[-1],)

        return total_shards, mesh_shape

    def _merge_by_placement(self, tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        """Merges a list of tensors based on their DTensor placement"""
        if placement.is_replicate():
            return tensors[0]
        elif placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        elif placement.is_shard():
            return torch.cat(tensors, dim=placement.dim).contiguous()

        raise NotImplementedError(f"Unsupported placement: {placement}")

    def _load_and_merge_state_dicts(
        self, world_size: int, total_shards: int, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, model_state_dict_lst: list):
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return state_dict

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                future.result()

        # Merge state dicts from all shards
        state_dict = {}
        param_placements: dict[str, list] = {}

        for key in set(model_state_dict_lst[0].keys()):
            state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                # add tensor shard in order of rank to state_dict[key]
                tensor = model_state_shard.pop(key)
                if isinstance(tensor, DTensor):
                    state_dict[key].append(tensor._local_tensor.bfloat16())

                    placements = tuple(tensor.placements)
                    # replicated placement at dp dimension can be discarded
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(tensor.bfloat16())

        del model_state_dict_lst

        # Merge tensors
        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                print(f"No need to merge key {key}")
                continue
            if key in param_placements:
                # merge shards
                placements: tuple[Shard] = param_placements[key]
                if len(mesh_shape) == 1:
                    # 1-D list, FSDP without TP
                    assert len(placements) == 1
                    shards = state_dict[key]
                    state_dict[key] = self._merge_by_placement(shards, placements[0])
                else:
                    # 2-D list, FSDP + TP
                    raise NotImplementedError("FSDP + TP is not supported yet")
            else:
                state_dict[key] = torch.cat(state_dict[key], dim=0)

        return state_dict

    def merge_and_save(self):
        world_size = self._get_world_size()
        rank_zero_state_dict = self._load_rank_zero_state_dict(world_size)

        mesh, mesh_dim_names = self._extract_device_mesh_info(rank_zero_state_dict, world_size)
        print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

        total_shards, mesh_shape = self._calculate_shard_configuration(mesh, mesh_dim_names)
        print(f"Processing model shards with {total_shards} {mesh_shape} in total")

        merged_state_dict = self._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self._test_state_dict(merged_state_dict)
        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            if self.config.hf_upload:
                self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

    def _test_state_dict(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()

        hf_model = auto_model_class.from_pretrained(self.config.test_hf_dir, torch_dtype=torch.bfloat16)
        hf_state_dict = hf_model.state_dict()
        del hf_model

        hf_model_keys = set(hf_state_dict.keys())
        collected_keys = set(state_dict.keys())

        missing_keys = hf_model_keys - collected_keys
        assert len(missing_keys) == 0, f"Missing keys in collected state dict: {list(sorted(missing_keys))}"

        extra_keys = collected_keys - hf_model_keys
        assert len(extra_keys) == 0, f"Extra keys in collected state dict: {list(sorted(extra_keys))}"

        for key in hf_model_keys:
            hf_shape = hf_state_dict[key].shape
            collected_shape = state_dict[key].shape
            assert hf_shape == collected_shape, (
                f"Shape mismatch for key '{key}': original {hf_shape} vs collected {collected_shape}"
            )

            hf_dtype = hf_state_dict[key].dtype
            collected_dtype = state_dict[key].dtype
            assert hf_dtype == collected_dtype, (
                f"Dtype mismatch for key '{key}': original {hf_dtype} vs collected {collected_dtype}"
            )

            torch.testing.assert_close(hf_state_dict[key], state_dict[key], atol=1e-6, rtol=1e-6)

        print("FSDP checks passed: The merged state_dict matches the hf model saved by FSDPCheckpointManager.")


class MegatronModelMerger(BaseModelMerger):
    def __init__(self, config: ModelMergerConfig):
        from verl.utils.megatron_utils import get_hf_config_and_tokenizer_checkpoint_path

        config.hf_model_config_path = get_hf_config_and_tokenizer_checkpoint_path(config.local_dir)
        super().__init__(config)

        self.params_mapping = {
            # megatron core gpt model name, huggingface model name
            # NOTICE: It's a little bit tricky, when 2 keys have the same prefix, we need to make sure the longer key within the containing relationship is processed first.
            "embedding.word_embeddings": "model.embed_tokens",
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

    def _get_tp_pp_rank_from_sharded_dir(self, sharded_dir: str) -> tuple[int, int]:
        tp_rank = pp_rank = None
        rank_list = sharded_dir.split("_")[2:]
        if re.match(r"mp_rank_(\d\d)_(\d\d\d)", sharded_dir):
            tp_rank = int(rank_list[0])
            pp_rank = int(rank_list[1])
        elif re.match(r"mp_rank_(\d\d)", sharded_dir):
            tp_rank = int(rank_list[0])
            pp_rank = 0

        assert tp_rank is not None and pp_rank is not None, f"Invalid sharded dir {sharded_dir}"

        return tp_rank, pp_rank

    def _check_megatron_checkpoint_path(self, model_path: str) -> tuple[list[str], int, int]:
        """
        Validates the Megatron checkpoint structure (presence of 'model.pt' in sharded directories).
        Determines TP and PP sizes from directory names.
        """
        tp_size = 0
        pp_size = 0
        sharded_dirs = sorted(os.listdir(model_path))
        for sharded_dir in sharded_dirs:
            assert "model.pt" in os.listdir(Path(model_path) / sharded_dir), f"model.pt not found in {sharded_dir}"
            tp_rank, pp_rank = self._get_tp_pp_rank_from_sharded_dir(sharded_dir)
            tp_size = max(tp_size, tp_rank + 1)
            pp_size = max(pp_size, pp_rank + 1)
        return sharded_dirs, tp_size, pp_size

    def _merge_across_tp(
        self,
        key: str,
        tp_data: list[torch.Tensor],
        config: PretrainedConfig,
        tp_size: int,
        is_value_model: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        if "linear_fc1.weight" in key:
            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            for infer_param in tp_data:
                gate, up = infer_param.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            return [gate, up]
        elif "self_attention.linear_qkv." in key and "layer_norm" not in key:
            # if the tensor is qkv, for each param on tp, split into q, k, v
            # concat q, k, v separately.
            q_lst = []
            k_lst = []
            v_lst = []
            assert config.num_attention_heads % config.num_key_value_heads == 0
            num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
            assert tp_data[0].shape[0] % (num_q_per_kv + 2) == 0
            kv_size_per_tp = tp_data[0].shape[0] // (num_q_per_kv + 2)
            split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]

            for infer_param in tp_data:
                num_query_groups_per_partition = config.num_key_value_heads // tp_size
                for chunk in infer_param.chunk(num_query_groups_per_partition):
                    split_size = [
                        kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                    ]
                    q, k, v = chunk.split(split_size)
                    q_lst.append(q)
                    k_lst.append(k)
                    v_lst.append(v)

            q = torch.cat(q_lst, dim=0)
            k = torch.cat(k_lst, dim=0)
            v = torch.cat(v_lst, dim=0)
            return [q, k, v]
        elif "layer_norm" in key or "layernorm" in key or "router" in key or ("output_layer" in key and is_value_model):
            return tp_data[0]
        else:
            dim = 0
            if "linear_fc2.weight" in key or "self_attention.linear_proj" in key:
                dim = 1
            return torch.cat(tp_data, dim=dim)

    def _load_state_dicts(
        self, model_ckpt_path: str, sharded_dirs: list[str], tp_size: int, pp_size: int
    ) -> list[list[dict]]:
        model_state_dict_lst = [[None for _ in range(tp_size)] for _ in range(pp_size)]

        def _process_one_megatron_shard(sharded_dir: str):
            model_file_path = Path(model_ckpt_path) / sharded_dir / "model.pt"
            state_dict = torch.load(model_file_path, map_location="cpu", weights_only=False)
            tp_rank, pp_rank = self._get_tp_pp_rank_from_sharded_dir(sharded_dir)
            model_state_dict_lst[pp_rank][tp_rank] = state_dict

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = [executor.submit(_process_one_megatron_shard, sharded_dir) for sharded_dir in sharded_dirs]
            for future in tqdm(futures, desc=f"Loading {len(sharded_dirs)} Megatron shards", total=len(sharded_dirs)):
                future.result()

        return model_state_dict_lst

    def _check_megatron_state_key(self, key: str) -> bool:
        """
        Checks if the key is a valid Megatron state key.

        Now the model merger only supports keys that start with "decoder/embedding/output_layer/vision_model" in TransformerLayer.
        Shall not use key starts with "model."
        """
        if key.startswith("model."):
            raise ValueError(
                f"Invalid key {key} in Megatron state_dict. Expected keys to start with 'decoder/embedding/output_layer/vision_model' in TransformerLayer."
            )

        skip_checking_keys = ["embedding.word_embeddings", "output_layer", "vision_model"]
        for skip_key in skip_checking_keys:
            if skip_key in key:
                print(f"skip checking key {key}")
                return

        # Exclude extra state keys
        if not key.startswith("decoder"):
            raise ValueError(
                f"Invalid key {key} in Megatron state_dict. Expected keys to start with 'decoder' in TransformerLayer."
            )

    def _merge_state_dicts(
        self, model_state_dict_lst: list[list[dict]], tp_size: int, pp_size: int
    ) -> dict[str, torch.Tensor]:
        state_dict = {}
        vpp_size = len(model_state_dict_lst[0][0])
        layers_cum = 0

        for vpp_rank in range(vpp_size):
            for pp_rank in range(pp_size):
                layers_handled = 0
                keys = model_state_dict_lst[pp_rank][0][vpp_rank].keys()
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

                    tp_data = [model_state_dict_lst[pp_rank][tp_rank][vpp_rank][key] for tp_rank in range(tp_size)]
                    merged = self._merge_across_tp(key, tp_data, self.model_config, tp_size, self.config.is_value_model)

                    if not isinstance(merged, list):
                        state_dict[hf_name] = merged
                    elif len(merged) == 3:
                        # split qkv
                        for n, d in zip(["q", "k", "v"], merged):
                            state_dict[hf_name.replace("qkv", n)] = d
                    elif len(merged) == 2:
                        # split gate up
                        state_dict[hf_name.replace("gate_up", "gate")] = merged[0]
                        state_dict[hf_name.replace("gate_up", "up")] = merged[1]
                    print(
                        f"converted {key} to {hf_name} with shape {merged.shape if isinstance(merged, torch.Tensor) else [t.shape for t in merged]}"
                    )

                layers_cum += layers_handled + 1  # zero based

        return state_dict

    def merge_and_save(self):
        from verl.utils.megatron_utils import get_model_checkpoint_path

        model_ckpt_path = get_model_checkpoint_path(self.config.local_dir)
        sharded_dirs, tp_size, pp_size = self._check_megatron_checkpoint_path(model_ckpt_path)
        print(f"sharded_dirs: {sharded_dirs}, tp_size: {tp_size}, pp_size: {pp_size}, mp_size: {len(sharded_dirs)}")

        model_state_dict_lst = self._load_state_dicts(model_ckpt_path, sharded_dirs, tp_size, pp_size)
        merged_state_dict = self._merge_state_dicts(model_state_dict_lst, tp_size, pp_size)
        del model_state_dict_lst

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self._test_state_dict(merged_state_dict)
        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            if self.config.hf_upload:
                self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

    def _test_state_dict(self, state_dict: dict[str, torch.Tensor]):
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
            if self.config.tie_word_embedding and "lm_head.weight" in name:
                continue
            if name not in ref_state_dict:
                raise RuntimeError(f"key: {name} not exist in state_dict")
            param = ref_state_dict[name]
            assert loaded_weight.dtype == param.dtype
            torch.testing.assert_close(loaded_weight, param, atol=1e-2, rtol=5e-2)

    def _replace_name(self, megatron_name: str, name_mapping: dict[str, str]) -> str:
        for m_name, v_name in name_mapping.items():
            if m_name not in megatron_name:
                continue

            megatron_name = megatron_name.replace("decoder", "model")
            param_name = megatron_name.replace(m_name, v_name)
            return param_name

        return None  # Return None if no mapping found


def main():
    parser = argparse.ArgumentParser(description="verl model merger")
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Specify 'merge' or 'test' operation.")

    base_op_parser = argparse.ArgumentParser(add_help=False)
    base_op_parser.add_argument(
        "--backend", type=str, required=True, choices=["fsdp", "megatron"], help="The backend of the model"
    )
    base_op_parser.add_argument("--local_dir", type=str, required=True, help="Path to the saved model checkpoints")
    base_op_parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="(Deprecated) Path to the original Hugging Face model for config.",
    )
    base_op_parser.add_argument(
        "--tie-word-embedding",
        action="store_true",
        help="Whether to tie word embedding weights (currently only Megatron supported)",
    )
    base_op_parser.add_argument(
        "--is-value-model",
        action="store_true",
        help="Whether the model is a value model (currently only Megatron supported)",
    )

    merge_parser = subparsers.add_parser("merge", parents=[base_op_parser], help="Merge model checkpoints and save.")
    merge_parser.add_argument(
        "--target_dir", default="tmp", type=str, help="Directory to save the merged huggingface model"
    )
    merge_parser.add_argument(
        "--hf_upload_path", default=None, type=str, help="Hugging Face repository ID to upload the model"
    )
    merge_parser.add_argument(
        "--private", action="store_true", help="Whether to upload the model to a private Hugging Face repository"
    )

    test_parser = subparsers.add_parser(
        "test", parents=[base_op_parser], help="Test merged model against a reference Hugging Face model"
    )
    test_parser.add_argument(
        "--test_hf_dir", type=str, required=True, help="Path to the reference Hugging Face model directory for testing"
    )

    args = parser.parse_args()

    common_config_args = {
        "operation": args.operation,
        "backend": args.backend,
        "tie_word_embedding": args.tie_word_embedding,
        "is_value_model": args.is_value_model,
        "local_dir": args.local_dir,
        "hf_model_path": args.hf_model_path,
        "hf_model_config_path": args.local_dir,
    }

    if args.operation == "merge":
        config = ModelMergerConfig(
            **common_config_args,
            target_dir=args.target_dir,
            hf_upload_path=args.hf_upload_path,
            private=args.private,
            test_hf_dir=None,
        )
        os.makedirs(config.target_dir, exist_ok=True)
    elif args.operation == "test":
        config = ModelMergerConfig(
            **common_config_args,
            test_hf_dir=args.test_hf_dir,
            # the following args are not used by test operation
            target_dir=None,
            hf_upload_path=None,
            private=False,
        )
    else:
        raise NotImplementedError(f"Unknown operation: {args.operation}")

    if config.backend == "fsdp":
        merger = FSDPModelMerger(config)
    elif config.backend == "megatron":
        merger = MegatronModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    merger.merge_and_save()


if __name__ == "__main__":
    main()

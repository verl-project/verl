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

import argparse
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, GenerationConfig

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.transformers_compat import copy_generation_config, drop_tied_target_keys, get_auto_model_for_vision2seq

from .output_validation import validate_hf_model_output

AutoModelForVision2Seq = get_auto_model_for_vision2seq()


def parse_args():
    parser = argparse.ArgumentParser(description="verl model merger")
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Specify 'merge' or 'test' operation.")

    base_op_parser = argparse.ArgumentParser(add_help=False)
    base_op_parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["fsdp", "megatron", "veomni"],
        help="The backend of the model",
    )
    base_op_parser.add_argument("--local_dir", type=str, default=None, help="Path to the saved model checkpoints.")
    base_op_parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to the HuggingFace model the training started from. Required by the veomni backend for models "
        "exported in their original quantized layout (DeepSeek-V4).",
    )
    base_op_parser.add_argument(
        "--tie-word-embedding",
        action="store_true",
        help="Whether to tie word embedding weights (currently only Megatron supported)",
    )
    base_op_parser.add_argument("--trust-remote-code", action="store_true", help="Whether to trust remote code")
    base_op_parser.add_argument(
        "--is-value-model",
        action="store_true",
        help="Whether the model is a value model (currently only Megatron supported)",
    )
    base_op_parser.add_argument(
        "--use_cpu_initialization",
        action="store_true",
        help="Whether to use CPU initialization for the model. This is useful for large models that cannot "
        "fit into GPU memory during initialization.",
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
    return args


@dataclass
class ModelMergerConfig:
    """Configuration for model merger operations.

    Args:
        operation (str): Operation type - 'merge' or 'test'.
        backend (str): Backend type for the model ('fsdp', 'megatron' or 'veomni').
        target_dir (Optional[str]): Directory to save the merged huggingface model. Defaults to "tmp".
        hf_upload_path (Optional[str]): Hugging Face repository ID to upload the model. Defaults to None.
        private (bool): Whether to upload the model to a private Hugging Face repository. Defaults to False.
        test_hf_dir (Optional[str]): Path to the reference Hugging Face model directory for testing. Defaults to None.
        tie_word_embedding (bool): Whether to tie word embedding weights (currently only Megatron
            supported). Defaults to False.
        trust_remote_code (bool): Whether to trust remote code. Defaults to False.
        is_value_model (bool): Whether the model is a value model (currently only Megatron
            supported). Defaults to False.
        local_dir (Optional[str]): Path to the saved model checkpoints. Defaults to None.
        hf_model_config_path (Optional[str]): Path to HuggingFace model configuration files. Defaults to None.
        hf_upload (bool): Whether to upload to HuggingFace (computed automatically). Not for initialization.
        use_cpu_initialization (bool): Whether to use CPU initialization for large models. Defaults to False.
        base_model_path (Optional[str]): Path to the HuggingFace model the training started from. Defaults to None.
    """

    operation: str  # 'merge' or 'test'
    backend: str
    target_dir: Optional[str] = "tmp"
    hf_upload_path: Optional[str] = None
    private: bool = False
    test_hf_dir: Optional[str] = None
    tie_word_embedding: bool = False
    trust_remote_code: bool = False
    is_value_model: bool = False
    local_dir: Optional[str] = None
    hf_model_config_path: Optional[str] = None
    hf_upload: bool = field(init=False)
    use_cpu_initialization: bool = False
    base_model_path: Optional[str] = None

    def __post_init__(self):
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)
        if self.operation == "test":
            self.target_dir = None
            self.hf_upload_path = None
            self.private = False


def _default_hf_model_config_path(backend: str, local_dir: str) -> str:
    """Return the default path to HuggingFace config/tokenizer artifacts.

    - FSDP: ``<local_dir>/huggingface`` (unchanged, single-tree layout).
    - Megatron (v2 layout): ``<local_dir>/model/huggingface`` — the HF tree
      is nested under ``model/`` alongside the optional ``model/dist_ckpt/``
      shards.  Checkpoints produced before the v2 refactor will therefore
      not resolve here; run ``scripts/migrate_megatron_checkpoint_layout.py``
      first.
    """
    if backend == "megatron":
        return os.path.join(local_dir, "model", "huggingface")
    return os.path.join(local_dir, "huggingface")


def generate_config_from_args(args: argparse.Namespace) -> ModelMergerConfig:
    common_config_args = {
        "operation": args.operation,
        "backend": args.backend,
        "tie_word_embedding": args.tie_word_embedding,
        "trust_remote_code": args.trust_remote_code,
        "is_value_model": args.is_value_model,
        "local_dir": args.local_dir,
        "hf_model_config_path": _default_hf_model_config_path(args.backend, args.local_dir),
        "use_cpu_initialization": args.use_cpu_initialization,
        "base_model_path": args.base_model_path,
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
    return config


class BaseModelMerger(ABC):
    """
    Abstract base class for merging distributed model checkpoints into HuggingFace format.

    This class provides common functionality for converting model checkpoints from different
    distributed training backends (FSDP, Megatron) into standard HuggingFace format that
    can be easily loaded and used for inference or further training.

    The merger supports two main operations:
    - merge: Convert and save checkpoints to HuggingFace format
    - test: Validate merged checkpoints against a reference model

    Args:
        config (ModelMergerConfig): Configuration object containing paths, backend type,
            and operation parameters.

    Attributes:
        config (ModelMergerConfig): The configuration object passed during initialization.
        hf_model_config_path (str): Path to the HuggingFace model configuration files.
        model_config (PretrainedConfig): Loaded HuggingFace model configuration.
    """

    def __init__(self, config: ModelMergerConfig):
        self.config = config
        self.hf_model_config_path = config.hf_model_config_path
        self.model_config = AutoConfig.from_pretrained(
            self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code
        )

    def get_transformers_auto_model_class(self):
        has_remote_code = hasattr(self.model_config, "auto_map") and any(
            self.model_config.architectures[0] in val for val in self.model_config.auto_map.values()
        )
        if has_remote_code:
            auto_class = next(
                k for k, v in self.model_config.auto_map.items() if self.model_config.architectures[0] in v
            )
            match auto_class:
                case "AutoModelForCausalLM":
                    return AutoModelForCausalLM
                case "AutoModelForTokenClassification":
                    return AutoModelForTokenClassification
                case "AutoModelForVision2Seq":
                    return AutoModelForVision2Seq
                case "AutoModelForImageTextToText":
                    return AutoModelForVision2Seq
                case _:
                    raise NotImplementedError(f"Unknown auto class {auto_class}")
        else:
            if "ForTokenClassification" in self.model_config.architectures[0]:
                return AutoModelForTokenClassification
            elif "ForCausalLM" in self.model_config.architectures[0]:
                return AutoModelForCausalLM
            elif "ForConditionalGeneration" in self.model_config.architectures[0]:
                return AutoModelForVision2Seq

            raise NotImplementedError(f"Unknown architecture {self.model_config.architectures}")

    def patch_model_generation_config(self, model):
        """
        The generation_config created from model config may be different to the pretrained model,
        this may lead to error when generating: https://github.com/verl-project/verl/issues/1246

        This function patch the generation_config created from model config to the pretrained model.
        """
        if model.can_generate():
            generation_config = self.load_generation_config()
            if generation_config is not None:
                model.generation_config = generation_config
        return model

    def load_generation_config(self, warn_if_missing: bool = True):
        """Load the generation config stored alongside the pretrained model config."""
        generation_config_path = os.path.join(self.hf_model_config_path, "generation_config.json")
        if not os.path.isfile(generation_config_path):
            if warn_if_missing:
                print(
                    f"Warning: Generation config file not found in {self.hf_model_config_path}, using a "
                    f"generation config created from the model config."
                )
            return None
        return GenerationConfig.from_pretrained(self.hf_model_config_path)

    def save_generation_config(self, target_dir):
        """Preserve the pretrained generation config in a merged model directory."""
        generation_config = self.load_generation_config(warn_if_missing=False)
        if generation_config is not None:
            generation_config.save_pretrained(target_dir)
        else:
            stale_config_path = os.path.join(target_dir, "generation_config.json")
            if os.path.isfile(stale_config_path):
                os.remove(stale_config_path)

    @staticmethod
    def _parse_single_lora_meta_entry(entry: dict, meta_path: str) -> dict[str, object]:
        parsed: dict[str, object] = {}
        if "r" in entry:
            try:
                parsed["r"] = int(entry["r"])
            except (TypeError, ValueError):
                warnings.warn(f"Invalid LoRA rank in {meta_path}: {entry['r']}", stacklevel=2)

        if "lora_alpha" in entry:
            try:
                parsed["lora_alpha"] = int(entry["lora_alpha"])
            except (TypeError, ValueError):
                warnings.warn(f"Invalid lora_alpha in {meta_path}: {entry['lora_alpha']}", stacklevel=2)

        if "task_type" in entry:
            task_type = entry["task_type"]
            if task_type is None:
                pass
            elif isinstance(task_type, str):
                parsed["task_type"] = task_type
            else:
                warnings.warn(f"Invalid task_type in {meta_path}: {task_type}", stacklevel=2)

        if "modules_to_save" in entry:
            mts = entry["modules_to_save"]
            if isinstance(mts, list) and all(isinstance(x, str) for x in mts):
                parsed["modules_to_save"] = list(mts)
            elif mts is not None:
                warnings.warn(f"Invalid modules_to_save in {meta_path}: {mts}", stacklevel=2)

        if "bias" in entry:
            bias = entry["bias"]
            if isinstance(bias, str):
                parsed["bias"] = bias
            elif bias is not None:
                warnings.warn(f"Invalid bias in {meta_path}: {bias}", stacklevel=2)

        if "target_modules" in entry:
            tm = entry["target_modules"]
            if tm is None or isinstance(tm, str):
                parsed["target_modules"] = tm
            elif isinstance(tm, list) and all(isinstance(x, str) for x in tm):
                parsed["target_modules"] = list(tm)
            else:
                warnings.warn(f"Invalid target_modules in {meta_path}: {tm}", stacklevel=2)

        if "target_parameters" in entry:
            tp = entry["target_parameters"]
            if tp is None:
                pass
            elif isinstance(tp, str):
                parsed["target_parameters"] = tp
            elif isinstance(tp, list) and all(isinstance(x, str) for x in tp):
                parsed["target_parameters"] = list(tp)
            else:
                warnings.warn(f"Invalid target_parameters in {meta_path}: {tp}", stacklevel=2)

        if "use_rslora" in entry:
            use_rslora = entry["use_rslora"]
            if isinstance(use_rslora, bool):
                parsed["use_rslora"] = use_rslora
            elif use_rslora is not None:
                warnings.warn(f"Invalid use_rslora in {meta_path}: {use_rslora}", stacklevel=2)
        return parsed

    def _load_lora_train_meta(self) -> Optional[dict[str, object]]:
        if not self.config.local_dir:
            return None

        meta_path = os.path.join(self.config.local_dir, "lora_train_meta.json")
        if not os.path.exists(meta_path):
            return None

        import json

        try:
            with open(meta_path, encoding="utf-8") as f:
                lora_meta = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to read LoRA metadata from {meta_path}: {e}", stacklevel=2)
            return None

        if not isinstance(lora_meta, dict):
            warnings.warn(
                f"Invalid LoRA metadata in {meta_path}: expected JSON object, got {type(lora_meta).__name__}.",
                stacklevel=2,
            )
            return None

        result = self._parse_single_lora_meta_entry(lora_meta, meta_path)

        raw_adapters = lora_meta.get("adapters")
        if isinstance(raw_adapters, dict):
            parsed_adapters = {}
            for adapter_name, entry in raw_adapters.items():
                if isinstance(entry, dict):
                    parsed_entry = self._parse_single_lora_meta_entry(entry, meta_path)
                    if parsed_entry:
                        parsed_adapters[str(adapter_name)] = parsed_entry
            if parsed_adapters:
                result["adapters"] = parsed_adapters

        return result if len(result) > 0 else None

    @staticmethod
    def _extract_target_module_name(stripped_lora_key: str) -> str:
        parts = stripped_lora_key.split(".")
        for idx, part in enumerate(parts):
            if part.startswith("lora_") and idx > 0:
                return parts[idx - 1]
        return parts[-3] if len(parts) >= 3 else parts[0]

    def save_lora_adapter(self, state_dict: dict[str, torch.Tensor]):
        """
        Save lora adapter(s) to safetensors.

        For single-adapter checkpoints, writes to ``<target_dir>/lora_adapter``.
        For multi-adapter checkpoints, writes each adapter to
        ``<target_dir>/lora_adapters/<adapter_name>``.

        Returns:
            lora_path: str, the path to the lora adapter directory (or parent
                ``lora_adapters`` directory for multi-adapter checkpoints).
                None if no lora adapter found.

        Note:
            This function mutates ``state_dict`` in place.
        """
        adapter_param_names = [name for name in state_dict.keys() if "lora_" in name or ".modules_to_save." in name]

        if len(adapter_param_names) == 0:
            return None

        import json
        from typing import OrderedDict

        import peft
        from safetensors.torch import save_file

        from verl.utils.lora_utils import (
            DEFAULT_ADAPTER_NAME,
            extract_adapter_name_from_param_key,
            strip_adapter_name_from_param_key,
            validate_adapter_name,
        )

        lora_meta = self._load_lora_train_meta()
        adapters_meta = lora_meta.get("adapters") if isinstance(lora_meta, dict) else None
        known_adapters = set(adapters_meta.keys()) if isinstance(adapters_meta, dict) else None

        grouped_names: OrderedDict[str, list[str]] = OrderedDict()
        for name in adapter_param_names:
            adapter_name = (
                extract_adapter_name_from_param_key(name, known_adapters=known_adapters) or DEFAULT_ADAPTER_NAME
            )
            grouped_names.setdefault(adapter_name, []).append(name)

        is_multi_adapter = len(grouped_names) > 1
        out_root = os.path.join(self.config.target_dir, "lora_adapters") if is_multi_adapter else None

        last_lora_path = None
        for adapter_name, names in grouped_names.items():
            validate_adapter_name(adapter_name)
            lora_params: OrderedDict[str, torch.Tensor] = OrderedDict()
            target_modules = set()
            inferred_modules_to_save = set()
            inferred_lora_rank = None

            for name in names:
                is_modules_to_save = ".modules_to_save." in name
                lora_key = strip_adapter_name_from_param_key(name, adapter_name)
                if is_modules_to_save:
                    parts = name.split(".")
                    mts_idx = parts.index("modules_to_save")
                    if mts_idx > 0:
                        prefix_path = ".".join(parts[:mts_idx])
                        if prefix_path.startswith("base_model.model."):
                            prefix_path = prefix_path[len("base_model.model.") :]
                        elif prefix_path.startswith("base_model."):
                            prefix_path = prefix_path[len("base_model.") :]
                        if prefix_path:
                            inferred_modules_to_save.add(prefix_path)
                else:
                    target_modules.add(self._extract_target_module_name(lora_key))
                tensor = state_dict.pop(name)
                lora_params[lora_key] = tensor
                if not is_modules_to_save and inferred_lora_rank is None and tensor.ndim >= 2:
                    inferred_lora_rank = min(tensor.shape[0], tensor.shape[1])

            if inferred_lora_rank is None:
                first_tensor = next(iter(lora_params.values()))
                inferred_lora_rank = int(first_tensor.shape[0]) if first_tensor.ndim > 0 else 0

            adapter_meta = None
            if isinstance(adapters_meta, dict) and adapter_name in adapters_meta:
                adapter_meta = adapters_meta[adapter_name]
            elif not is_multi_adapter and isinstance(lora_meta, dict):
                adapter_meta = lora_meta

            lora_rank = inferred_lora_rank
            lora_alpha = 0
            task_type = None
            modules_to_save = sorted(inferred_modules_to_save) if inferred_modules_to_save else None
            bias = None
            has_meta_target_modules = False
            meta_target_modules = None
            target_parameters = None
            use_rslora = None

            if adapter_meta is not None:
                meta_rank = adapter_meta.get("r")
                if meta_rank is not None and meta_rank > 0:
                    if meta_rank != inferred_lora_rank:
                        warnings.warn(
                            f"LoRA rank mismatch for adapter {adapter_name!r} between metadata ({meta_rank}) "
                            f"and adapter weights ({inferred_lora_rank}); using metadata rank.",
                            stacklevel=2,
                        )
                    lora_rank = meta_rank

                meta_alpha = adapter_meta.get("lora_alpha")
                if meta_alpha is not None:
                    lora_alpha = meta_alpha

                meta_task_type = adapter_meta.get("task_type")
                if meta_task_type is not None:
                    task_type = meta_task_type

                meta_mts = adapter_meta.get("modules_to_save")
                if meta_mts:
                    uncovered_inferred = {
                        path
                        for path in inferred_modules_to_save
                        if not any(path == m or path.endswith(f".{m}") for m in meta_mts)
                    }
                    modules_to_save = sorted(set(meta_mts) | uncovered_inferred)

                meta_bias = adapter_meta.get("bias")
                if meta_bias is not None:
                    bias = meta_bias

                if "target_modules" in adapter_meta:
                    has_meta_target_modules = True
                    meta_target_modules = adapter_meta["target_modules"]

                meta_target_parameters = adapter_meta.get("target_parameters")
                if meta_target_parameters is not None:
                    target_parameters = meta_target_parameters

                meta_use_rslora = adapter_meta.get("use_rslora")
                if meta_use_rslora is not None:
                    use_rslora = meta_use_rslora

            if lora_alpha == 0:
                warnings.warn(
                    f"LoRA alpha metadata for adapter {adapter_name!r} is missing or equals 0; "
                    f"falling back to lora_alpha=0. Please verify checkpoint LoRA metadata (lora_train_meta.json).",
                    stacklevel=2,
                )

            resolved_target_modules = meta_target_modules if has_meta_target_modules else sorted(target_modules)
            peft_dict = {
                "r": lora_rank,
                "lora_alpha": lora_alpha,
                "target_modules": resolved_target_modules,
            }
            if target_parameters is not None:
                peft_dict["target_parameters"] = target_parameters
            if task_type is not None:
                peft_dict["task_type"] = task_type
            if modules_to_save:
                peft_dict["modules_to_save"] = modules_to_save
            if bias is not None:
                peft_dict["bias"] = bias
            if use_rslora is not None:
                peft_dict["use_rslora"] = use_rslora
            peft_config = peft.LoraConfig(**peft_dict).to_dict()
            peft_config["task_type"] = (
                peft_config["task_type"].value
                if hasattr(peft_config["task_type"], "value")
                else (peft_config["task_type"] or None)
            )
            peft_config["peft_type"] = (
                peft_config["peft_type"].value
                if hasattr(peft_config["peft_type"], "value")
                else (peft_config["peft_type"] or None)
            )
            if isinstance(peft_config.get("target_modules"), (set, tuple)):
                peft_config["target_modules"] = sorted(peft_config["target_modules"])
            if isinstance(peft_config.get("target_parameters"), (set, tuple)):
                peft_config["target_parameters"] = sorted(peft_config["target_parameters"])

            lora_path = (
                os.path.join(out_root, adapter_name)
                if is_multi_adapter
                else os.path.join(self.config.target_dir, "lora_adapter")
            )
            os.makedirs(lora_path, exist_ok=True)
            with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)
            save_file(lora_params, os.path.join(lora_path, "adapter_model.safetensors"))
            last_lora_path = lora_path

        for name in list(state_dict.keys()):
            key = name.replace("base_model.model.", "").replace(".base_layer.", ".").replace(".original_module.", ".")
            if key.startswith("original_module."):
                key = key[len("original_module.") :]
            if key.startswith("base_layer."):
                key = key[len("base_layer.") :]
            state_dict[key] = state_dict.pop(name)

        return out_root if is_multi_adapter else last_lora_path

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()
        with init_empty_weights():
            model = auto_model_class.from_config(
                self.model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.config.trust_remote_code
            )
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        lora_path = self.save_lora_adapter(state_dict)
        if lora_path:
            print(f"Saving lora adapter to {lora_path}")

        drop_tied_target_keys(state_dict, model, self.model_config)

        print(f"Saving model to {self.config.target_dir}")
        refused_generation_config = False
        if model.can_generate():
            try:
                model.generation_config.validate(strict=True)
            except ValueError:
                # model.save_pretrained saves the generation config before the weights, and raises for one that
                # transformers refuses to save (GLM-5's sets top_p without do_sample). Save the model with a
                # generation config built from the model config, then write the pretrained one after it.
                # TODO: drop once verl requires a transformers release with huggingface/transformers#49914.
                refused_generation_config = True
                model.generation_config = GenerationConfig.from_model_config(model.config)
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        if refused_generation_config:
            copy_generation_config(self.hf_model_config_path, self.config.target_dir)
        del state_dict
        del model

        processor = hf_processor(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        tokenizer = hf_tokenizer(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        if processor is not None:
            print(f"Saving processor to {self.config.target_dir}")
            processor.save_pretrained(self.config.target_dir)
        if tokenizer is not None:
            print(f"Saving tokenizer to {self.config.target_dir}")
            tokenizer.save_pretrained(self.config.target_dir)

        validate_hf_model_output(self.config.target_dir)

    def upload_to_huggingface(self):
        import requests
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

        api = HfApi()
        try:
            # Attempt to create repository
            api.create_repo(repo_id=self.config.hf_upload_path, private=self.config.private, exist_ok=True)
        except HfHubHTTPError as e:
            # Handle authentication/API errors
            if e.response.status_code == 401:
                raise PermissionError(
                    "Hugging Face authentication failed. Verify your token is valid and has write permissions."
                ) from e
            elif e.response.status_code == 404:
                raise RepositoryNotFoundError(f"Repository path not found: {self.config.hf_upload_path}") from e
            else:
                raise ConnectionError(f"Failed to create repository ({e.response.status_code}): {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("Network connection failed. Check your internet connection.") from e

        try:
            # Attempt folder upload
            api.upload_folder(folder_path=self.config.target_dir, repo_id=self.config.hf_upload_path, repo_type="model")
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise PermissionError("Authentication failed during upload. Token may have expired.") from e
            else:
                raise RuntimeError(f"Upload failed ({e.response.status_code}): {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("Network interruption during upload. Try again with stable connection.") from e
        except OSError as e:
            raise FileNotFoundError(f"Local folder error: {self.config.target_dir} - {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during upload: {str(e)}") from e

    @abstractmethod
    def merge_and_save(self):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def cleanup(self):
        raise NotImplementedError("Subclasses should implement this method to clean up resources if needed")

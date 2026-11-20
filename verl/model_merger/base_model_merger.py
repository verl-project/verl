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
    merge_parser.add_argument(
        "--fuse-lora",
        action="store_true",
        help="Fuse LoRA adapter weights into the base model before saving. "
        "Without this flag, LoRA checkpoints are split into base model + separate adapter directory.",
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
    fuse_lora: bool = False

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
            fuse_lora=getattr(args, "fuse_lora", False),
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

        result = {}
        if "r" in lora_meta:
            try:
                result["r"] = int(lora_meta["r"])
            except (TypeError, ValueError):
                warnings.warn(f"Invalid LoRA rank in {meta_path}: {lora_meta['r']}", stacklevel=2)

        if "lora_alpha" in lora_meta:
            try:
                result["lora_alpha"] = int(lora_meta["lora_alpha"])
            except (TypeError, ValueError):
                warnings.warn(f"Invalid lora_alpha in {meta_path}: {lora_meta['lora_alpha']}", stacklevel=2)

        if "task_type" in lora_meta:
            task_type = lora_meta["task_type"]
            if task_type is None:
                pass
            elif isinstance(task_type, str):
                result["task_type"] = task_type
            else:
                warnings.warn(f"Invalid task_type in {meta_path}: {task_type}", stacklevel=2)

        if "use_rslora" in lora_meta and lora_meta["use_rslora"] is not None:
            if isinstance(lora_meta["use_rslora"], bool):
                result["use_rslora"] = lora_meta["use_rslora"]
            else:
                warnings.warn(f"Invalid use_rslora in {meta_path}: {lora_meta['use_rslora']}", stacklevel=2)

        if "target_parameters" in lora_meta and lora_meta["target_parameters"] is not None:
            if isinstance(lora_meta["target_parameters"], list):
                result["target_parameters"] = [str(p) for p in lora_meta["target_parameters"]]
            else:
                warnings.warn(
                    f"Invalid target_parameters in {meta_path}: {lora_meta['target_parameters']}",
                    stacklevel=2,
                )

        if "alpha_pattern" in lora_meta and lora_meta["alpha_pattern"] is not None:
            if isinstance(lora_meta["alpha_pattern"], dict):
                result["alpha_pattern"] = {str(k): v for k, v in lora_meta["alpha_pattern"].items()}
            else:
                warnings.warn(f"Invalid alpha_pattern in {meta_path}: {lora_meta['alpha_pattern']}", stacklevel=2)

        if "rank_pattern" in lora_meta and lora_meta["rank_pattern"] is not None:
            if isinstance(lora_meta["rank_pattern"], dict):
                result["rank_pattern"] = {str(k): int(v) for k, v in lora_meta["rank_pattern"].items()}
            else:
                warnings.warn(f"Invalid rank_pattern in {meta_path}: {lora_meta['rank_pattern']}", stacklevel=2)

        return result if len(result) > 0 else None

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
            if lora_key.endswith((".lora_embedding_A.default", ".lora_embedding_B.default")):
                lora_key = lora_key.removesuffix(".default")
            lora_params[lora_key] = state_dict.pop(name)

        inferred_lora_rank = min(lora_params[lora_key].shape[0], lora_params[lora_key].shape[1])
        lora_meta = self._load_lora_train_meta()

        lora_rank = inferred_lora_rank
        lora_alpha = 0
        task_type = None
        use_rslora = False

        if lora_meta is not None:
            meta_rank = lora_meta.get("r")
            if meta_rank is not None and meta_rank > 0:
                if meta_rank != inferred_lora_rank:
                    warnings.warn(
                        f"LoRA rank mismatch between metadata ({meta_rank}) and adapter weights "
                        f"({inferred_lora_rank}); using metadata rank.",
                        stacklevel=2,
                    )
                lora_rank = meta_rank

            meta_alpha = lora_meta.get("lora_alpha")
            if meta_alpha is not None:
                lora_alpha = meta_alpha

            meta_task_type = lora_meta.get("task_type")
            if meta_task_type is not None:
                task_type = meta_task_type

            meta_use_rslora = lora_meta.get("use_rslora")
            if isinstance(meta_use_rslora, bool):
                use_rslora = meta_use_rslora

        if lora_alpha == 0:
            warnings.warn(
                "LoRA alpha metadata is missing or equals 0; falling back to lora_alpha=0. "
                "Please verify checkpoint LoRA metadata (lora_train_meta.json).",
                stacklevel=2,
            )

        peft_dict = {
            "r": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": list(target_modules),
            "use_rslora": use_rslora,
        }
        if task_type is not None:
            peft_dict["task_type"] = task_type
        if lora_meta is not None:
            if lora_meta.get("alpha_pattern"):
                peft_dict["alpha_pattern"] = dict(lora_meta["alpha_pattern"])
            if lora_meta.get("rank_pattern"):
                peft_dict["rank_pattern"] = dict(lora_meta["rank_pattern"])
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

    def _fuse_lora_into_model(self, state_dict: dict[str, torch.Tensor], model) -> Optional[str]:
        """Fuse LoRA adapter weights into the base model.

        Loads the base weights into the model, applies the LoRA adapter
        using PEFT, calls merge_and_unload() to fold the adapter into
        the base weights, and replaces state_dict contents with the
        fused weights.

        Returns:
            None (no separate adapter directory is created).
        """
        lora_params_names = [name for name in state_dict.keys() if "lora_" in name]
        if len(lora_params_names) == 0:
            return None

        base_params_names = [name for name in state_dict.keys() if "lora_" not in name]
        if len(base_params_names) == 0:
            raise ValueError(
                f"Cannot fuse LoRA weights in {self.config.local_dir}: "
                "checkpoint contains only LoRA adapter weights and no base model weights."
            )

        lora_meta = self._load_lora_train_meta()
        if not lora_meta or not lora_meta.get("lora_alpha"):
            raise ValueError(
                f"Cannot fuse LoRA weights in {self.config.local_dir}: "
                "lora_train_meta.json must record a nonzero lora_alpha."
            )
        if not isinstance(lora_meta.get("use_rslora"), bool):
            raise ValueError(
                f"Cannot fuse LoRA weights in {self.config.local_dir}: "
                "lora_train_meta.json must explicitly record boolean use_rslora."
            )

        has_param_targeted_keys = any(
            ".base_layer." in name and not name.endswith((".base_layer.weight", ".base_layer.bias"))
            for name in base_params_names
        )
        if lora_meta.get("target_parameters") or has_param_targeted_keys:
            raise ValueError(
                f"Cannot fuse LoRA weights in {self.config.local_dir}: "
                "adapters configured with target_parameters are not supported by --fuse-lora."
            )

        import shutil

        import peft

        # First, save the adapter to a temp directory so PeftModel can load it
        lora_path = self.save_lora_adapter(state_dict)
        if lora_path is None:
            return None

        try:
            # state_dict now has only base weights (save_lora_adapter pops LoRA keys).
            # Discover tied parameter groups first, populate missing tied aliases in state_dict,
            # and untie duplicate Parameter instances before PEFT wrapping so merging an embedding
            # adapter cannot mutate an untargeted tied lm_head.
            tied_groups: list[list[str]] = []
            if callable(getattr(model, "tie_weights", None)):
                model.tie_weights()
            if callable(getattr(model, "named_parameters", None)):
                names_by_id: dict[int, list[str]] = {}
                for name, param in model.named_parameters(remove_duplicate=False):
                    if isinstance(name, str):
                        names_by_id.setdefault(id(param), []).append(name)
                tied_groups = [group for group in names_by_id.values() if len(group) > 1]

            for group in tied_groups:
                present_alias = next((alias for alias in group if alias in state_dict), None)
                if present_alias is not None:
                    for alias in group:
                        if alias not in state_dict:
                            state_dict[alias] = state_dict[present_alias].clone()
                if callable(getattr(model, "get_submodule", None)):
                    for alias in group[1:]:
                        submod_path, _, attr_name = alias.rpartition(".")
                        try:
                            parent_mod = model.get_submodule(submod_path) if submod_path else model
                            orig_param = getattr(parent_mod, attr_name, None)
                            if isinstance(orig_param, torch.nn.Parameter):
                                setattr(
                                    parent_mod,
                                    attr_name,
                                    torch.nn.Parameter(
                                        orig_param.detach().clone(),
                                        requires_grad=orig_param.requires_grad,
                                    ),
                                )
                        except Exception:
                            pass

            load_result = model.load_state_dict(state_dict, strict=False)
            missing_keys = getattr(load_result, "missing_keys", None)
            if isinstance(missing_keys, list | tuple) and missing_keys:
                tied_satisfied: set[str] = set()
                for group in tied_groups:
                    if any(alias in state_dict for alias in group):
                        tied_satisfied.update(group)
                unresolved_missing = [key for key in missing_keys if key not in tied_satisfied]
                if unresolved_missing:
                    raise ValueError(
                        f"Cannot fuse LoRA weights in {self.config.local_dir}: "
                        f"missing base model parameters: {unresolved_missing}"
                    )

            # Apply LoRA adapter and fuse
            orig_tie_word_embeddings = getattr(getattr(model, "config", None), "tie_word_embeddings", None)
            if tied_groups and getattr(model, "config", None) is not None:
                model.config.tie_word_embeddings = False
            print("Fusing LoRA adapter into base model weights...")
            peft_model = peft.PeftModel.from_pretrained(model, lora_path)
            fused_model = peft_model.merge_and_unload()

            # Replace state_dict with fused weights
            state_dict.clear()
            state_dict.update(fused_model.state_dict())

            # If any formerly tied parameters diverged after fusion (e.g. embedding LoRA without lm_head LoRA),
            # disable tie_word_embeddings so drop_tied_target_keys and save_pretrained keep both tensors distinct;
            # otherwise restore the original tie_word_embeddings setting.
            tied_diverged = False
            for group in tied_groups:
                present = [alias for alias in group if alias in state_dict]
                if len(present) > 1 and any(
                    not torch.equal(state_dict[present[0]], state_dict[a]) for a in present[1:]
                ):
                    tied_diverged = True
                    break
            if tied_diverged:
                if getattr(self, "model_config", None) is not None:
                    self.model_config.tie_word_embeddings = False
                if getattr(model, "config", None) is not None:
                    model.config.tie_word_embeddings = False
            elif tied_groups and orig_tie_word_embeddings is not None and getattr(model, "config", None) is not None:
                model.config.tie_word_embeddings = orig_tie_word_embeddings
        finally:
            # Clean up the temp adapter directory on both success and failure
            shutil.rmtree(lora_path, ignore_errors=True)

        print("LoRA adapter fused into model weights.")
        return None

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()
        with init_empty_weights():
            model = auto_model_class.from_config(
                self.model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.config.trust_remote_code
            )
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        if self.config.fuse_lora:
            lora_path = self._fuse_lora_into_model(state_dict, model)
        else:
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

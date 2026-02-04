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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    GenerationConfig,
)

from verl.utils import hf_processor, hf_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="verl model merger")
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Specify 'merge' or 'test' operation.")

    base_op_parser = argparse.ArgumentParser(add_help=False)
    base_op_parser.add_argument(
        "--backend", type=str, required=True, choices=["fsdp", "megatron"], help="The backend of the model"
    )
    base_op_parser.add_argument("--local_dir", type=str, default=None, help="Path to the saved model checkpoints.")
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
        backend (str): Backend type for the model ('fsdp' or 'megatron').
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

    def __post_init__(self):
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)
        if self.operation == "test":
            self.target_dir = None
            self.hf_upload_path = None
            self.private = False


def generate_config_from_args(args: argparse.Namespace) -> ModelMergerConfig:
    common_config_args = {
        "operation": args.operation,
        "backend": args.backend,
        "tie_word_embedding": args.tie_word_embedding,
        "trust_remote_code": args.trust_remote_code,
        "is_value_model": args.is_value_model,
        "local_dir": args.local_dir,
        "hf_model_config_path": os.path.join(args.local_dir, "huggingface"),
        "use_cpu_initialization": args.use_cpu_initialization,
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
                    # Handle different transformers versions for Vision2Seq models
                    import transformers
                    from packaging import version

                    if version.parse(transformers.__version__) >= version.parse("4.54.0"):
                        # transformers >= 4.54.0 uses AutoModelForImageTextToText
                        from transformers import AutoModelForImageTextToText

                        return AutoModelForImageTextToText
                    else:
                        # transformers < 4.54.0 uses AutoModelForVision2Seq
                        from transformers import AutoModelForVision2Seq

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
        this may lead to error when generating: https://github.com/volcengine/verl/issues/1246

        This function patch the generation_config created from model config to the pretrained model.
        """
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(self.hf_model_config_path)
            except OSError:
                print(
                    f"Warning: Generation config file not found in {self.hf_model_config_path}, using a "
                    f"generation config created from the model config."
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
            model = auto_model_class.from_config(
                self.model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.config.trust_remote_code
            )
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        lora_path = self.save_lora_adapter(state_dict)
        if lora_path:
            print(f"Saving lora adapter to {lora_path}")

        print(f"Saving model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
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

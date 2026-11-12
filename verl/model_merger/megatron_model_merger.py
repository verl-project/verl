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

"""Export MCore distributed checkpoints through Megatron-Bridge."""

from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import init_empty_weights

from verl.models.mcore.bridge import AutoBridge
from verl.models.mcore.bridge_checkpoint import (
    build_conversion_model,
    compare_hf_weights,
    get_dynamic_pipeline_shards,  # noqa: F401 (compatibility for existing callers)
    initialize_conversion,
    load_conversion_checkpoint,
)

from .base_model_merger import BaseModelMerger, ModelMergerConfig
from .output_validation import validate_hf_model_output


class MegatronModelMerger(BaseModelMerger):
    """Load a verl model checkpoint and delegate architecture mappings to Bridge."""

    def __init__(self, config: ModelMergerConfig):
        super().__init__(config)
        initialize_conversion()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        hf_config = deepcopy(self.model_config)
        if config.is_value_model:
            # Bridge builds the backbone as a causal LM; the engine's standard
            # pre-wrap hook replaces its output layer with a scalar value head.
            hf_config.architectures = [
                name.replace("ForTokenClassification", "ForCausalLM") for name in hf_config.architectures
            ]
            hf_config.tie_word_embeddings = False
        if config.tie_word_embedding and not config.is_value_model:
            hf_config.tie_word_embeddings = True
            if hasattr(hf_config, "text_config"):
                hf_config.text_config.tie_word_embeddings = True
        if config.operation == "test" and not config.is_value_model:
            # The reference checkpoint owns its on-disk expert layout, which may
            # differ from the layout used by the installed Transformers version.
            self.bridge = AutoBridge.from_hf_pretrained(config.test_hf_dir, trust_remote_code=config.trust_remote_code)
        elif config.is_value_model:
            self.bridge = AutoBridge.from_hf_config(hf_config)
        else:
            # A config alone does not describe packed/per-expert HF weight keys.
            # Give Bridge a weightless HF model so it exports the layout expected
            # by the installed Transformers implementation, without requiring the
            # original pretrained weights or allocating a second full model.
            self.bridge = AutoBridge.from_hf_pretrained(
                config.hf_model_config_path, trust_remote_code=config.trust_remote_code, device="meta"
            )
            self.bridge.hf_pretrained.config = hf_config
            with init_empty_weights():
                self.bridge.hf_pretrained.model = self.get_transformers_auto_model_class().from_config(
                    hf_config, torch_dtype=torch.bfloat16, trust_remote_code=config.trust_remote_code
                )
        self.bridge.hf_model_id = config.hf_model_config_path
        self.bridge.trust_remote_code = config.trust_remote_code
        self.models = build_conversion_model(
            self.bridge,
            self.world_size,
            use_cpu_initialization=config.use_cpu_initialization,
            is_value_model=config.is_value_model,
        )
        if config.operation == "merge" and not config.is_value_model:
            hf_keys = set(self.bridge.hf_pretrained.state.keys())
            tasks = self.bridge.get_conversion_tasks(self.models)
            mapped_keys = set()
            for task in tasks:
                if task is None:
                    continue
                names = task.mapping.hf_param
                mapped_keys.update([names] if isinstance(names, str) else names.values())
            if mapped_keys - hf_keys:
                # Older Bridges may export the legacy HF layout, which
                # Transformers converts while loading (e.g. individual experts).
                # Do not let the meta state filter those conversion tasks out.
                print("Bridge mappings use a legacy HF layout; exporting with config-only mappings", flush=True)
                self.bridge = AutoBridge.from_hf_config(hf_config)
                self.bridge.hf_model_id = config.hf_model_config_path
                self.bridge.trust_remote_code = config.trust_remote_code

    def _export_weights(self):
        for name, tensor in self.bridge.export_hf_weights(self.models, cpu=True):
            if self.config.is_value_model and name == "lm_head.weight":
                name = "score.weight"
            yield name, tensor

    def merge_and_save(self):
        checkpoint = Path(self.config.local_dir) / "model" / "dist_ckpt"
        load_conversion_checkpoint(self.models, checkpoint)
        if self.config.operation == "test":
            compare_hf_weights(self._export_weights(), self.config.test_hf_dir)
        elif self.config.operation == "merge":
            if self.config.is_value_model:
                weights = {}
                for name, tensor in self._export_weights():
                    if self.rank == 0:
                        weights[name] = tensor
                if self.rank == 0:
                    self.model_config.architectures = [
                        name.replace("ForCausalLM", "ForTokenClassification")
                        for name in self.model_config.architectures
                    ]
                    self.model_config.num_labels = 1
                    self.model_config.tie_word_embeddings = False
                    with init_empty_weights():
                        value_model = self.get_transformers_auto_model_class().from_config(self.model_config)
                    bias = value_model.state_dict().get("score.bias")
                    if bias is not None and "score.bias" not in weights:
                        weights["score.bias"] = torch.zeros(bias.shape, dtype=weights["score.weight"].dtype)
                    del value_model
                    self.save_hf_model_and_tokenizer(weights)
            else:
                self.bridge.save_hf_pretrained(
                    self.models, self.config.target_dir, source_path=self.config.hf_model_config_path
                )
            if self.rank == 0:
                validate_hf_model_output(self.config.target_dir)
                if self.config.hf_upload:
                    self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")
        dist.barrier()

    def cleanup(self):
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

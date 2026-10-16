# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

from transformers import PretrainedConfig


class MuseSparkTextConfig(PretrainedConfig):
    """Muse Spark text model config: the checkpoint's ``text_config`` fields, kept verbatim."""

    model_type = "muse_spark_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        super().__init__(
            pad_token_id=kwargs.pop("pad_token_id", 200018),
            bos_token_id=kwargs.pop("bos_token_id", 200000),
            eos_token_id=kwargs.pop("eos_token_id", 200001),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            **kwargs,
        )
        for name, value in kwargs.items():
            setattr(self, name, value)

    @classmethod
    def from_checkpoint_config(cls, config: dict) -> MuseSparkTextConfig:
        drop = ("model_type", "architectures", "quantization_config")
        text = {k: v for k, v in config.get("text_config", config).items() if k not in drop}
        return cls(architectures=["MuseSparkForCausalLM"], dtype="bfloat16", **text)


__all__ = ["MuseSparkTextConfig"]

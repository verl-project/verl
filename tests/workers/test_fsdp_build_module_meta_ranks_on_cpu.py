# Copyright contributors to the PEAR project
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

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tiny untied causal LM saved to disk, as the engine's model path."""
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    path = tmp_path_factory.mktemp("tiny_llama")
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    return config, str(path)


def _engine(checkpoint):
    config, path = checkpoint
    engine = object.__new__(FSDPEngine)
    engine.engine_config = SimpleNamespace(model_dtype=None, forward_only=False, mixed_precision=None)
    engine.model_config = SimpleNamespace(
        hf_config=config,
        model_type="language_model",
        local_path=path,
        trust_remote_code=False,
        use_liger=False,
        fused_kernel_options=None,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
    )
    engine.device_mesh = None
    engine.use_remove_padding = False
    engine.ulysses_sequence_parallel_size = 1
    return engine


def test_a_rank_that_receives_its_weights_does_not_read_the_checkpoint(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def from_pretrained(*args, **kwargs):
        raise AssertionError("a rank on the meta device read the checkpoint")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", from_pretrained)
    module = _engine(checkpoint)._build_module()

    assert all(param.is_meta for param in module.parameters())
    assert all(param.dtype == torch.float32 for param in module.parameters())


def test_the_source_rank_loads_the_checkpoint(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    module = _engine(checkpoint)._build_module()
    saved = AutoModelForCausalLM.from_pretrained(checkpoint[1], torch_dtype=torch.float32)

    loaded = dict(module.named_parameters())
    for name, param in saved.named_parameters():
        assert torch.equal(loaded[name], param), name

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

"""Regression test for FSDP2 loading from a rank-zero CPU full state.

Launch:
    torchrun --standalone --nproc-per-node=2 \
        tests/special_distributed/test_fsdp2_full_state_load.py
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_no_placement_param_registrations,
    materialize_no_placement_params,
    set_no_placement_param_registrations,
    temporarily_detach_no_placement_params,
    to_empty_preserving_shared_params,
)


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, tokens):
        return self.linear(self.embedding(tokens))


class BufferedModel(nn.Module):
    _no_split_modules = ["ToyBlock"]
    _no_placement_params = ["large.weight"]

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.block = ToyBlock()
        self.large = nn.Embedding(8, 4)
        self.large.weight.requires_grad_(False)
        self.register_buffer("marker", torch.arange(4, dtype=torch.bfloat16), persistent=False)

    def forward(self, tokens):
        return self.block(tokens)


def _build_model(rank: int) -> tuple[BufferedModel, dict[str, torch.Tensor]]:
    model = BufferedModel()
    if rank == 0:
        model.large.weight.copy_(torch.arange(32, dtype=torch.float32).view(8, 4))
    else:
        model.large.weight = nn.Parameter(
            torch.empty(model.large.weight.shape, device="meta"),
            requires_grad=False,
        )
    registrations = materialize_no_placement_params(get_no_placement_param_registrations(model))
    set_no_placement_param_registrations(model, registrations)
    with temporarily_detach_no_placement_params(model):
        state = model.state_dict() if rank == 0 else {}
    return model, state


def _forbid_full_model_to(*args, **kwargs) -> None:
    raise AssertionError("full model must not be materialized with Module.to")


def _check_tied_checkpoint_roundtrip(rank, mesh):
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
    state = model.state_dict() if rank == 0 else {}
    buffers = {name: buffer.detach().cpu() for name, buffer in model.named_buffers()}
    to_empty_preserving_shared_params(model, "meta")
    assert model.model.embed_tokens.weight is model.lm_head.weight
    apply_fsdp2(
        model,
        {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16), "reshard_after_forward": True},
        {"wrap_policy": {"transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"]}},
    )
    fsdp2_load_full_state_dict(model, state, mesh, buffers=buffers)
    assert model.model.embed_tokens.weight is model.lm_head.weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    model(input_ids=tokens, labels=tokens).loss.backward()
    optimizer.step()
    merged = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        torch.testing.assert_close(merged["model.embed_tokens.weight"], merged["lm_head.weight"], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(directory, state_dict=merged)
            restored = Qwen2ForCausalLM.from_pretrained(directory, dtype=torch.bfloat16).state_dict()
            for name in merged:
                torch.testing.assert_close(merged[name], restored[name], atol=0, rtol=0)
    dist.barrier()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("test_fsdp2_full_state_load skipped: requires two CUDA devices")
        return
    _, rank, world_size = initialize_global_process_group()
    if world_size != 2:
        raise RuntimeError(f"expected two ranks, got {world_size}")
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    with tempfile.TemporaryDirectory() as directory:
        directories = [directory]
        dist.broadcast_object_list(directories, src=0)
        os.environ["VERL_NO_PLACEMENT_MMAP_DIR"] = directories[0]
        model, full_state = _build_model(rank)
        ptr = model.large.weight.data_ptr()
        mapping = next(
            line.split()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if int(line.split()[0].split("-")[0], 16) <= ptr < int(line.split()[0].split("-")[1], 16)
        )
        backing_files = [None] * world_size
        dist.all_gather_object(backing_files, mapping[3:5])  # device and inode, not virtual address
        assert backing_files[0] == backing_files[1]
        assert mapping[4] != "0"
        dist.barrier()
        assert not list(Path(directories[0]).iterdir())
        dist.barrier()
        del os.environ["VERL_NO_PLACEMENT_MMAP_DIR"]
    expected = (
        full_state["block.linear.weight"].detach().clone().to("cuda")
        if rank == 0
        else torch.empty((4, 4), device="cuda", dtype=torch.float32)
    )
    dist.broadcast(expected, src=0)
    buffers = {name: buffer.detach().cpu() for name, buffer in model.named_buffers()}
    registrations = vars(model)["_verl_no_placement_param_registrations"]
    with temporarily_detach_no_placement_params(model):
        to_empty_preserving_shared_params(model, "meta")
    apply_fsdp2(
        model,
        {
            "mesh": mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            ),
            "offload_policy": None,
            "reshard_after_forward": True,
            "ignored_params": {param for _, _, param, _ in registrations},
        },
        {"wrap_policy": {"transformer_layer_cls_to_wrap": ["ToyBlock"]}},
    )
    model.to = _forbid_full_model_to
    with temporarily_detach_no_placement_params(model):
        fsdp2_load_full_state_dict(model, full_state, mesh, buffers=buffers)

    torch.testing.assert_close(model.block.linear.weight.full_tensor().float(), expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(model.marker, torch.arange(4, device="cuda", dtype=torch.bfloat16))
    assert model.large.weight.device.type == "cpu"
    expected_large = torch.arange(32, dtype=torch.float32).view(8, 4)
    torch.testing.assert_close(model.large.weight, expected_large)
    # A nested embedding must not create an overlapping FSDP unit inside ToyBlock.
    output = model(torch.tensor([1, 2], device="cuda"))
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert model.block.linear.weight.grad is not None
    _check_tied_checkpoint_roundtrip(rank, mesh)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("test_fsdp2_full_state_load passed")


if __name__ == "__main__":
    main()

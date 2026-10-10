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
import os

# flash-attn consumes FLASH_ATTENTION_DETERMINISTIC while *its own package* is being imported, and
# importing `verl` below already imports flash-attn (verl.utils.torch_functional probes
# flash_attn.ops.triton.cross_entropy at module scope, see also huggingface/transformers 64d14ef).
# Setting these variables later -- as this file did, at the bottom of __main__ -- silently has no
# effect, which is what made the bit-exact logits comparison at the end of this test flaky.
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import shutil
import tempfile

import torch
import torch.distributed
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from verl.utils.attention_utils import unpad_input
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2


def create_random_input_ids(batch_size, seq_len, vocab_size):
    from verl.utils.model import compute_position_id_with_mask, create_random_mask

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=get_device_name())

    attention_mask = create_random_mask(
        input_ids, max_ratio_of_left_padding=0.1, min_ratio_of_valid_token=0.5, max_ratio_of_valid_token=0.7
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    input_ids = unpad_input(input_ids.unsqueeze(-1), attention_mask)[0].transpose(0, 1)
    position_ids = unpad_input(position_ids.unsqueeze(-1), attention_mask)[0].transpose(0, 1)
    return input_ids, position_ids


def test_fsdp_ckpt(strategy="fsdp"):
    assert get_torch_device().device_count() >= 2, "need at least 2 gpus for test"
    local_rank, rank, world_size = initialize_global_process_group()
    # The last assertion compares logits from two independent optimizer steps with atol=rtol=0.
    # That is only well defined for deterministic kernels (the embedding backward accumulates with
    # atomics) and reproducible inputs, so pin both here rather than letting the run decide.
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2026)
    device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

    model_name = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    config = Qwen2Config(num_hidden_layers=1)

    with torch.device(get_device_name()):
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model = model.to(device=get_device_name())

    # Wrap model with FSDP
    if strategy == "fsdp":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        model = FSDP(
            model,
            use_orig_params=False,
            device_id=get_torch_device().current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        )
        fsdp_kwargs = {
            "mesh": device_mesh,
            "mp_policy": mp_policy,
        }
        apply_fsdp2(model, fsdp_kwargs, {})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Create checkpoint manager
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    checkpoint_manager = FSDPCheckpointManager(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, tokenizer=tokenizer
    )

    # Generate sample input
    batch_size = 10
    seq_len = 1024
    vocab_size = config.vocab_size
    # First input for initial update
    input_ids1, position_ids1 = create_random_input_ids(batch_size, seq_len, vocab_size)

    # Second input for verification
    input_ids2, position_ids2 = create_random_input_ids(batch_size, seq_len, vocab_size)

    # Step 1: Initial update and save checkpoint
    outputs1 = model(input_ids=input_ids1, position_ids=position_ids1)
    loss1 = outputs1.logits.mean()
    loss1.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    # Save checkpoint after first update
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "checkpoint")
    checkpoint_manager.save_checkpoint(local_path=checkpoint_path, hdfs_path=None, global_step=0)
    # Snapshot the *values* that were just written to disk. ``model.state_dict()`` hands back the
    # live parameter tensors (FSDP1 flat params / FSDP2 DTensors), so keeping the returned dict
    # itself makes the round-trip check below compare the model against itself: it passes no matter
    # what ``load_checkpoint`` restored, and the only real signal left is the flaky logits compare.
    saved_state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}

    # Step 2: Second update and forward pass
    outputs2 = model(input_ids=input_ids2, position_ids=position_ids2)
    loss2 = outputs2.logits.mean()
    loss2.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    # Record logits after second update
    with torch.no_grad():
        logits_before_load = model(input_ids=input_ids2, position_ids=position_ids2).logits

    # Step 3: Load checkpoint and repeat second update
    checkpoint_manager.load_checkpoint(checkpoint_path)
    loaded_state_dict = model.state_dict()
    for key in loaded_state_dict:
        assert key in saved_state_dict, f"Key {key} not found in saved state dict"
        torch.testing.assert_close(loaded_state_dict[key], saved_state_dict[key], atol=0.0, rtol=0.0)

    # Repeat the second update with same input
    outputs3 = model(input_ids=input_ids2, position_ids=position_ids2)
    loss3 = outputs3.logits.mean()
    loss3.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    # Record logits after loaded checkpoint and update
    with torch.no_grad():
        logits_after_load = model(input_ids=input_ids2, position_ids=position_ids2).logits

    # Step 4: Verify outputs match
    torch.testing.assert_close(logits_before_load, logits_after_load, atol=0.0, rtol=0.0)
    print("Checkpoint save/load test passed!")

    # Cleanup
    shutil.rmtree(temp_dir)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    strategy = os.environ.get("STRATEGY", "fsdp")
    test_fsdp_ckpt(strategy=strategy)

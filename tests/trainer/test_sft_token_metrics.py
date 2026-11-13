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

"""Run with torchrun --nproc_per_node=4 to check SFT metrics under CP2/DP2."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


def _run(output=None):
    from megatron.core import parallel_state as mpu

    from verl.trainer.sft_trainer import SFTTrainer

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    mpu.initialize_model_parallel(context_parallel_size=2, create_gloo_process_groups=False)
    records = []
    try:
        lengths = (8, 12) if mpu.get_data_parallel_rank() == 0 else (6, 10)
        trainer = object.__new__(SFTTrainer)
        trainer.device_name = "cuda"
        trainer.engine = SimpleNamespace(
            get_data_parallel_size=mpu.get_data_parallel_world_size,
            get_data_parallel_group=mpu.get_data_parallel_group,
        )
        for padding in (False, True):
            if padding:
                data = {
                    "input_ids": torch.zeros(2, 12, dtype=torch.long, device=device),
                    "attention_mask": torch.arange(12, device=device)[None, :]
                    < torch.tensor(lengths, device=device)[:, None],
                }
            else:
                data = {
                    "input_ids": torch.nested.nested_tensor(
                        [torch.zeros(length, dtype=torch.long, device=device) for length in lengths],
                        layout=torch.jagged,
                    )
                }
            values = trainer._get_batch_seqlens(data)
            assert sorted(values) == [6, 8, 10, 12], ("global_tokens", dist.get_rank(), values)
            records.append({"padding": padding, "batch_lengths": values, "global_tokens": sum(values)})
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()
    if int(os.environ["RANK"]) == 0:
        print(json.dumps(records), flush=True)
        if output is not None:
            Path(output).write_text(json.dumps(records, indent=2) + "\n")


def test_sft_token_metrics_with_context_and_data_parallelism():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available():
        pytest.skip("Requires four GPUs and torchrun")
    _run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    _run(parser.parse_args().output)

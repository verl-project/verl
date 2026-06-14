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
This module is used to merge huggingface model and test verl checkpoints from FSDP and Megatron backends.

To merge FSDP checkpoints:
```sh
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

To merge Megatron checkpoints:
```sh
python -m verl.model_merger merge \
    --backend megatron \
    --tie-word-embedding \
    --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

or use distribtued merge for large models like dpskv3 671B

```sh
torchrun --nproc_per_node 1 --nnodes 8 --node_rank ${RANK} -m verl.model_merger merge\
    --backend megatron \
    --local_dir ./checkpoints/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```


For more details, please refer to documentation:
https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model
"""

from .base_model_merger import generate_config_from_args, parse_args


def main():
    args = parse_args()
    config = generate_config_from_args(args)
    print(f"config: {config}")

    if config.backend == "fsdp":
        from .fsdp_model_merger import FSDPModelMerger

        merger = FSDPModelMerger(config)
    elif config.backend == "megatron":
        from .megatron_model_merger import MegatronModelMerger

        merger = MegatronModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()

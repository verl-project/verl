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
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os
import random

import numpy as np
import torch
from datasets import Dataset
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, get_benchmark


def patched_get_task_init_states(self, i):
    init_states_path = os.path.join(
        get_libero_path("init_states"),
        self.tasks[i].problem_folder,
        self.tasks[i].init_states_file,
    )
    init_states = torch.load(init_states_path, weights_only=False)
    return init_states


Benchmark.get_task_init_states = patched_get_task_init_states


def compute_total_num_group_envs(task_suite: Benchmark):
    total_num_group_envs = 0
    trial_id_bins = []
    for task_id in range(task_suite.get_num_tasks()):
        task_num_trials = len(task_suite.get_task_init_states(task_id))
        trial_id_bins.append(task_num_trials)

        total_num_group_envs += task_num_trials

    cumsum_trial_id_bins = np.cumsum(trial_id_bins)
    return total_num_group_envs, cumsum_trial_id_bins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument(
        "--local_save_dir", default="~/data/libero_rl", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    task_suite = get_benchmark("libero_10")()
    total_num_group_envs, cumsum_trial_id_bins = compute_total_num_group_envs(task_suite)
    print(f"Total number of group envs: {total_num_group_envs}")
    print(f"Cumsum trial id bins: {cumsum_trial_id_bins}")

    # Total number of group envs: 500
    # Cumsum trial id bins: [ 50 100 150 200 250 300 350 400 450 500]
    def get_state_ids_for_task(task_id):
        start_id = 0 if task_id == 0 else cumsum_trial_id_bins[task_id - 1]
        end_id = cumsum_trial_id_bins[task_id]
        return list(range(start_id, end_id))

    all_task_ids = list(range(task_suite.get_num_tasks()))
    train_task_ids = sorted(random.sample(all_task_ids, 9))
    ood_test_task_id = list(set[int](all_task_ids) - set(train_task_ids))[0]  # for OOD test

    print("\n[Data Split Plan]")
    print(f"Training Task IDs: {train_task_ids}")
    print(f"OOD Test Task ID: {ood_test_task_id}")
    train_metadata = []
    test_metadata = []
    for task_id in train_task_ids:
        all_trials = get_state_ids_for_task(task_id)
        random.shuffle(all_trials)
        selected_train_trials = all_trials[:40]
        for state_id in selected_train_trials:
            train_metadata.append({"task_id": task_id, "state_id": state_id, "data_source": "train"})

    # ID
    for task_id in train_task_ids:
        all_trials = get_state_ids_for_task(task_id)
        random.shuffle(all_trials)
        selected_id_test_trials = all_trials[40:]
        for state_id in selected_id_test_trials[:10]:
            test_metadata.append({"task_id": task_id, "state_id": state_id, "data_source": "test_in_distribution"})

    # OOD
    ood_all_trials = get_state_ids_for_task(ood_test_task_id)
    random.shuffle(ood_all_trials)
    selected_ood_trials = ood_all_trials[:20]
    for state_id in selected_ood_trials:
        test_metadata.append(
            {"task_id": ood_test_task_id, "state_id": state_id, "data_source": "test_out_of_distribution"}
        )
    print(f"Generated {len(train_metadata)} training samples.")
    print(f"Generated {len(test_metadata)} testing samples.")
    print("-" * 20)
    train_ds_meta = Dataset.from_list(train_metadata)
    test_ds_meta = Dataset.from_list(test_metadata)

    def map_and_process(example, idx):
        task_id = example["task_id"]
        state_id = example["state_id"]
        data_source = example["data_source"]
        split = "train" if data_source == "train" else "test"
        task = task_suite.get_task(task_id)
        # demonstration = task.get_demonstration(state_id)

        data = {
            "data_source": data_source,
            "prompt": task.language,
            "state_ids": state_id,
            "task_ids": task_id,
            "ability": "robot",
            "extra_info": {
                "split": split,
                "state_ids": state_id,
                "index": idx,
                "task": task,
                "task_ids": task_id,
            },
        }
        return data

    print("Mapping and processing training dataset...")
    train_dataset = train_ds_meta.map(map_and_process, with_indices=True, num_proc=8)
    print("Mapping and processing test dataset...")
    test_dataset = test_ds_meta.map(map_and_process, with_indices=True, num_proc=8)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    print(f"Saving training dataset to {os.path.join(local_save_dir, 'train.parquet')}")
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    print(f"Saving test dataset to {os.path.join(local_save_dir, 'test.parquet')}")
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print("\nDataset generation complete!")

    print("\n--- Verification ---")
    print("Train dataset data sources:", train_dataset.unique("data_source"))
    print("Test dataset data sources:", test_dataset.unique("data_source"))
    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

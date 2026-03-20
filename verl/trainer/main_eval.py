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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

# from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


# @ray.remote
# def process_item(reward_fn, data_source, response_lst, reward_data):
#     ground_truth = reward_data["ground_truth"]
#     score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
#     return data_source, np.mean(score_lst)

@ray.remote
def process_item(data_source, response_lst, reward_data, reward_file_path, reward_func_name):
    """
    在远程节点内部动态加载奖励函数，避免 Ray 序列化自定义模块
    """
    import importlib.util
    import sys
    
    # 动态加载模块
    module_name = "custom_reward_local"
    spec = importlib.util.spec_from_file_location(module_name, reward_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # 获取评分函数
    reward_fn = getattr(module, reward_func_name)
    
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(r, ground_truth, method="flexible") for r in response_lst]
    return data_source, np.mean(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    data_source_correct_count = defaultdict(int)  # 统计正确数量
    # compute_score = get_custom_reward_fn(config)

    # 获取配置中的路径和函数名，而不是加载函数对象
    reward_file_path = config.custom_reward_function.path
    reward_func_name = config.custom_reward_function.name

    # # Create remote tasks
    # remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

     # Create remote tasks - 传递路径字符串而非函数对象
    remote_tasks = [
        process_item.remote(
            data_sources[i], 
            responses[i], 
            reward_model_data[i],
            reward_file_path,      # 传递字符串，可序列化
            reward_func_name       # 传递字符串，可序列化
        ) for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                if score >= 0.99:
                    data_source_correct_count[data_source] += 1
                pbar.update(1)

    # metric_dict = {}
    # for data_source, rewards in data_source_reward.items():
    #     metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    # print(metric_dict)

    # 计算并输出指标
    print("\n" + "="*60)
    print("📊 Evaluation Results")
    print("="*60)
    
    metric_dict = {}
    total_correct = 0
    total_samples = 0
    
    for data_source in sorted(data_source_reward.keys()):
        rewards = data_source_reward[data_source]
        correct = data_source_correct_count[data_source]
        count = len(rewards)
        accuracy = np.mean(rewards) * 100  # 转换为百分比
        
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)
        metric_dict[f"accuracy/{data_source}"] = accuracy
        metric_dict[f"correct/{data_source}"] = correct
        metric_dict[f"total/{data_source}"] = count
        
        total_correct += correct
        total_samples += count
        
        print(f"\n📁 Data Source: {data_source}")
        print(f"   ✅ Correct: {correct}/{count}")
        print(f"   🎯 Accuracy: {accuracy:.2f}%")
        print(f"   📈 Average Score: {np.mean(rewards):.4f}")
    
    # 总体统计
    if len(data_source_reward) > 1:
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        print(f"\n📊 Overall Statistics:")
        print(f"   ✅ Total Correct: {total_correct}/{total_samples}")
        print(f"   🎯 Overall Accuracy: {overall_accuracy:.2f}%")
        metric_dict["test_score/overall"] = total_correct / total_samples if total_samples > 0 else 0
        metric_dict["accuracy/overall"] = overall_accuracy

    print("="*60)
    print("Raw Metrics Dictionary:")
    print(metric_dict)
    print("="*60)

    return metric_dict


if __name__ == "__main__":
    main()

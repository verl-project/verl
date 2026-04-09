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

import os
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.utils.fs import copy_to_local
from verl.utils.eval.metrics import compute_all_metrics


@ray.remote
def process_item(data_source, response_lst, reward_data, reward_file_path, reward_func_name, pred_func_name, idx):
    """
    在远程节点内部动态加载奖励函数，避免 Ray 序列化自定义模块
    现在返回详细的每条 response 的得分和索引，用于后续保存
    """
    import importlib.util
    import sys
    
    # 动态加载模块
    module_name = "custom_reward_local"
    spec = importlib.util.spec_from_file_location(module_name, reward_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # 获取评分函数 & pred extract fun.
    reward_fn = getattr(module, reward_func_name)
    
    ground_truth = reward_data["ground_truth"]

    if isinstance(response_lst, str):
        response_lst = [response_lst]

    # 计算每条 response 的得分
    score_lst = [reward_fn(r, ground_truth, method="strict") for r in response_lst]

    # Extract the prediction of each response
    pred_lst = []
    if pred_func_name is not None:
        pred_fn = getattr(module, pred_func_name)
        pred_lst = [pred_fn(r) for r in response_lst]
    
    # 标记每条 response 是否正确 (score >= 0.99)
    is_correct_lst = [score >= 0.99 for score in score_lst]

    return data_source, np.mean(score_lst), is_correct_lst, idx, score_lst, pred_lst


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

    # 获取配置中的路径和函数名，而不是加载函数对象
    reward_file_path = config.custom_reward_function.path
    reward_func_name = config.custom_reward_function.name
    pred_func_name   = config.custom_reward_function.pred_name if config.custom_reward_function.calc_maj is True else None

    # Create remote tasks - 传递路径字符串而非函数对象
    remote_tasks = [
        process_item.remote(
            data_sources.iloc[i], 
            responses.iloc[i], 
            reward_model_data.iloc[i],
            reward_file_path,       # 传递字符串，可序列化
            reward_func_name,       # 传递字符串，可序列化
            pred_func_name,         # ...
            i                       # row idx
        ) for i in range(total)
    ]

    # evaluate test_score based on data source
    # toll the amount of correct responses (data point)
    data_source_reward = defaultdict(list)
    data_source_correct_count = defaultdict(int)  # 统计正确数量

    # Store detailed results for each data point in the correct order
    # is_correct_per_response[i] = [True, False, True, ...]
    is_correct_per_response = [None] * total
    scores_per_response = [None] * total
    preds_per_response = [None] * total

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, mean_score, is_correct_lst, idx, score_lst, pred_lst = ray.get(result_id)
                
                # Toll: Correct Answers
                data_source_reward[data_source].append(mean_score)
                if mean_score >= 0.99:
                    data_source_correct_count[data_source] += 1
                is_correct_per_response[idx] = is_correct_lst
                scores_per_response[idx] = score_lst
                preds_per_response[idx] = pred_lst
                pbar.update(1)

    # Add correctness toll to dataframe
    dataset['is_correct'] = is_correct_per_response
    dataset['reward_scores'] = scores_per_response
    dataset['preds'] = preds_per_response
    dataset['correctness_label'] = [
        ['correct' if x else 'incorrect' for x in row] for row in is_correct_per_response
    ]

    output_path = config.data.output_path
    base_name = os.path.basename(output_path)
    name, ext = os.path.splitext(base_name)
    
    if (ext == '.parquet'):
        dataset.to_parquet(output_path, index=False)
    elif (ext == '.json'):
        dataset.to_json(output_path, index=False)
    else:
        print("WARNING: Unsupported output file extention, fallback to .json")
        dir_name = os.path.dirname(output_path)
        output_path = os.path.join(dir_name, f"{name}.json")
        dataset.to_json(output_path, index=False)


    # Calculate & Output Metrics
    results = compute_all_metrics(dataset, scores_per_response, preds_per_response,
                                  data_source_reward,
                                  is_correct_per_response, data_sources, max_n=None, num_bootstrap=1000, seed=42)

    print("\n" + "="*60)
    print("📊 Evaluation Results")
    print("="*60)
    
    for ds in sorted(data_source_reward.keys()):        
        print(f"\n📁 Data Source: {ds}")

        for metric, value in results[ds].items():
            if 'accuracy' in metric:
                print(f"    {metric}: {value:.2f}%")
            elif 'total' in metric or 'correct' in metric:
                print(f"    {metric}: {value}")
            else:
                print(f"    {metric}: {value:.4f}")
    
    # 总体统计
    if 'overall' in results.keys():
        print(f"\n📊 Overall Statistics:")
        print(f"   ✅ Total Correct: {results['overall']['correct']}/{results['overall']['total']}")
        print(f"   🎯 Overall Accuracy: {results['overall']['accuracy']:.2f}%")

    print("="*60)
    print("Raw Metrics Dictionary:")
    print(results)
    print("="*60)

    return results


if __name__ == "__main__":
    main()

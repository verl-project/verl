import numpy as np
from verl.trainer.ppo.metric_utils import process_validation_metrics
from collections import defaultdict


def compute_best_and_maj_at_n(dataset, scores_per_response, preds_per_response):
    total = len(dataset)
    
    # 准备平铺数据（long format）
    data_sources_list = []
    sample_inputs_list = []
    infos_dict = {'score': [], 'pred': []}

    for idx in range(total):
        row = dataset.iloc[idx]
        scores = scores_per_response[idx]
        preds = preds_per_response[idx]
        
        # 将每个 response 展开为独立条目
        for score, pred in zip(scores, preds):
            data_sources_list.append(row['data_source'])
            # 使用 problem 或 prompt 作为唯一标识，确保同一问题的 responses 被分组
            sample_inputs_list.append(str(row['problem']))  
            infos_dict['score'].append(float(score))
            infos_dict['pred'].append(pred)

    # 调用 metric 计算（包含 bootstrap best@n/maj@n）
    metric_results = process_validation_metrics(
        data_sources=data_sources_list,
        sample_inputs=sample_inputs_list,
        infos_dict=infos_dict,
        seed=42
    )

    # 重构结果为按 data_source 分类的字典（类似 compute_pass_at_n 的返回格式）
    results = {}
    
    for ds, var_dict in metric_results.items():
        results[ds] = {}
        
        for var, metrics in var_dict.items():
            # 只保留 best@n, maj@n 和 mean@n 相关指标
            for metric_name, value in metrics.items():
                if any(x in metric_name for x in ['best@', 'maj@', 'mean@']):
                    # 构造完整的指标名，如：score/best@4/mean 或直接 best@4/mean
                    # key = f"{var}/{metric_name}"
                    results[ds][metric_name] = float(value)
    
    return results


def compute_pass_at_n(is_correct_per_response, data_sources, max_n=None, num_bootstrap=1000, seed=42):
    """
    计算 pass@n: n 次尝试中至少一次正确的比例 (带 bootstrap 估计方差)
    """
    rng = np.random.RandomState(seed)
    
    # 按 data_source 分组
    ds_to_correctness = defaultdict(list)
    for ds, corrects in zip(data_sources, is_correct_per_response):
        ds_to_correctness[ds].append(np.array(corrects, dtype=bool))
    
    results = {}
    
    for ds, all_corrects in ds_to_correctness.items():
        results[ds] = {}

        # 确定该数据源的 n 取值（2 的幂次直到 max_n 或最大 response 数）
        max_available = max(len(c) for c in all_corrects)
        upper = min(max_n, max_available) if max_n else max_available
        
        ns = []
        n = 2
        while n < upper:
            ns.append(n)
            n *= 2
        ns.append(upper)
        
        for n in ns:
            pass_means = []
            for corrects in all_corrects:
                if len(corrects) < n:
                    continue
                # Bootstrap：多次采样 n 个 responses，计算是否至少一个正确
                passes = []
                for _ in range(num_bootstrap):
                    sampled = rng.choice(corrects, size=n, replace=False)
                    passes.append(sampled.any())  # any = 至少一个正确
                pass_means.append(np.mean(passes))
            
            if pass_means:
                results[ds][f"pass@{n}/mean"] = np.mean(pass_means)
                results[ds][f"pass@{n}/std"] = np.std(pass_means)
    
    return results


def compute_acc_avg_per_source(data_source_reward, data_sources, is_correct_per_response):
    # 按 data_source 分组
    ds_to_correctness = defaultdict(list)
    for ds, corrects in zip(data_sources, is_correct_per_response):
        ds_to_correctness[ds].append(np.array(corrects, dtype=bool))
    
    results = {}
    total_correct = 0
    total_samples = 0
    
    for data_source in sorted(data_source_reward.keys()):
        results[data_source] = {}

        rewards = data_source_reward[data_source]
        correctness_arr = np.array(ds_to_correctness[data_source]).flatten()
        bools, bools_count = np.unique(correctness_arr, return_counts=True)
        correct = bools_count[bools.tolist().index(True)] if True in bools else 0
        count = np.sum(bools_count)
        accuracy = correct / count * 100  # 转换为百分比
        
        results[data_source]["test_score"] = np.mean(rewards)
        results[data_source]["accuracy"] = accuracy
        results[data_source]["correct"] = correct
        results[data_source]["total"] = count
        
        total_correct += correct
        total_samples += count
    
    # 总体统计
    if len(data_source_reward) > 1:
        results['overall'] = {}

        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        results['overall']["accuracy"] = overall_accuracy
        results['overall']["correct"] = total_correct
        results['overall']["total"] = total_samples

    return results


def compute_all_metrics(dataset, scores_per_response, preds_per_response,
                        data_source_reward,
                        is_correct_per_response, data_sources, max_n=None, num_bootstrap=1000, seed=42):
    
    results_1 = compute_best_and_maj_at_n(dataset, scores_per_response, preds_per_response)
    results_2 = compute_acc_avg_per_source(data_source_reward, data_sources, is_correct_per_response)
    results_3 = compute_pass_at_n(is_correct_per_response, data_sources, max_n, num_bootstrap, seed)

    all_results = [results_2, results_1, results_3]
    merged = defaultdict(dict)

    for d in all_results:
        for source, metrics in d.items():
            merged[source].update(metrics)

    final_result = dict(merged)
    return final_result

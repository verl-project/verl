import json
import random

def sample_sentences_by_percentage_distribution(sentences_list, percentage_distribution={}, total_samples=1000):
    """
    根据指定的百分比分布从句子列表中抽样句子。
    
    参数:
    sentences_list: 包含句子的列表，每个元素是一个字符串。
    percentage_distribution: 一个字典，键为元组表示长度范围(包含)，值为该范围内需要抽样的句子的百分比。
    total_samples: 总共需要抽样的句子数量。
    
    返回:
    sampled_sentences: 一个列表，包含按照指定百分比分布抽样的句子。
    """
    if not percentage_distribution:
        percentage_distribution = {
        (0, 10): 0.7,   
        (11, 20): 0.15,  
        (21, 30): 0.10,
        (31, 60): 0.05,
    }
    
    # 计算每个长度范围应抽取的句子数量
    length_to_sample = {
        length_range: max(0, int(total_samples * percentage))
        for length_range, percentage in percentage_distribution.items()
    }
    
    # 创建一个字典来保存按长度范围分组的句子
    grouped_sentences = {}
    for sentence in sentences_list:
        sentence_length = len(sentence)
        for (min_len, max_len) in length_to_sample.keys():
            if min_len <= sentence_length <= max_len:
                if (min_len, max_len) not in grouped_sentences:
                    grouped_sentences[(min_len, max_len)] = []
                grouped_sentences[(min_len, max_len)].append(sentence)
                break
    
    # 抽样
    sampled_sentences = []
    for (min_len, max_len), n in length_to_sample.items():
        filtered_sentences = grouped_sentences.get((min_len, max_len), [])
        
        # 检查是否有足够的句子可供抽样
        if len(filtered_sentences) < n:
            print(f"警告：长度在{min_len}到{max_len}之间的句子不足{n}个，只有{len(filtered_sentences)}个。")
            n = len(filtered_sentences)
        
        # 如果有足够多的句子，则进行抽样；否则取全部
        if n > 0:
            sampled = random.sample(filtered_sentences, n)
            sampled_sentences.extend(sampled)
    
    # 如果抽样总数少于要求的总数，打印警告
    if len(sampled_sentences) < total_samples:
        print(f"Warning: Only {len(sampled_sentences)} sentences were sampled.")
    
    # 打乱最终的句子列表以保持随机性
    random.shuffle(sampled_sentences)
    
    return sampled_sentences[:total_samples]

def downsample_dict_list_by_unique_field(data_list, unique_field):
    """
    对输入的字典列表进行降采样，确保指定字段唯一。
    
    参数:
        data_list (list): 包含字典的列表，每个字典中必须包含 unique_field 指定的字段。
        unique_field (str): 需要保证唯一的字段名。
    
    返回:
        list: 降采样后的字典列表，保证指定字段的值只出现一次。
    """
    if not isinstance(data_list, list):
        raise ValueError("输入必须是一个列表")
    if not isinstance(unique_field, str):
        raise ValueError("unique_field 必须是字符串类型")
    
    seen_values = set()  # 用于记录已经处理过的字段值
    result = []          # 存储降采样后的结果
    
    for item in data_list:
        if unique_field not in item:
            raise KeyError(f"字典中缺少 '{unique_field}' 字段")
        
        field_value = item[unique_field]
        if field_value not in seen_values:
            # 如果该字段值尚未处理过，添加到结果并标记为已处理
            result.append(item)
            seen_values.add(field_value)
    
    return result
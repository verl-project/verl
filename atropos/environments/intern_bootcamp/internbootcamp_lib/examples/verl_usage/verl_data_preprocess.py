import os
import json
import random
import pandas as pd
import fire
import subprocess
import json
import argparse

def get_split_from_path(path):
    """
    根据文件路径的祖先目录判断 split 值。
    :param path: 文件的完整路径
    :return: split 值（train/test/other）
    """
    # 逐级向上检查目录，直到根目录
    max_depth = 10
    depth = 0
    while path and depth < max_depth:
        parent_dir = os.path.basename(path)
        if parent_dir == "train":
            return "train"
        elif parent_dir == "test":
            return "test"
        path = os.path.dirname(path)  # 向上一级目录
        depth += 1
    return "test"  # 如果没有找到 train 或 test，则返回 test

def shuffle_and_merge_parquet_files(input_dir, output_file):
    """
    将指定目录中的所有 Parquet 文件 shuffle 并合并成一个文件。

    参数:
        input_dir (str): 包含 Parquet 文件的输入目录路径。
        output_file (str): 输出合并后的 Parquet 文件路径。
    """
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        raise ValueError(f"输入目录不存在: {input_dir}")

    # 获取目录中所有的 Parquet 文件
    parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.parquet')]
    if not parquet_files:
        raise ValueError(f"输入目录中没有找到 Parquet 文件: {input_dir}")

    # 读取所有 Parquet 文件并合并为一个 DataFrame
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # 对合并后的 DataFrame 进行 shuffle
    shuffled_df = combined_df.sample(frac=1, random_state=42)

    # make sure the output directory exists
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    # 将 shuffle 后的数据写入输出文件
    shuffled_df.to_parquet(output_file, engine='pyarrow', index=False)

    print(f"合并后的 Parquet 文件已保存到: {output_file}")

def convert_to_parquet(src_jsonl, tgt_parquet, split, shuffle=True):
    """
    将 JSONL 文件转换为 Parquet 格式，并根据 split 设置额外信息。
    :param src_jsonl: 源 JSONL 文件路径
    :param tgt_parquet: 目标 Parquet 文件路径
    :param split: 数据集划分（train/test/other）
    """    
    # 用于存储转换后的数据
    data_list = []
    
    # 读取 JSONL 文件并逐行处理
    with open(src_jsonl, 'r', encoding='utf-8') as f:
        # Shulffe 数据
        lines = list(f.readlines())
        if shuffle:
            random.shuffle(lines)
        for idx, line in enumerate(lines):
            try:
                # 解析每一行的 JSON 数据
                record = json.loads(line.strip())
                
                # 提取所需字段
                data_source = record.get("data_source", "")
                prompt = record.get("prompt", "")
                ground_truth = record.get("ground_truth", "")
                
                # 构造目标格式的数据结构
                formatted_data = {
                    "data_source": 'bootcamp/' + data_source,
                    "prompt": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
                    },
                    "extra_info": {
                        'split': split,  # 使用传入的 split 值
                        'index': idx
                    }
                }
                
                # 将构造好的数据添加到列表中
                data_list.append(formatted_data)
            
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line {idx + 1}: {e}")
                continue
    
    # 将数据列表转换为 Pandas DataFrame
    df = pd.DataFrame(data_list)
    
    
    
    # 保存为 Parquet 文件
    df.to_parquet(tgt_parquet, index=False)


def _main(src, tgt):
    """递归处理目录或文件"""
    if os.path.isdir(src):
        # 如果是目录，创建对应的目标目录
        os.makedirs(tgt, exist_ok=True)
        
        for sub in os.listdir(src):
            src_path = os.path.join(src, sub)
            tgt_path = os.path.join(tgt, sub)
            _main(src_path, tgt_path)  # 递归调用
    elif src.endswith(".jsonl"):
        # 如果是 .jsonl 文件，添加 verl 前缀并进行转换
        base_name = os.path.basename(src)
        tgt_file_name = f"verl_{base_name}"  # 添加 verl 前缀
        
        # tgt 转为parquet后缀
        if ".jsonl" in tgt_file_name:
            tgt_file_name = tgt_file_name.replace(".jsonl", ".parquet")
        tgt_path = os.path.join(os.path.dirname(tgt), tgt_file_name)
        tmp_tgt = tgt_path + ".tmp"
        
        # 获取当前文件所属的 split
        split = get_split_from_path(src)  # 调用函数获取 split
        
        try:
            convert_to_parquet(src, tmp_tgt, split)  # 传递 split 参数
            subprocess.run(f"mv {tmp_tgt} {tgt_path}", shell=True, check=True)
        except Exception as e:
            print(f"Error processing {src}: {e}")
            subprocess.run(f"rm -f {tmp_tgt}", shell=True, check=True)


def main(src, tgt=None):
    """
    主函数，支持目录或文件作为输入
    :param src: 源文件或目录路径
    :param tgt: 目标文件或目录路径
    """
    if not tgt and os.path.isdir(src):
        tgt = src + '_for_verl'
    
    if not os.path.exists(src):
        raise ValueError(f"Source path does not exist: {src}")

    if os.path.isfile(src) and not src.endswith(".jsonl"):
        raise ValueError(f"Source file is not a .jsonl file: {src}")

    _main(src, tgt)
    
    # merge N shuffle
    shuffle_and_merge_parquet_files(input_dir=os.path.join(tgt, 'train'), output_file=os.path.join(tgt + '_merged', 'bootcamps', 'train.parquet'))
    shuffle_and_merge_parquet_files(input_dir=os.path.join(tgt, 'test'), output_file=os.path.join(tgt + '_merged', 'bootcamps', 'test.parquet'))    
    # shuffle_and_merge_parquet_files(input_dir=os.path.join(tgt, 'verified_bench'), output_file=os.path.join(tgt + '_merged', 'bootcamps', 'verified_bench.parquet'))    

    
if __name__ == '__main__':
    """
    示例用法：
    python examples/verl_usage/verl_preprocess.py --src examples/bootcamp_generator_outputs/2025-03-07-16:48:28
    将 `v2_bootcamp_data` 目录下的所有 .jsonl 文件转换为 verl 格式 .jsonl，并保留目录结构输出到默认输出目录
    输出的 .jsonl 文件会带有 verl 前缀。
    """
    fire.Fire(main)
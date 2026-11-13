import os
import json
import fire
import subprocess

import sys
sys.set_int_max_str_digits(128*1024)

def convert_jsonl(src_jsonl, tgt_jsonl):
    """将单个 .jsonl 文件转换为目标格式"""
    with open(tgt_jsonl, "w", encoding="utf-8") as writer:
        with open(src_jsonl, "r", encoding="utf-8") as reader:
            for line in reader:
                item = json.loads(line)
                new_item = {
                    "message_data": [{"role": "user", "content": item["prompt"]}],
                    "metadata": {
                        "data_source": item["data_source"],  # 必要字段，用于配置文件中将数据源和 judger 对应
                        "ground_truth": item["ground_truth"],
                    }
                }
                writer.write(json.dumps(new_item, ensure_ascii=False) + '\n')


def _main(src, tgt):
    """递归处理目录或文件"""
    if os.path.isdir(src):
        # 如果是目录，创建对应的目标目录
        os.makedirs(tgt, exist_ok=True)
        for sub in os.listdir(src):
            src_path = os.path.join(src, sub)
            tgt_path = os.path.join(tgt, sub)
            _main(src_path, tgt_path)
    elif src.endswith(".jsonl"):
        # 如果是 .jsonl 文件，添加 xpuyu 前缀并进行转换
        base_name = os.path.basename(src)
        tgt_file_name = f"xpuyu_{base_name}"  # 添加 xpuyu 前缀
        tgt_path = os.path.join(os.path.dirname(tgt), tgt_file_name)
        tmp_tgt = tgt_path + ".tmp"
        try:
            convert_jsonl(src, tmp_tgt)
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
        tgt = src + '_for_xpuyu'
    
    if not os.path.exists(src):
        raise ValueError(f"Source path does not exist: {src}")

    if os.path.isfile(src) and not src.endswith(".jsonl"):
        raise ValueError(f"Source file is not a .jsonl file: {src}")

    _main(src, tgt)
    subprocess.run(f"cat {tgt}/train/*.jsonl > {tgt}/merge_train.jsonl", shell=True, check=True)
    subprocess.run(f"shuf {tgt}/merge_train.jsonl -o {tgt}/merge_train.jsonl", shell=True, check=True)
    


if __name__ == '__main__':
    """
    示例用法：
    python examples/xpuyu_usage/xpuyu_preprocess.py --src examples/bootcamp_generator_outputs/2025-03-07-16:48:28
    将 `v2_bootcamp_data` 目录下的所有 .jsonl 文件转换为 xpuyu 格式 .jsonl，并保留目录结构输出到默认输出目录
    输出的 .jsonl 文件会带有 xpuyu 前缀。
    """
    fire.Fire(main)
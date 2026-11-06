import json


def append_json_lines(source_file, target_file):
    """
    将source_file中的所有JSON Lines记录追加到target_file中。
    
    参数：
        source_file: 要追加的JSON Lines文件路径
        target_file: 目标JSON Lines文件路径，新的记录将被追加到这里
    """
    # 使用 'a' 模式打开目标文件以进行追加写入
    with open(target_file, 'a', encoding='utf-8') as outfile:
        # 打开源文件并逐行读取其内容
        with open(source_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                outfile.write(line)
                
def read_jsonl(path, encoding='utf-8'):
    """
    Reads a jsonl file and returns a list of dictionaries.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def add_jsonl(data, path, encoding='utf-8'):
    """
    add a dictionary to a jsonl file.
    """
    with open(path, 'a', encoding=encoding) as f:
        f.write(json.dumps(data, ensure_ascii=False)+'\n')
        
def write_jsonl(data, path, encoding='utf-8'):
    """
    Write a list of dictionaries to a jsonl file.
    """
    with open(path, 'w', encoding=encoding) as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')

def extend_jsonl(datas, path, encoding='utf-8'):
    """
    Extend a jsonl file with a list of dictionaries
    """
    with open(path, 'a', encoding=encoding) as f:
        for line in datas:
            f.write(json.dumps(line, ensure_ascii=False)+'\n')

import os
from itertools import count


def merge_jsonlines(input_directory, output_file):
    id_counter = count(start=1)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 使用 os.walk 遍历目录及其子目录
        for root, dirs, files in os.walk(input_directory):
            for filename in files:
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            try:
                                record = json.loads(line)
                                # 添加新的字段到记录中
                                new_record = {}
                                new_record['id'] = next(id_counter)
                                new_record['source_filename'] = os.path.relpath(file_path, input_directory)
                                for key, value in record.items():
                                    new_record[key] = value
                                # 将更新后的记录写入输出文件
                                outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode line in {filename}")
                            

if __name__ == '__main__':
    directory = 'data_generator_outputs/_Eval/20250114_153504/easy'
    output_file = 'data_generator_outputs/_Eval/20250114_153504/easy_test.jsonl'
    merge_jsonlines(directory, output_file)
import json
import jsonlines
import os

import os
import json
import glob
import re

# 每个puzzle的gen数量
train_sample_number = 1000
test_sample_number = 64

def checkpath(target_dir):
# 检查目录是否存在
    if not os.path.exists(target_dir):
        # 如果目录不存在，则创建它
        try:
            os.makedirs(target_dir)
            print(f"目录 {target_dir} 创建成功。")
        except FileExistsError:
            # 理论上不会触发这个异常，但以防万一，比如多线程环境下同时创建
            print(f"目录 {target_dir} 已存在。")
        except PermissionError:
            print(f"没有权限创建目录 {target_dir}。")
        except Exception as e:
            print(f"创建目录 {target_dir} 时出现错误: {e}")

def process_data_config():
    # 遍历data_config目录下所有符合条件的json文件
    data_dir = 'examples/pipelines/puzzle_configs'

    json_files = os.listdir(data_dir)
    train_data = []
    test_data = []
    json_files.sort(key=lambda x: x.capitalize())
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        if 'test.json' in file_name:
            name_res = file_name.split("_test.json")
            mode = 'test'
        elif 'train.json' in file_name:
            name_res = file_name.split("_train.json")
            mode = 'train'
        else:
            continue
        
        config_file = name_res[0]
        bootcamp_name = name_res[0].replace("_", "")
        bootcamp_cls_name = bootcamp_name[0].upper() + bootcamp_name[1:] if bootcamp_name else ''

        entry_test = {
            "bootcamp_name": bootcamp_name,
            "sample_number": test_sample_number,
            "config_file": config_file,
            "bootcamp_cls_name": f"{bootcamp_cls_name}bootcamp"
        }

        entry_train = {
            "bootcamp_name": bootcamp_name,
            "sample_number": train_sample_number,
            "config_file": config_file,
            "bootcamp_cls_name": f"{bootcamp_cls_name}bootcamp"
        }

        if mode == 'train':
            train_data.append(entry_train)
        elif mode == 'test':
            test_data.append(entry_test)
    
    save_dir = 'examples/pipelines/data_configs'
    # 检查dir
    checkpath(save_dir)
    output_file_train = f'{save_dir}/data_config_train.jsonl'
    with open(output_file_train, 'w', encoding='utf-8') as f_out:
        for entry in train_data:
            f_out.write(json.dumps(entry) + '\n')
    
    output_file_train = 'examples/pipelines/data_configs/data_config_test.jsonl'
    with open(output_file_train, 'w', encoding='utf-8') as f_out:
        for entry in test_data:
            f_out.write(json.dumps(entry) + '\n')


if __name__ == '__main__':
    process_data_config()
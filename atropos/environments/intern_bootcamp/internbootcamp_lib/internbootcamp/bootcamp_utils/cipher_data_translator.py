import os
import random

from .jsonlines import read_jsonl, write_jsonl


def to_ICL_encode(path, final_path):
    datas = read_jsonl(path)
    path = os.path.basename(path)
    output_path = os.path.join(final_path,f'icl_encode_{path}')
    results = []
    for data in datas:
        result = {
            "cipher_name": data['cipher_source'],
            "prompt": "",
            "input": data['decode_text'],
            "extra_args": data['extra_args'],
            "output": "",
            "ground_truth": data['encode_text']
        }
        ICL = []
        for _ in range(random.randint(1, 5)):
            # 使用random随机选择一个data
            random_data = random.choice(datas)
            while data == random_data:
                random_data = random.choice(datas)
            if data['extra_args']:
                ICL.append(f'明文: {random_data["plain"]} 密钥或额外参数: {random_data["extra_args"]} 加密成为密文: {random_data["encode_text"]}\n')
            else:
                ICL.append(f'明文: {random_data["plain"]} 加密成为密文: {random_data["encode_text"]}\n')

        if data['extra_args']:
            ICL.append(f'明文: {data["plain"]} 密钥或额外参数: {data["extra_args"]} 加密成为密文: ? 一步一步完成')
        else:
            ICL.append(f"明文: {data['plain']} 加密成为密文: ? 一步一步完成")
        result["prompt"] += ''.join(ICL)
        result["output"] += data['encode_steps']
        results.append(result)
    write_jsonl(results, output_path)


def to_ICL_decode(path, final_path):
    datas = read_jsonl(path)
    path = os.path.basename(path)
    output_path = os.path.join(final_path,f'icl_decode_{path}')
    results = []
    for data in datas:
        result = {
            "cipher_name": data['cipher_source'],
            "prompt": "",
            "input": data['encode_text'],
            "extra_args": data['extra_args'],
            "output": "",
            "ground_truth": data['decode_text']
        }
        ICL = []
        for _ in range(random.randint(1, 5)):
            # 使用random随机选择一个data
            random_data = random.choice(datas)
            while data == random_data:
                random_data = random.choice(datas)
            if data['extra_args']:
                ICL.append(f'密文: {random_data["encode_text"]} 密钥或额外参数: {random_data["extra_args"]} 解密成为明文: {random_data["plain"]}\n')
            else:
                ICL.append(f'密文: {random_data["encode_text"]} 解密成为明文: {random_data["plain"]}\n')
        if data['extra_args']:
            ICL.append(f'密文: {data["encode_text"]} 密钥或额外参数: {data["extra_args"]} 解密成为明文: ? 一步一步完成')
        else:
            ICL.append(f"密文: {data['encode_text']} 解密成为明文: ? 一步一步完成")
        result["prompt"] += ''.join(ICL)
        result["output"] += data['decode_steps']
        results.append(result)
    write_jsonl(results, output_path)


def to_ICL_with_rule_encode(path, final_path):
    datas = read_jsonl(path)
    path = os.path.basename(path)
    output_path = os.path.join(final_path,f'icl_with_rule_encode_{path}')
    results = []
    for data in datas:
        result = {
            "cipher_name": data['cipher_source'],
            "prompt": f"请根据加密算法对明文进行加密\n{data['encode_rule']}\n",
            "input": data['decode_text'],
            "extra_args": data['extra_args'],
            "output": "",
            "ground_truth": data['encode_text']
        }
        ICL = []
        for _ in range(random.randint(0, 3)):
            # 使用random随机选择一个data
            random_data = random.choice(datas)
            while data == random_data:
                random_data = random.choice(datas)
            if data['extra_args']:
                ICL.append(f'明文: {random_data["plain"]} 密钥或额外参数: {random_data["extra_args"]} 加密成为密文: {random_data["encode_text"]}\n')
            else:
                ICL.append(f'明文: {random_data["plain"]} 加密成为密文: {random_data["encode_text"]}\n')
        if data['extra_args']:
            ICL.append(f'明文: {data["plain"]} 密钥或额外参数: {data["extra_args"]} 加密成为密文: ? 一步一步完成')
        else:
            ICL.append(f"明文: {data['plain']} 加密成为密文: ? 一步一步完成")
        result["prompt"] += ''.join(ICL)
        result["output"] += data['encode_steps']
        results.append(result)
    write_jsonl(results, output_path)


def to_ICL_with_rule_decode(path, final_path):
    datas = read_jsonl(path)
    path = os.path.basename(path)
    output_path = os.path.join(final_path,f'icl_with_rule_decode_{path}')
    results = []
    for data in datas:
        result = {
            "cipher_name": data['cipher_source'],
            "prompt": f"请根据加密算法和解密算法对密码进行解密\n加密算法:\n{data['encode_rule']}\n解密算法:\n{data['decode_rule']}\n",
            "input": data['encode_text'],
            "extra_args": data['extra_args'],
            "output": "",
            "ground_truth": data['decode_text']
        }
        ICL = []
        for _ in range(random.randint(0, 3)):
            # 使用random随机选择一个data
            random_data = random.choice(datas)
            while data == random_data:
                random_data = random.choice(datas)
            if data['extra_args']:
                ICL.append(f'密文: {random_data["encode_text"]} 密钥或额外参数: {random_data["extra_args"]} 解密成为明文: {random_data["plain"]}\n')
            else:
                ICL.append(f'密文: {random_data["encode_text"]} 解密成为明文: {random_data["plain"]}\n')
        if data['extra_args']:
            ICL.append(f'密文: {data["encode_text"]} 密钥或额外参数: {data["extra_args"]} 解密成为明文: ? 一步一步完成')
        else:
            ICL.append(f"密文: {data['encode_text']} 解密成为明文: ? 一步一步完成")
        result["prompt"] += ''.join(ICL)
        result["output"] += data['decode_steps']
        results.append(result)
    write_jsonl(results, output_path)
def process_path(path, final_path):
    to_ICL_encode(path, final_path)
    to_ICL_with_rule_encode(path, final_path)
    to_ICL_decode(path, final_path)
    to_ICL_with_rule_decode(path, final_path)

def translate(origin_data_path,final_data_path):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    # 获取文件列表
    train_list = [os.path.join(f'{origin_data_path}', puzzle) for puzzle in os.listdir(f'{origin_data_path}')]
    
    # 使用上下文管理器确保线程池正确关闭
    with ThreadPoolExecutor(max_workers=5) as executor:  # 设置最大线程数为5
        # 提交所有任务到线程池，并获取future对象
        futures = {executor.submit(process_path, path, final_data_path): path for path in train_list}

        # 遍历完成的任务，并用tqdm创建进度条
        for future in tqdm(as_completed(futures), total=len(train_list), desc='cipher data translating'):
            path = futures[future]
            try:
                # 这里可以处理每个任务的结果，如果需要的话
                result = future.result()
            except Exception as exc:
                print(f'Path {path} generated an exception: {exc}')

   
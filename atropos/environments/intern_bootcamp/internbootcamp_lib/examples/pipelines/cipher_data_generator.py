
import argparse
import time
import shutil
import pandas as pd

from tqdm import tqdm

from internbootcamp.libs.cipher import *
from internbootcamp.bootcamp_utils import sample_sentences_by_percentage_distribution as sspd
from internbootcamp.bootcamp_utils.cipher_data_translator import translate
from internbootcamp.bootcamp_utils.cipher_prompt_enhance import enhance
from internbootcamp.bootcamp_utils.formatted_time import formatted_time

import hashlib
from internbootcamp.bootcamp_utils.random_things import random_word
from internbootcamp.bootcamp_utils import append_json_lines,get_difference_optimized,deduplicate_jsonl_by_field
import json
import random
import os

from tqdm import tqdm
class CipherDataGenerator():
    def __init__(self, env_class:BaseCipherEnvironment, input_list):
        """
        Initialize the Star Battle data generator with a specific grid size and attempts intervals.

        Args:
            size (int): The size of the Star Battle grid.
            attempts_intervals (list of tuples): A list of tuples defining the intervals for attempts.
                Each tuple contains two elements: the lower and upper bound of an interval.
                e.g., [(0, 5), (6, 10)] would create files for puzzles solved in 0-5 and 6-10 attempts.
        """
        super().__init__()
        self.env = env_class()
        self.input_list = input_list
        
    def generate_sample(self,plaintext: str = "", **kwargs):
        hashable_kwargs = json.dumps(kwargs, sort_keys=True)
        combined_str = plaintext + hashable_kwargs
        sample_id = hashlib.sha256(combined_str.encode('utf-8')).hexdigest()
        gen_steps,ciphertext = self.env.generator(plaintext, **kwargs)
        solve_steps,decodetext = self.env.solver(**kwargs)
        self.env.reset()
        
        return {
            "id": sample_id,
            "cipher_source": self.env.cipher_name,
            "plain": plaintext,
            "extra_args": kwargs,
            "encode_text": ciphertext,
            "decode_text": decodetext,
            "encode_rule": self.env.get_encode_rule(),
            "decode_rule": self.env.get_decode_rule(),
            "encode_steps": gen_steps,
            "decode_steps": solve_steps
        }
    
    def write_to_jsonl(self, timetamp):
        
        origin_data = os.path.join(f"examples/bootcamp_generator_outputs/{timetamp}/cipher//origin_data/", f"{self.env.cipher_name}.jsonl")
        origin_tmp_data = os.path.join(f"examples/bootcamp_generator_outputs/{timetamp}/cipher/origin_tmp_data/", f"{self.env.cipher_name}.jsonl")
        
        if not os.path.exists(f"examples/bootcamp_generator_outputs/{timetamp}/cipher/origin_data/"):
            # print(f"WARNING: examples/bootcamp_generator_outputs/{timetamp}/cipher/origin_data/ does not exist. Creating it.")
            os.makedirs(f"examples/bootcamp_generator_outputs/{timetamp}/cipher/origin_data/")
        
        if not os.path.exists(f"examples/bootcamp_generator_outputs/{timetamp}/cipher/origin_tmp_data/"):
            os.makedirs(f"examples/bootcamp_generator_outputs/{timetamp}/cipher/origin_tmp_data/")
        
        with open(origin_tmp_data, "w") as file:
            for sentence in self.input_list:
                puzzles_extra = {
                    'FourSquareCipher': {'str1': random_word(), 'str2': random_word()},
                    'Caesar_Cipher': {'shift': random.randint(1,26)},
                    'Kor_rule8_PortaCipher': {'key': random_word()},
                    'Autokey_Cipher': {'key': random_word()},
                    'Beale_Ciphers': {'codebook': "A big cat danced elegantly, finding giraffes happily in joyful kingdoms. Lions moved near owls, peacefully quizzing rabbits. Suddenly, tigers under vast waterfalls xeroxed yellow zebras."},
                    'Beaufort_Cipher': {'keyword': random_word()},
                    'KEYWORD_Cipher': {'keyword': random_word()},
                    'Rail_Fence_Cipher': {'num_rails': random.randint(2, 5)}                   
                }
                try:
                    if self.env.cipher_name in puzzles_extra:
                        extra_args = puzzles_extra[self.env.cipher_name]
                    else:
                        extra_args = {}
                    sample = self.generate_sample(plaintext=sentence.strip(),**extra_args)
                    if sample:
                        file.write(json.dumps(sample,ensure_ascii=False) + '\n')
                except Exception as e:
                    # print(f'{self.env.cipher_name} 错误：', e)
                    pass
        get_difference_optimized(file1=origin_tmp_data, file2=origin_data, key_field='id', output_file=origin_tmp_data)
        append_json_lines(source_file=origin_tmp_data, target_file=origin_data)

import json

def restructure_jsonl_file(file_path):
    """
    对指定的 JSONL 文件中的每一行进行字段重组：
    将除 'prompt' 和 'input' 字段之外的字段放入新字段 'ground_truth' 中。

    参数:
        file_path (str): JSONL 文件的路径。
    """
    if not isinstance(file_path, str) or not file_path.endswith('.jsonl'):
        raise ValueError("输入必须是一个有效的 .jsonl 文件路径")

    # 读取并处理文件内容
    updated_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())  # 解析每一行的 JSON 对象
                if not isinstance(data, dict):
                    raise ValueError("文件中的某一行不是有效的 JSON 对象")

                # 提取 'prompt' 和 'input' 字段（如果存在）
                result = {}
                if 'prompt' in data:
                    result['prompt'] = data['prompt']

                # 将其他字段放入 'ground_truth'
                ground_truth = {key: value for key, value in data.items() if key not in ('prompt')}
                if ground_truth:
                    result['ground_truth'] = ground_truth
                    result["data_source"] = 'Cipher'
                # 将重组后的 JSON 对象转换为字符串并保存
                updated_lines.append(json.dumps(result, ensure_ascii=False))
            except json.JSONDecodeError as e:
                raise ValueError(f"文件中的某一行无法解析为 JSON: {line.strip()}") from e
    random.shuffle(updated_lines)
    # 将更新后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(updated_lines))


def main_pipeline(cipher_envs, input_set, timestamp):

    for cipher_env in tqdm(cipher_envs, desc='cipher data generating'):
        generator = CipherDataGenerator(env_class=cipher_env,input_list=input_set)
        generator.write_to_jsonl(timestamp)

    final_data_path = fr'examples/bootcamp_generator_outputs/{timestamp}/cipher/final_data/'
    if not os.path.exists(final_data_path):
        os.makedirs(final_data_path)
    else:
        # print(f'Warning: {final_data_path} already exists')
        pass
    translate(fr'examples/bootcamp_generator_outputs/{timestamp}/cipher/origin_tmp_data', final_data_path)
    final_data_enhanced_path = final_data_path +'_enhanced'
    enhance(refined_rules = 'internbootcamp/libs/data/cipher_rules.jsonl',inputDir = final_data_path,outputDir = final_data_enhanced_path)
    from internbootcamp.bootcamp_utils import merge_jsonlines
    merge_output_file=fr"examples/bootcamp_generator_outputs/{timestamp}/cipher.jsonl"
    merge_jsonlines(input_directory=final_data_enhanced_path,output_file=merge_output_file)
    restructure_jsonl_file(merge_output_file)
    try:
        shutil.rmtree(fr'examples/bootcamp_generator_outputs/{timestamp}/cipher')
    except:
        pass
    return final_data_enhanced_path

if __name__ == '__main__':
    # 指定cipher环境列表
    # cipher_env_list为所有cipher环境
    # kor_cipher_env_list仅含kor cipher环境
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--nums', type=int)
    parser.add_argument('--split', type=str)
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--filepath', type=str)
    args = parser.parse_args()

    # 获取传入的参数
    nums = int(args.nums)
    if nums == 0:
        exit()
    split = str(args.split)
    timestamp = str(args.timestamp)
    # df = pd.read_csv(r'internbootcamp/libs/data/train.tsv', sep='\t')
    # df = pd.read_csv(args.filepath, sep='\t')
    sentences = open(args.filepath, 'r', encoding='utf-8').readlines()
    
    cipher_env_list.remove(DoubleCeaserEnvironment)
    cipher_env_list.remove(KorPigpenMasonicCipherEnvironment)
    main_pipeline(cipher_envs=cipher_env_list, input_set = sspd(sentences, percentage_distribution={(2,20):1},total_samples=nums), timestamp=timestamp + f"/{split}")
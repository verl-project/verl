import ast
import re
import json
import distance

from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.cipher import *
from bootcamp_utils import catch_print

cipher_env_dict = {}
for cipher_env in cipher_env_list:
    temp = cipher_env()
    cipher_env_dict[temp.cipher_name] = cipher_env


class Cipherbootcamp(Basebootcamp):
    
    @staticmethod
    def prompt_func(question_ori) -> str:
        """
        Process the input_data and return the processed prompt.
        
        Args:
            question_ori: The question to be processed.
        
        Returns:
            str: The processed prompt.
        """
        instruction_following = """
Let's think step by step and output the final answer with an example markdown formatting:
Final-answer: ```text
BTWTBIGTKTGBGIKHGTBTBEME
```
"""
        prompt = question_ori + '\n' + instruction_following
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        Extract the output from the solution.
        
        Args:
            output: Model output to be processed.
        
        Returns:
            The processed output.
        """
        pattern = pattern = r'```text\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, output)

        if matches:
            # 获取 JSON 字符串
            json_str = matches[-1]
            # print('match?', json_str)
            # print('solution generated? first lines', output[:200])
            # print('solution generated? last lines', output[-200:])
            # 替换单引号为双引号，将元组表示改为列表表示
            json_str = json_str.replace("'", '"').replace("(", "[").replace(")", "]")
            try:
                # 解析 JSON 字符串为 Python 字典
                result_dict = json.loads(json_str) if type(json_str) == dict else json_str
                return result_dict
            except json.JSONDecodeError as e:
                # print(f"JSON 解析错误: {e}")
                return json_str
        else:
            return None
        
    @staticmethod 
    def _verify_correction(solution, identity)->bool:
        
        input_str = identity.pop('input')
        cipher_source = identity.pop('source_filename')
        cipher_name = identity.pop('cipher_name')
        extra_args = identity.pop('extra_args',{})
        
        this_cipher_env = None
        for cipher_env_name,cipher_env in cipher_env_dict.items():
            if cipher_env_name == cipher_name:
                this_cipher_env = cipher_env
                break
        if not this_cipher_env:
            raise ValueError(f"cipher_source {cipher_source} is not supported")
        else:
            this_cipher = this_cipher_env()
            
        # 将solution转为小写
        solution = solution.lower()
        
            
        if 'encode' in cipher_source:
            this_cipher.generator(plaintext=input_str, **extra_args)
            # ground_truth 小写
            ground_truth = str(this_cipher.ciphertext).lower()
            score = 1 - min(distance.levenshtein(solution, ground_truth) / len(ground_truth), 1.0)
        elif 'decode' in cipher_source:
            # if 'ASCII' in cipher_source:
            #     input_str = ast.literal_eval(input_str)
            _,ground_truth = catch_print(this_cipher.decode,text=input_str, **extra_args)
            ground_truth = str(ground_truth).lower()
            score = 1 - min(distance.levenshtein(solution, ground_truth) / len(ground_truth), 1.0)
            
        return score*score


if __name__ == "__main__":


    candidate_str = "`text\nULTXAO2OCLC2W2\n```\n</think>\n\nFinal-answer: ```text\nBBBB\n```"
    is_valid = Cipherbootcamp.verify_score(model_output=candidate_str,identity={
    "id": 28,
    "source_filename": "icl_with_rule_decode_Kor_rule25_SHACipher_cn.jsonl",
    "cipher_name": "Kor_rule25_SHACipher",
    "input": "26a37e1c9c2830e646b1163cfb",
    "extra_args": {},
    "output": "开始解密过程...\n加密的十六进制文本: 26a37e1c9c2830e646b1163cfb\n生成的SHA-256密钥: 73ef2a4edd7a7fbf07fd5f6faf99674dc0c25a025fd74c221f4c35849e5c0fb3\n十六进制转换为字节序列: b'&\\xa3~\\x1c\\x9c(0\\xe6F\\xb1\\x16<\\xfb'\n开始XOR解密...\n解密后的字节序列: b'ULTRAROYALIST'\n最终解密结果: ULTRAROYALIST\n",
    "ground_truth": "ULTRAROYALIST"
  }, short_penalty=False )
    
    print("Is the candidate path valid?", is_valid)

import json
import os
import random
import shutil
import tempfile
import traceback

def walk_dir(dir_path, func):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for file in files:
            func(os.path.join(root, file))

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from internbootcamp.bootcamp.cipher.cipher_default import Cipherbootcamp

def walk_dir_multithreaded(dir_path, func, max_workers=32,**kwargs):
    # 首先统计文件总数
    file_count = sum([len(files) for _, _, files in os.walk(dir_path)])
    
    # 收集所有文件路径
    file_paths = [os.path.join(root, name) for root, _, files in os.walk(dir_path) for name in files]

    # 使用ThreadPoolExecutor来并行处理文件
    with tqdm(total=file_count, desc='cipher data enhancing', unit='file') as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(func, file_path, **kwargs): file_path for file_path in file_paths}
            for future in as_completed(futures):
                try:
                    result = future.result()  # 获取任务结果
                except Exception as exc:
                    print("File processing generated an exception:")
                    traceback.print_exc()
                pbar.update(1)  # 每完成一个任务，更新进度条
    # 注意：func 应该是线程安全的，并且不会引起竞争条件。

def read_jsonl(file_path):
    res = []
    with open(file_path, 'r') as f:
        for line in f:
            res.append(json.loads(line))
    return res

def get_refine_rule(rule:str, cn:bool,refined_rules):
    res = []
    for rule_dict in refined_rules:
        if rule_dict['rule'] == rule:
            res.extend(rule_dict['replace'] if cn else rule_dict['replace_en'])
    try:
        return random.choice(res)
    except:
        return None
        
def prompt_enhance(prompt:str, decode:bool, cn:bool, has_rule:bool,refined_rules):
    if has_rule:
        if '\n明文:' in prompt:
            rule = prompt.split('\n明文:')[0]
        elif '\n密文:' in prompt:
            rule = prompt.split('\n密文:')[0]
        else:
            print('rule not found')
        refine_rule = get_refine_rule(rule,cn=cn,refined_rules=refined_rules)
        if not refine_rule:
            return None
        prompt = prompt.replace(rule, refine_rule)
    rep_dict_cn = {
        '文: ?': random.choice(['文: ?','文是什么？',]),
        '明文': random.choice(['明文','原文','原始信息','初始文本', '非加密信息']),
        '密文': random.choice(['密文','暗文','加密信息','加密文本','暗码','隐文']), 
        '加密成为':random.choice(['加密成为','加密为','编码为', '加密成', ]),
        '解密成为':random.choice(['解密成为','解密为','解码为', '解密成', ]),
        '密钥或额外参数: ': random.choice(['密钥或额外参数: ','密钥: ','额外参数: ']),
        '一步一步完成': random.choice(['','一步一步完成','请一步一步完成, 制定合理的解题计划并严格执行。','请规划流程一步步实施，确保每一步都经过仔细检查，最终达到预期效果。','请细心地依照步骤行动，确保过程中的每个细节都不被忽视，以达成准确无误的目标。','请一步一步完成，确保过程详细严谨，结果正确。','精心完成每一步。',])
    }
    rep_dict_en = {
        '文: ?': random.choice(['文: ?','文 is: ?',]),
        '明文': random.choice(['plain text','original information','clear text']),
        '密文': random.choice(['cipher text','encrypted text','encoded text']),
        '加密成为':random.choice(['encrypt into ','encrypt to ','encode into ','encode to ']),
        '解密成为':random.choice(['decrypt into ','decrypt to ','decode into ','decode to ']),
        '密钥或额外参数: ': random.choice(['secret key or extra parameter: ','secret key: ','extra parameter: ']),
        '一步一步完成': random.choice(['','step by step','Please complete it step by step, formulate a reasonable problem-solving plan, and strictly adhere to it.','Please plan the process and implement it step by step, ensuring that each step is carefully checked to ultimately achieve the desired outcome.','Please proceed carefully according to the steps, ensuring that every detail in the process is not overlooked, to achieve an accurate and error-free goal.',' Carefully complete each step.','Please complete it step by step, ensuring the process is detailed and rigorous, and the result is correct.'])
        
    }
    if cn:
        rep_dict = rep_dict_cn
    else:
        rep_dict = rep_dict_en
    for k, v in rep_dict.items():
        prompt = prompt.replace(k, v)
    
    insert_list_encode = ['','您是一位杰出的密文加密专家，请参考以下案例和信息进行加密操作。',
                        '作为编码器，您的任务是依据给出的案例中的加密算法，将明文加密为密文。',
                        '作为密码学领域的专家，您需要运用您的专业知识，分析案例中的加密算法，并对数据实施加密处理。',
                        '您的任务是使用相应的算法将敏感信息转换为不可读的形式，以保障其传输过程中的安全性。',
                        '运用您的专业技能，将提供的数据通过加密算法转换为安全的密文形式，是您的主要职责。',
                        '凭借您在密码学方面的深厚造诣，您的工作是分析并应用案例中的加密技术，确保信息在传输过程中不会被非法截获。',]
    insert_list_decode = ['','您是一位杰出的密文解密专家，请参考以下案例和信息进行解密操作。',
                        '作为解码大师，您的任务是依据案例中描述的解密算法，将密文还原为原始的明文。',
                        '作为密码学领域的专家，您需要运用您的专业知识，分析案例中的加密算法，并对数据实施解密处理。',
                        '您的任务是使用正确的算法将看似无意义的密文转换回可读的原始信息，确保信息的准确性和完整性。',
                        '您的主要职责是运用您的专业技能，将提供的密文通过恰当的解密算法恢复成最初的数据形式。',
                        '凭借您在密码学方面的深厚造诣，您的工作是分析并应用案例中的加密技术，确保信息在传输过程中不会被非法截获。',
    ]
    insert_list_decode_en = ['','You are an excellent cipher decoder, please refer to the following examples and information to decode the ciphertext.',
                        'As a decoder, your task is to use the encryption algorithm described in the examples to decrypt the ciphertext.',
                        'As a specialist in cryptography, your job is to analyze the encryption algorithm in the examples and implement the decryption process on the data.',
                        'Your task is to convert the seemingly meaningless ciphertext into readable information using the appropriate algorithm, ensuring the accuracy and integrity of the information.',
                        'Your primary responsibility is to use your professional skills to decode the provided ciphertext using the correct algorithm and ensure the accuracy and integrity of the information.',
                        'By your deep knowledge in cryptography, your work is to analyze and apply the encryption techniques in the examples, ensuring the security of information during transmission.',
                        'Please decode the ciphertext according to the examples and the given information.'
    ]
    insert_list_encode_en = ['','You are an excellent cipher encoder, please refer to the following examples and information to encode the plaintext.',
                        'As an encoder, your task is to use the encryption algorithm described in the examples to encrypt the plaintext.',
                        'As a specialist in cryptography, your job is to analyze the encryption algorithm in the examples and implement the encryption process on the data.',
                        'Your task is to convert the plaintext into an unreadable form usingthe appropriate algorithm, ensuring the security of the information during transmission.',
                        'Your primary responsibility is to use your professional skills to encode the provided plaintext using the correct algorithm and ensure the security of information during transmission.',
                        'By your deep knowledge in cryptography, your work is to analyze and apply the encryption techniques in the examples, ensuring the security of information during transmission.',
                        'Please encode the plaintext step by step, ensuring the process is detailed and rigorous, and the result is correct.'
    ]
    if decode:
        insert_prompt = random.choice(insert_list_decode if cn else insert_list_decode_en)
    else:
        insert_prompt = random.choice(insert_list_encode if cn else insert_list_encode_en)
    return Cipherbootcamp.prompt_func(insert_prompt + '\n' + prompt)

def substring_from_keyword(s, keyword):
    # 查找关键词在字符串中的位置
    index = s.find(keyword)
    
    # 如果找到了关键词，并且它不是位于字符串的开始处
    if index != -1 and index != 0:
        # 从关键词的起始位置开始截取子串，包括关键词本身
        return s[index:]
    else:
        # 如果没有找到关键词或者关键词位于字符串开始，则返回原始字符串
        return s

def make_rules_sure(path):
    def get_rule(file_path):
        if 'with_rule' not in file_path:
            return
        with open(file_path, 'r') as f:
            lines = f.readlines()
        line = json.loads(lines[0])
        prompt = line['prompt']
        if '\n明文:' in prompt:
            rule = prompt.split('\n明文:')[0]
        elif '\n密文:' in prompt:
            rule = prompt.split('\n密文:')[0]
        else:
            print('rule not found')
        save_rule(rule)

    def save_rule(rule):
        with open('internbootcamp/libs/data/cipher_rules.jsonl', 'a') as f:
            # 检测rule是否已存在
            if rule in [json.loads(line)['rule'] for line in open('internbootcamp/libs/data/cipher_rules.jsonl', 'r')]:
                # print('rule already exists')
                return
            line = json.dumps({'rule': rule,'replace': [rule,],'replace_en': []}, ensure_ascii=False)
            f.write(line)
            f.write('\n')
    
    def get_rule_en(rule):
        prompt = f"Your task is to rewrite the given text into English. Do not delete the information given in the given text, any information cannot be omitted or deleted.. <Given Text>{rule}</Given Text>. Please only return the text you wrote as synonymous text"
        return call_model(prompt,base_url='http://127.0.0.1:8000/v1',model='qwen2_5_72B_instruct')

    
    def call_model(prompt:str, model:str, base_url:str ,history_list:list[dict] = None,sysprompt:str = None,temperature = 0.001, top_p=0.8, max_tokens=4096, api_key = 'EMPTY'):
            client = OpenAI(base_url=base_url,api_key=api_key)
            messages = []
            if sysprompt:
                messages.append({"role": "system", "content": sysprompt})
            messages.extend(history_list + [{"role": "user", "content": prompt}] if history_list else [{"role": "user", "content": prompt}])
            response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=1,
                        # timeout=120,
                    )
            return response.choices[0].message.content
    
    def translate_rule(rule_path):
        # Read all rules into a list
        with open(rule_path, 'r') as f:
            rules = [json.loads(line) for line in f.readlines()]

        # Process each rule and update the replace_en field if necessary
        for rule in tqdm(rules, desc='Processing rules', unit='rule'):
            if not rule.get('replace_en'):  # Ensure the key exists and is not empty
                rule['replace_en'] = []  # Initialize the list if it doesn't exist
                rule_en = get_rule_en(rule['rule'])
                rule['replace_en'].append(rule_en)

        # Write the updated rules back to the file
        temp_file_path = tempfile.mkstemp()[1]
        try:
            with open(temp_file_path, 'w') as f:
                for rule in rules:
                    line = json.dumps(rule, ensure_ascii=False)
                    f.write(line + '\n')
            # Replace the original file with the updated one
            shutil.move(temp_file_path, rule_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            # Clean up the temporary file in case of an error
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    walk_dir_multithreaded(path,get_rule)
    translate_rule('internbootcamp/libs/data/cipher_rules.jsonl')
    
def main(path,outputDir,refined_rules):
    jsonl = read_jsonl(path)
    decode = True if 'decode' in path else False
    cn = True if 'cn' in path else False
    has_rule = True if 'with_rule' in path else False
    if '_Kor_' in path and not has_rule:
        # kor cipher 跳过 without rule
        return
    res = []
    for line in jsonl:
        enhanced_prompt = prompt_enhance(line['prompt'],decode=decode,cn=cn,has_rule=has_rule,refined_rules=refined_rules)
        if enhanced_prompt:
            line['prompt'] = enhanced_prompt.strip()
            res.append(line)
        else:
            print(f"data {path} enhanced prompt is empty, skip")
            break
        
        # line['refine_output'] = substring_from_keyword(line['refine_output'].strip(), '<restate>')
    if not res:
        return
    path_base = os.path.basename(path)
    with open(os.path.join(outputDir,path_base), 'w') as f:
        for line in res:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def to_cn_en(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # 遍历源目录下的所有文件
    for filename in os.listdir(src_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(src_dir, filename)
            
            # 如果是.py文件，创建cn.py和en.py两个副本
            base_name, ext = os.path.splitext(filename)
            cn_file = f"{base_name}_cn{ext}"
            en_file = f"{base_name}_en{ext}"

            # 定义新的文件路径
            cn_dest_path = os.path.join(dest_dir, cn_file)
            en_dest_path = os.path.join(dest_dir, en_file)

            # 复制文件到目标目录
            shutil.copy2(file_path, cn_dest_path)  # 使用copy2来保留元数据
            shutil.copy2(file_path, en_dest_path)
            
def enhance(refined_rules,inputDir,outputDir):
    make_rules_sure(inputDir)
    to_cn_en(inputDir,inputDir+'_tmp')
    
    if not os.path.exists(outputDir):
        # print(f'Warning: {outputDir} does not exist. Creating...')
        os.makedirs(outputDir)
    walk_dir_multithreaded(inputDir+'_tmp', main ,outputDir = outputDir, refined_rules = read_jsonl(refined_rules))
    try:
        shutil.rmtree(inputDir+'_tmp')
    except:
        pass

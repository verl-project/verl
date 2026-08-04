"""# 谜题训练场开发任务

## 任务概述
你是一位资深程序员，我需要你帮我实现一个特定谜题的训练场环境类。这个类继承自`Basebootcamp`，用于生成谜题实例并验证解答。

## 背景说明
我正在开发一系列谜题训练场，每个训练场对应一个特定类型的谜题。训练场类命名为`{PuzzleName}bootcamp`，其中`PuzzleName`是谜题的名称。

每个训练场类主要提供两个核心功能：
1. 生成该谜题类型的问题实例
2. 验证用户对问题的回答是否正确

## 技术接口规范

### 类方法实现要求

```python
from bootcamp import Basebootcamp

class {PuzzleName}bootcamp(Basebootcamp):
    def __init__(self, **params):
        \"\"\"
        请你自定义params，以保存该puzzle相关的参数，例如网格大小等，参数配有默认值
        \"\"\"
        pass
    
    def case_generator(self):
        \"\"\"
        生成谜题实例，提示：为保证谜题有解，可以先生成结果再对结果处理得到谜题
        返回：一个可JSON序列化的字典（避免包含set等无法通过json.dumps处理的数据结构）
        \"\"\"
        pass
    
    @staticmethod
    def prompt_func(question_case) -> str:
        \"\"\"
        将case_generator生成的谜题实例转换为文本形式的问题，问题中包含问题背景、对谜题规则的介绍、具体要解决的谜题实例、期望最终答案的格式，
        例如：你是xxxx，请你解答yyyy，规则如下：yyyy，最终答案放置在：zzzzz
        注意：请参照提供的谜题描述进行复述，规则应当描述详细，包括任务背景、具体任务操作规则、对题目格式和答案格式的含义介绍等，

        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 格式化的问题字符串
            
        注意:
            1. 需考虑问题的格式，以便后续能正确提取
            2. 问题描述中应包含期望的答案格式说明，以便后续能正确提取，为了避免抽取时匹配出干扰项，请要求模型将答案放在特定标签（如双括号）内，例如[[your answer here]]
        \"\"\"
        pass
    
    @staticmethod
    def extract_output(output):
        \"\"\"
        从LLM的回复中提取符合格式要求的答案，如有多个，请抽取最后一个，避免使用re.search等只抽取第一个结果的方式。
        
        参数:
            output: LLM的完整输出（包含原始问题和回答）
            
        返回:
            提取的答案，若未找到符合格式的答案则返回None
        \"\"\"
        pass
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        \"\"\"
        验证提取的答案是否正确，注意一个问题可以能有多个解，按照谜题规则进行检验，不要直接匹配可能的答案。
        
        参数:
            solution: extract_output提取的答案
            identity: case_generator生成的谜题实例
            
        返回:
            bool: 答案是否正确
        \"\"\"
        pass
```

### 验证评分方法（基类已实现）

```python
@classmethod
def verify_score(cls, model_output, identity:dict, format_score=0.1) -> float:
    \"\"\"
    验证输出结果并评分。
    
    参数:
        model_output: 模型的完整输出
        identity: 谜题实例（由case_generator生成）
        format_score: 答案格式正确时的基础分数
    
    返回:
        float: 评分结果（0-1之间）
    \"\"\"
    score = 0. 
    try:
        extract_solution = cls.extract_output(model_output)
        if extract_solution is None:
            return score
        else:
            score = format_score # 格式正确时的基础分数
        if cls._verify_correction(extract_solution, identity):
            score = 1.  # 答案完全正确时的满分
    except Exception as e:
        # 处理异常情况
        pass
    return score
```

### 使用示例

```python
# 初始化谜题训练场
bootcamp = Puzzlebootcamp()

# 生成谜题实例
case = bootcamp.case_generator()

# 将谜题转换为文本问题
prompt = Puzzlebootcamp.prompt_func(case)

# 获取LLM对问题的解答
response = get_response(prompt, \"LLM\")

# 从完整对话中提取答案
extracted_output = Puzzlebootcamp.extract_output(prompt + response)

# 验证答案并评分
score = Puzzlebootcamp.verify_score(extracted_output, case)
```

## 你的任务
请根据以下谜题描述（谜题描述可能不完整，请先结合你的知识澄清规则），实现一个完整的谜题训练场类：

### 谜题描述
1.The puzzle gives a set of letters and the number and length of words to be spelled, e.g. 2 words: 2 letters, 3 letters, 3 letters.
2.The player has to use the given letters to spell a word of the required length and number of letters.
3.Each letter can be used at most once in a word.Example questions are as follows:

<example 0>
P E A 2 words:3 letter,3 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
T C A 2 words:3 letter,3 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
T R A 7 words: 2 letter,2 letter,2 letter,3 letter,3 letter,3 letter,3 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
N D K I 7 words: 3 letter,3 letter,3 letter,3 letter,3 letter,4 letter,4 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
A E B T 4 words: 4 letter,4 letter,4 letter,4 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
T I E D 5 words: 4 letter,4 letter,4 letter,4 letter,4 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
N E M A 5 words: 4 letter,4 letter,4 letter,4 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
B D E N 5 words: 2 letter,4 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
U N T A 3 words: 4 letter,4 letter,4 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
O B W L 6 words: 3 letter, 3 letter,3 letter,3 letter,4 letter,4 letter.
The answers should be given in order,i.e. If the requirement is for 3 words: 2 letter,3 letter,3 letter then a two letter word is given first followed by two three letter words separated by spaces.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import Counter
from itertools import permutations
from bootcamp import Basebootcamp

class KorPuzzleConnectWordsbootcamp(Basebootcamp):
    def __init__(self, word_pool=None):
        self.word_pool = word_pool or self.default_word_pool()
        
    @staticmethod
    def default_word_pool():
        return {
            2: {'BE', 'NO', 'IN', 'TO', 'UP'},
            3: {'CAT', 'DOG', 'BED', 'PEN', 'CAR', 'BAT', 'RAT', 'MAT', 'PAN'},
            4: {'DESK', 'BALL', 'FISH', 'CAKE', 'ROAD', 'BEAN', 'UNIT'},
            5: {'APPLE', 'TABLE', 'CHAIR'}
        }

    def case_generator(self):
        # 动态生成有效案例（保证有解）
        while True:
            # 随机选择单词数量（1-5）
            num_words = random.randint(1, 3)
            
            # 随机选择单词长度组合
            possible_lengths = [k for k in self.word_pool.keys() 
                               if k * num_words <= 7]  # 控制总字母数
            if not possible_lengths:
                continue
            target_lengths = random.choice([
                random.choices(possible_lengths, k=num_words)
            ])
            
            # 生成有效单词组合
            valid_combination = self.find_valid_combination(target_lengths)
            if valid_combination:
                all_letters = list(''.join(valid_combination))
                return {
                    'letters': sorted(all_letters),
                    'word_lengths': target_lengths,
                    'solution': valid_combination
                }

    def find_valid_combination(self, lengths):
        # 确保所有单词存在并共享字母
        for _ in range(100):  # 防止无限循环
            candidate = []
            used_letters = []
            for length in lengths:
                valid_words = [w for w in self.word_pool.get(length, [])
                              if not set(w).intersection(used_letters)]
                if not valid_words:
                    break
                word = random.choice(valid_words)
                candidate.append(word)
                used_letters.extend(list(word))
            if len(candidate) == len(lengths):
                return candidate
        return None

    @staticmethod
    def prompt_func(question_case) -> str:
        letters = ' '.join(question_case['letters'])
        word_lengths = question_case['word_lengths']
        words_desc = ', '.join([f'{l} letter' for l in word_lengths])
        
        prompt = f"""Given letters: {letters}
Form {len(word_lengths)} words with lengths: {words_desc}

Rules:
1. Use each letter exactly once
2. No letter repetition in words
3. Order matters for word lengths
4. Valid English words only

Answer format: [[space-separated-words]]
Example: [[{" ".join(question_case['solution'])}]]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[([^\]]+)\]\]', output)
        if matches:
            last_match = matches[-1].strip().upper()
            if re.fullmatch(r'([A-Z]+\s)*[A-Z]+', last_match):
                return last_match
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # Basic parsing
            if not solution:
                return False
            answer_words = solution.split()
            expected_words = identity['solution']
            
            # Check word count
            if len(answer_words) != len(expected_words):
                return False
            
            # Verify all conditions
            used_letters = []
            for ans_word, exp_word, exp_len in zip(answer_words, expected_words, identity['word_lengths']):
                # Check length
                if len(ans_word) != exp_len:
                    return False
                # Check validity
                if ans_word.upper() not in cls.default_word_pool().get(exp_len, set()):
                    return False
                # Check letter composition
                if not set(ans_word).issubset(identity['letters']):
                    return False
                # Check duplicates
                if len(set(ans_word)) != len(ans_word):
                    return False
                used_letters.extend(list(ans_word))
            
            # Check total letters match
            return Counter(used_letters) == Counter(identity['letters'])
            
        except Exception as e:
            return False

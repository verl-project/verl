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
1.The game starts with a word and specifies an ending word.
2.Only one letter can be changed at a time, and each intermediate step must be a valid word.
3.Change from the start word to the end word by the fewest steps.
4.The question will give the start and end words, answer the minimum number of steps needed to change from the start word to the end word.Example questions are as follows:

<example 0>
From \"MOM\" to \"DAD\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 0>

<example 1>
From \"TEA\" to \"POT\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 1>

<example 2>
From \"FLY\" to \"CRY\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 2>

<example 3>
From \"WINE\" to \"BARE\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 3>

<example 4>
From \"COLD\" to \"WARM\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 4>

<example 5>
From \"LOST\" to \"HERE\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 5>

<example 6>
From \"SAME\" to \"COST\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 6>

<example 7>
From \"HEAD\" to \"TALE\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 7>

<example 8>
From \"COAL\" to \"COAT\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 8>

<example 9>
From \"POOR\" to \"RICH\".
Output the number in double brackets. For example, if it takes 3 steps from the start word to the end word, present the answer as [[3]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict, deque
import random
import re
# from nltk.corpus import words
from bootcamp import Basebootcamp

class KorPuzzleWordLadderbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.word_length = params.get('word_length', 3)
        self.word_list = self._load_words(self.word_length)
        self.word_set = set(self.word_list)
        self.adjacency = self._build_adjacency()
    
    def _load_words(self, length):
        words_file = "/".join(__file__.split('/')[:-4]) + "/" + "internbootcamp/libs/data/words_alpha_370000.txt"
        with open(words_file, 'r') as f:
            words = f.readlines()
            words = [w.strip() for w in words]
        word_list =  [word.lower() for word in words if word.isalpha() and len(word) == length]
        # try:
            # word_list = [word.lower() for word in words.words() 
            #             if word.isalpha() and len(word) == length]
        # except LookupError:
        #     import nltk
        #     nltk.download('words')
        #     word_list = [word.lower() for word in words.words() 
        #                 if word.isalpha() and len(word) == length]
        return word_list
    
    def _build_adjacency(self):
        adjacency = defaultdict(list)
        for word in self.word_list:
            for i in range(len(word)):
                original_char = word[i]
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == original_char:
                        continue
                    candidate = word[:i] + c + word[i+1:]
                    if candidate in self.word_set:
                        adjacency[word].append(candidate)
        return adjacency
    
    def case_generator(self):
        max_attempts = 100
        for _ in range(max_attempts):
            start = random.choice(self.word_list)
            end = random.choice(self.word_list)
            if start == end:
                continue
            steps = self._bfs(start, end)
            if steps is not None:
                return {
                    'start_word': start,
                    'end_word': end,
                    'correct_steps': steps
                }
        # Fallback to a known example
        return {
            'start_word': 'mom',
            'end_word': 'dad',
            'correct_steps': 3
        }
    
    def _bfs(self, start, end):
        if start not in self.adjacency or end not in self.adjacency:
            return None
        if start == end:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)
        while queue:
            current_word, steps = queue.popleft()
            for neighbor in self.adjacency[current_word]:
                if neighbor == end:
                    return steps + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, steps + 1))
        return None
    
    @staticmethod
    def prompt_func(question_case) -> str:
        start = question_case['start_word'].upper()
        end = question_case['end_word'].upper()
        prompt = (
            f"You are trying to solve a word ladder puzzle. The goal is to transform the start word into the end word by changing one letter at a time, with each intermediate step forming a valid English word. Your task is to determine the minimum number of steps required.\n\n"
            f"Start word: {start}\n"
            f"End word: {end}\n\n"
            "Rules:\n"
            "1. You can change only one letter in each step.\n"
            "2. All intermediate words formed must be valid English words.\n"
            "3. The answer should be the minimum number of steps needed.\n\n"
            "Please provide your answer inside double square brackets. For example, if the answer is 3, your response should be: [[3]]\n"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(\d+)\]\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_steps']

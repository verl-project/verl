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
1.A series of words will be given in the question.
2.These words have one thing in common, usually the same prefix of a suffix or the same combination of letters.
3.This commonality will be given in the question.
4.You need to answer which words or letter combinations are common to each of these words. e.g. If each of these words contains the name of an animal, the answer needs to be the name of the animal contained in each word in turn.Example questions are as follows:

<example 0>
\"feminine kindergarten canine overweight threaten\", these five words have in common is that they all end in a number.
Please output the number they contain.
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 0>

<example 1>
\"swedish vermont statues arthur africa sensation misunderstood\", these seven words have in common is that they all contain Monday to Sunday (MON, TUE, WED, THUR, FRI, SAT, SUN).
Output the abbreviations they contain.
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 1>

<example 2>
\"rotate refits sneaky throne tepees\", these five words have in common is that the middle two letters of the word plus the first and last letters of the word can make a new word.
Output these new words.
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 2>

<example 3>
\"sunflower pineapple butterfly newspaper catfish\", these words are all made up of two nouns words.
Please output the nouns that make up each word.
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 3>

<example 4>
\"Reappear Signings Arraigning Intestines Appeases\", these words have in common that each of the letters in the word appears twice.
Please output the letters which appears twice respectively.
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 4>

<example 5>
\"fig must hind slow smug\", these words can become a weather condition by change one letter.
Please output the weather condition that they can be changed to.
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 5>

<example 6>
\"Boundaries Cancerous Librarian Scorpions Chameleon\", these words are all contain on of the names from the 12 signs of the zodiac.
Please output the zodiac they contain.
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 6>

<example 7>
\"history building numbest\", these words are all made up of two words and the last letter of the first word is the first letter of the second word.
Please write the nouns that make up each word(6 words in total, sperate by space).
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 7>

<example 8>
\"Arrawarra Caraparac Nagubugan Oktahatko\", these words are all cities' name and are palindromes.
Please output the forward reading fields in these words (includes the letter in the middle).
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 8>

<example 9>
\"trance stain chime tuba\", these words can become a country name by change one letter.
Please output the country name that they can be changed to.
Please provide your answer in the same order as the words, and make sure to enclose your answer in double square brackets, like this: [[word1 word2 ...]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class KorPuzzleWordBrainTeasersbootcamp(Basebootcamp):
    def __init__(self, num_words=5, nouns=None):
        super().__init__()
        self.num_words = num_words
        default_nouns = [
            'sun', 'flower', 'pine', 'apple', 'butter', 'fly',
            'news', 'paper', 'cat', 'fish', 'history', 'building',
            'ball', 'room', 'fire', 'place', 'water', 'fall',
            'door', 'knob', 'light', 'house', 'tooth', 'paste',
            'book', 'case', 'cup', 'board', 'air', 'port',
            'rail', 'road', 'sea', 'shell', 'snow', 'ball'
        ]  # 扩展默认名词列表
        self.nouns = nouns if nouns is not None else default_nouns.copy()
        if len(self.nouns) < 4:  # 增强参数校验
            raise ValueError("At least four nouns are required to generate unambiguous compounds")

    def case_generator(self):
        words = []
        components = []
        max_attempts = 10000
        attempts = 0
        
        while len(words) < self.num_words:
            noun1 = random.choice(self.nouns)
            noun2 = random.choice(self.nouns)
            combined = noun1 + noun2
            
            # 唯一性校验
            if combined in [w for w, _ in words]:
                attempts += 1
                continue
                
            # 歧义性校验
            if self._has_ambiguous_decomposition(combined, noun1, noun2):
                attempts += 1
                continue
                
            words.append((combined, (noun1, noun2)))
            attempts = 0  # 重置尝试计数器
            
            if attempts > max_attempts:
                raise RuntimeError(f"Failed to generate {self.num_words} valid compounds after {max_attempts} attempts")

        # 打乱顺序避免模式泄露
        random.shuffle(words)
        return {
            'words': [w[0] for w in words],
            'components': [w[1] for w in words]
        }

    def _has_ambiguous_decomposition(self, word, original1, original2):
        """检查是否存在其他分解方式"""
        for i in range(1, len(word)):
            part1 = word[:i]
            part2 = word[i:]
            if (part1 in self.nouns and part2 in self.nouns and 
                (part1, part2) != (original1, original2) and
                (part2, part1) != (original1, original2)):
                return True
        return False

    @staticmethod
    def prompt_func(question_case):
        words = question_case['words']
        words_str = ' '.join([f'"{w}"' for w in words])
        return (
            f'Analyze these compound words: {words_str}\n\n'
            '**Task Requirements:**\n'
            '1. Each word is formed by combining TWO complete nouns\n'
            '2. Output must preserve the original compounding order\n'
            '3. Separate components with single spaces\n'
            '4. All components must exist as standalone nouns\n\n'
            '**Example:**\n'
            'Input: "sunflower"\n'
            'Valid Output: [[sun flower]]\n'
            'Invalid Output: [[sunflow er]] (er is not a noun)\n\n'
            '**Format:**\n'
            'Place your answer within double square brackets, like:\n'
            '[[noun1 noun2 noun3 noun4...]]'
        )

    @staticmethod
    def extract_output(output):
        # 匹配最后一个有效答案块
        matches = re.findall(r'\[\[([^\]]+)\]\]', output)
        if not matches:
            return None
        last_valid = None
        for m in reversed(matches):
            cleaned = re.sub(r'[^a-zA-Z\s]', '', m).strip()
            if cleaned:
                last_valid = cleaned
                break
        return last_valid.split() if last_valid else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = []
        for pair in identity['components']:
            expected.extend(pair)
        return solution == expected

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
In a simple conjunctive form (simple disjunctive form) containing n propositional variables, if each propositional variable and its negation appear exactly once, and the propositional variables or their negations are arranged in ascending order of subscripts or in lexicographical order, such a simple conjunctive form (simple disjunctive form) is called a paired conjunctive term (paired disjunctive term).
If the true assignment of a paired conjunctive term corresponds to a binary number equal to hexadecimal number i, this paired conjunctive term is denoted as mi (lowercase m). For example, the true assignment of p∧q is 11, and the binary number is 11, corresponding to hexadecimal number 3, denoted as m3.
If the false assignment of a paired disjunctive term corresponds to a binary number equal to hexadecimal number i, this paired disjunctive term is denoted as Mi (uppercase M). For example, the false assignment of ¬p∨¬q∨¬r is 111, and the binary number is 111, corresponding to hexadecimal number 7, denoted as M7.
The disjunctive normal form (conjunctive normal form) consisting of all paired conjunctive terms (paired disjunctive terms) is called the principal disjunctive normal form (principal conjunctive normal form).
Given a formula A containing n propositional variables:
- If the principal disjunctive normal form of A includes all 2^n paired conjunctive terms, A is a tautology.
- If the principal disjunctive normal form of A includes no paired conjunctive terms, A is a contradiction.
- If the principal disjunctive normal form of A includes m0, A is a basic formula.
- If the indices i of the paired conjunctive terms included in the principal disjunctive normal form of A are all even, A is an all-even formula.
- If the indices i of the paired conjunctive terms included in the principal disjunctive normal form of A are all odd, A is an all-odd formula.Example questions are as follows:

<example 0>
According to the above rules, what are the paired conjunctive terms of (¬p^¬q^r)∨(¬p^q^r)? How can this expression be denoted?
The answer should be in the format [[paired conjunctive terms:...]; [denoted:...]], with multiple paired conjunctive terms separated by commas.
</example 0>

<example 1>
According to the rules above, what are the paired disjunctive terms of (p∨¬q∨r)^(¬p∨¬q∨r)? How can this expression be denoted?
The answer should be in the format [[paired disjunctive terms:...];[denoted:...]], with multiple paired conjunctive terms separated by commas.
</example 1>

<example 2>
Identify ¬p∧¬q∧¬r as: (select all that apply)
A. Tautology B. Contradiction C. Basic formula D. All-even formula E. All-odd formula F. None of the above.
The answer format should be like [[AB...]].
</example 2>

<example 3>
Identify (¬p∧¬q∧r)∨ (p∧q∧r) as: (select all that apply)
A. Tautology B. Contradiction C. Basic formula D. All-even formula E. All-odd formula F. None of the above.
The answer format should be like [[AB...]].
</example 3>

<example 4>
Determine whether (¬p∧¬q∧¬r)V(¬p∧¬q∧r)V(¬p∧q∧r)V(p∧¬q∧r)V(p∧q∧r) conforms to the principal disjunctive normal form or principal conjunctive normal form? If yes, how can it be denoted?
The answer format is as follows: if the statement conforms to the main disjunctive or conjunctive normal form, the answer should be formatted as [[A];[denoted expression]]. If it does not conform, the format should be [[B]].
</example 4>

<example 5>
Determine whether (p∨r)∧(¬q∨r)∧(¬p∨q∨¬r) conforms to the principal disjunctive normal form or principal conjunctive normal form? If yes, how can it be denoted?
The answer format is as follows: if the statement conforms to the main disjunctive or conjunctive normal form, the answer should be formatted as [[A];[denoted expression]]. If it does not conform, the format should be [[B]].
</example 5>

<example 6>
Given that formula A contains 4 propositional variables, what should it be denoted as if it is both a tautology and a basic form? The answer format is [[]].
</example 6>

<example 7>
Given that formula A contains 4 propositional variables, how many formulas satisfy the conditions of being both a basic form and an all-even form?
The answer is a single number, in the format [[]].
</example 7>

<example 8>
A research institute needs to select 1-2 members from 3 key researchers A, B, and C to study abroad. Due to work requirements, the selection must satisfy the following conditions:

1. If A goes, then C must go.
2. If B goes, then C cannot go.
3. If C does not go, then either A or B can go.

Let p: A goes
Let q: B goes
Let r: C goes

Based on the given conditions, the formula can be derived as:
(p → r) ∧ (q → ¬r) ∧ (¬r → (p ∨ q))

The true assignments of this formula are the feasible selection schemes. Through derivation, we get:
(p → r) ∧ (q → ¬r) ∧ (¬r → (p ∨ q)) ↔ (¬p ∧ ¬q ∧ r) ∨ (¬p ∧ q ∧ ¬r) ∨ (p ∧ ¬q ∧ r)

The formula (¬p ∧ ¬q ∧ r) ∨ (¬p ∧ q ∧ ¬r) ∨ (p ∧ ¬q ∧ r) is in principal disjunctive normal form and can be denoted as what? This formula belongs to: (multiple choice)
A. Tautology B. Contradiction C. Basic Form D. All-Even Form E. All-Odd Form F. None of the Above

Answer format: [[denoted expression];[options]]
</example 8>

<example 9>
A research institute needs to select 1-2 members from 3 key researchers A, B, and C to study abroad. Due to work requirements, the selection must satisfy the following conditions:

1. If A goes, then C must go.
2. If B goes, then C cannot go.
3. If C does not go, then either A or B can go.

Let p: A goes
Let q: B goes
Let r: C goes

Based on the given conditions, the formula can be derived as:
(p → r) ∧ (q → ¬r) ∧ (¬r → (p ∨ q))

The true assignments of this formula are the feasible selection schemes. Through derivation, we get:
(p → r) ∧ (q → ¬r) ∧ (¬r → (p ∨ q)) ↔ (¬p ∧ ¬q ∧ r) ∨ (¬p ∧ q ∧ ¬r) ∨ (p ∧ ¬q ∧ r), which can be denoted as m1 ∨ m2 ∨ m5. 
Based on the three true assignments represented, what are the feasible selection schemes? 

Only give the letters of the people selected to go, separated by commas within a scheme, and different schemes separated by []; format: [[];[];..].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorLogicDisjunctiveNormalFormAndConjunctiveNormalFormbootcamp(Basebootcamp):
    def __init__(self, max_variables=3, **params):
        super().__init__(**params)
        self.max_variables = max_variables  # p,q,r,s etc.

    def case_generator(self):
        problem_type = random.choice(['classify', 'denote'])
        n = random.randint(2, self.max_variables)
        variables = ['p', 'q', 'r', 's'][:n]
        case = {'variables': variables, 'n': n}

        if problem_type == 'classify':
            # 生成主析取范式案例
            max_terms = 2 ** n
            term_count = random.choice([
                0, max_terms] + [random.randint(1, max_terms-1) for _ in range(3)]
            )
            terms = random.sample(range(max_terms), term_count) if 0 < term_count < max_terms else (
                [] if term_count == 0 else list(range(max_terms)))

            # 确定正确选项
            correct_options = []
            if term_count == max_terms:
                correct_options.append('A')
            elif term_count == 0:
                correct_options.append('B')
            else:
                has_m0 = False
                all_even = True
                all_odd = True
                for i in terms:
                    if i == 0:
                        has_m0 = True
                    if i % 2 != 0:
                        all_even = False
                    if i % 2 == 0:
                        all_odd = False
                if has_m0:
                    correct_options.append('C')
                if all_even and terms:
                    correct_options.append('D')
                if all_odd and terms:
                    correct_options.append('E')
                if not correct_options:
                    correct_options.append('F')

            # 构建表达式
            expr = " ∨ ".join([f"({self._get_conj_term(i, variables)})" for i in terms]) if terms else "∅"

            case.update({
                'type': 'classify',
                'expression': expr,
                'terms': terms,
                'correct_options': correct_options
            })

        elif problem_type == 'denote':
            form_type = random.choice(['disjunctive', 'conjunctive'])
            max_terms = 2 ** n
            
            if form_type == 'disjunctive':
                # 主析取范式
                term_count = random.randint(1, max_terms-1)
                terms = random.sample(range(max_terms), term_count)
                expr = " ∨ ".join([f"({self._get_conj_term(i, variables)})" for i in terms])
                denotation = " ∨ ".join([f"m{i}" for i in sorted(terms)])
            else:
                # 主合取范式
                # 生成假赋值对应的索引
                term_count = random.randint(1, max_terms-1)
                false_assignments = random.sample(range(max_terms), term_count)
                expr_parts = [self._get_disj_term(i, variables) for i in false_assignments]
                expr = " ∧ ".join([f"({part})" for part in expr_parts])
                denotation = " ∧ ".join([f"M{i}" for i in sorted(false_assignments)])
                terms = false_assignments

            case.update({
                'type': 'denote',
                'form_type': form_type,
                'expression': expr,
                'terms': terms,
                'correct_denote': denotation
            })

        return case

    @staticmethod
    def _get_conj_term(i, variables):
        """生成合取式项（主析取范式用）"""
        n = len(variables)
        binary = bin(i)[2:].zfill(n)[::-1]  # 低位在左
        literals = []
        for idx in range(n):
            if binary[idx] == '1':
                literals.append(variables[idx])
            else:
                literals.append(f"¬{variables[idx]}")
        return " ∧ ".join(literals)

    @staticmethod
    def _get_disj_term(i, variables):
        """生成析取式项（主合取范式用）"""
        n = len(variables)
        binary = bin(i)[2:].zfill(n)[::-1]  # 低位在左
        literals = []
        for idx in range(n):
            if binary[idx] == '0':  # 假赋值对应否定
                literals.append(variables[idx])
            else:
                literals.append(f"¬{variables[idx]}")
        return " ∨ ".join(literals)

    @staticmethod
    def prompt_func(question_case):
        if question_case['type'] == 'classify':
            return (
                f"给定逻辑表达式：\n{question_case['expression']}\n"
                "根据主范式分类规则判断属于哪些类别？\n"
                "A. 永真式 B. 矛盾式 C. 基本式 D. 全偶式 E. 全奇式 F. 以上都不是\n"
                "答案格式：[[大写字母组合]] 例如[[AB]]"
            )
        elif question_case['type'] == 'denote':
            return (
                f"将表达式转换为标准记号形式：\n{question_case['expression']}\n"
                "答案格式：[[记号表达式]] 例如[[m1∨m3]]或[[M2∧M5]]"
            )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if identity['type'] == 'classify':
                sol_set = set(solution.upper())
                truth_set = set(identity['correct_options'])
                return sol_set == truth_set
            
            elif identity['type'] == 'denote':
                # 标准化比较
                sol = re.sub(r'\s+', '', solution).lower().split('∨') if '∨' in solution else re.sub(r'\s+', '', solution).lower().split('∧')
                truth = re.sub(r'\s+', '', identity['correct_denote']).lower().split('∨') if '∨' in identity['correct_denote'] else re.sub(r'\s+', '', identity['correct_denote']).lower().split('∧')
                
                # 检查元素集合是否相同
                return set(sol) == set(truth) and len(sol) == len(truth)
            
            return False
        except:
            return False

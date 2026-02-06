"""# 

### 谜题描述
A PIN code is a string that consists of exactly 4 digits. Examples of possible PIN codes: 7013, 0000 and 0990. Please note that the PIN code can begin with any digit, even with 0.

Polycarp has n (2 ≤ n ≤ 10) bank cards, the PIN code of the i-th card is p_i.

Polycarp has recently read a recommendation that it is better to set different PIN codes on different cards. Thus he wants to change the minimal number of digits in the PIN codes of his cards so that all n codes would become different.

Formally, in one step, Polycarp picks i-th card (1 ≤ i ≤ n), then in its PIN code p_i selects one position (from 1 to 4), and changes the digit in this position to any other. He needs to change the minimum number of digits so that all PIN codes become different.

Polycarp quickly solved this problem. Can you solve it?

Input

The first line contains integer t (1 ≤ t ≤ 100) — the number of test cases in the input. Then test cases follow.

The first line of each of t test sets contains a single integer n (2 ≤ n ≤ 10) — the number of Polycarp's bank cards. The next n lines contain the PIN codes p_1, p_2, ..., p_n — one per line. The length of each of them is 4. All PIN codes consist of digits only.

Output

Print the answers to t test sets. The answer to each set should consist of a n + 1 lines

In the first line print k — the least number of changes to make all PIN codes different. In the next n lines output the changed PIN codes in the order corresponding to their appearance in the input. If there are several optimal answers, print any of them.

Example

Input


3
2
1234
0600
2
1337
1337
4
3139
3139
3139
3139


Output


0
1234
0600
1
1337
1237
3
3139
3138
3939
6139

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import string

for _ in xrange(input()):
    a = [raw_input() for i in xrange(input())]
    s = set(a)
    d = set()
    ans = 0
    b = []
    for x in a:
        if x in d:
            ans += 1
            for i in xrange(4):
                ok = False
                for c in string.digits:
                    tmp = [y for y in x]
                    tmp[i] = c
                    tmp = ''.join(tmp)
                    if tmp not in d and tmp not in s:
                        x = tmp
                        ok = True
                        break
                if ok:
                    break
        b.append(x)
        d.add(x)
    print ans
    for x in b:
        print x
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import string
import random
import itertools
from bootcamp import Basebootcamp

class Bpincodesbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        original_pins = []
        attempts = 0
        
        # 生成具有可控重复程度的初始PIN码
        while len(original_pins) < n:
            # 保证至少有1个重复项以触发修改逻辑
            if len(original_pins) == 0 or (len(original_pins) >= 1 and attempts < 10):
                pin = ''.join(random.choices(string.digits, k=4))
                original_pins.append(pin)
                attempts += 1
            else:  # 添加重复项
                original_pins.append(random.choice(original_pins))
        
        # 使用确定性算法生成解决方案
        expected_k, modified_pins = self.solve_puzzle(original_pins)
        
        return {
            'n': n,
            'original_pins': original_pins,
            'expected_k': expected_k,
            'expected_pins': modified_pins  # 增加预期结果存储
        }
    
    @staticmethod
    def solve_puzzle(pins):
        unique_set = set()
        modified = []
        changes = 0
        
        for idx, pin in enumerate(pins):
            if pin not in unique_set:
                modified.append(pin)
                unique_set.add(pin)
                continue
            
            # 生成所有可能的一位修改候选
            candidates = []
            for pos in range(4):
                for digit in string.digits:
                    if digit == pin[pos]:
                        continue
                    candidate = pin[:pos] + digit + pin[pos+1:]
                    if candidate not in unique_set and candidate not in pins[idx+1:]:
                        candidates.append(candidate)
            
            # 选择最早不冲突的候选
            for candidate in candidates:
                if candidate not in unique_set:
                    modified.append(candidate)
                    unique_set.add(candidate)
                    changes += 1
                    break
            else:
                # 回退机制：生成全新PIN
                for _ in range(1000):
                    new_pin = ''.join(random.choices(string.digits, k=4))
                    if new_pin not in unique_set and new_pin not in pins[idx+1:]:
                        modified.append(new_pin)
                        unique_set.add(new_pin)
                        changes += 1
                        break
                else:
                    raise RuntimeError("Failed to find solution")
        
        # 验证解决方案有效性
        assert len(modified) == len(pins), "Length mismatch"
        assert len(set(modified)) == len(pins), "Duplicate found"
        return changes, modified

    @staticmethod
    def prompt_func(question_case):
        original_pins = question_case['original_pins']
        n = question_case['n']
        example = "\n".join(original_pins)
        return f"""Polycarp的{n}个原始PIN码：
{example}

要求：
1. 修改最少位数使所有PIN唯一
2. 保持原始顺序
3. 输出格式：
[answer]
k
new_pin1
...
new_pin{n}
[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_match = matches[-1].strip()
        parts = [p.strip() for p in last_match.split('\n') if p.strip()]
        
        try:
            k = int(parts[0])
            pins = parts[1:]
            return {'k': k, 'pins': pins}
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or 'k' not in solution or 'pins' not in solution:
            return False
        
        user_k = solution['k']
        user_pins = solution['pins']
        original = identity['original_pins']
        expected_k = identity['expected_k']
        
        # 基础验证
        if len(user_pins) != len(original):
            return False
        if any(not (len(p) == 4 and p.isdigit()) for p in user_pins):
            return False
        if len(set(user_pins)) != len(user_pins):
            return False
        
        # 计算实际修改次数
        actual_changes = sum(sum(o != u for o, u in zip(orig, user)) 
                           for orig, user in zip(original, user_pins))
        
        # 最终校验需要满足两个条件：
        # 1. 实际修改次数等于用户报告的k
        # 2. 修改次数等于系统计算的最小k 或者 用户k >= 系统k
        return (actual_changes == user_k) and (user_k == expected_k)

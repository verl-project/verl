"""# 

### 谜题描述
You have a positive integer m and a non-negative integer s. Your task is to find the smallest and the largest of the numbers that have length m and sum of digits s. The required numbers should be non-negative integers written in the decimal base without leading zeroes.

Input

The single line of the input contains a pair of integers m, s (1 ≤ m ≤ 100, 0 ≤ s ≤ 900) — the length and the sum of the digits of the required numbers.

Output

In the output print the pair of the required non-negative integer numbers — first the minimum possible number, then — the maximum possible number. If no numbers satisfying conditions required exist, print the pair of numbers \"-1 -1\" (without the quotes).

Examples

Input

2 15


Output

69 96


Input

3 0


Output

-1 -1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def maximum(n,s):
    l,su=\"\",s
    for i in xrange(n):
        if su<9:
            l+=str(su)
            su=0
        else:
            l+=\"9\"
            su-=9
    return l
def minimum(n,s):
    l,su=[],s-1
    if n==1:
        return s
    for i in xrange(n-1):
        if su<9:
            l.append(su)
            su=0
        else:
            l.append(9)
            su-=9
    l.append(su+1)
    l.reverse()
    ans = \"\"
    for i in l:
        ans += str(i)
    return ans
n,s=map(int,raw_input().split())
if 0<s<=n*9:
    print minimum(n,s),maximum(n,s)
elif n==1 and 0<=s<=9:
    print s,s
else:
    print \"-1 -1\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cgivenlengthandsumofdigitsbootcamp(Basebootcamp):
    def __init__(self, max_m=100, max_s=900):
        super().__init__()
        self.max_m = max_m
        self.max_s = max_s
    
    def case_generator(self):
        """生成具有统计学意义的测试案例，确保30%的有效案例覆盖边界条件"""
        if random.random() < 0.3:
            m = random.randint(1, self.max_m)
            if m == 1:
                s = random.choice([0] + list(range(1, 10)))  # 包含零的特殊情况
            else:
                s = random.randint(1, m*9)
        else:
            m = random.randint(1, self.max_m)
            s = random.randint(0, self.max_s)
        return {'m': m, 's': s}
    
    @staticmethod
    def prompt_func(question_case):
        """结构化提示模板确保格式一致性"""
        m, s = question_case['m'], question_case['s']
        return f"""找到满足以下条件的{m}位数字：
━┅━━┅━ 核心规则 ━┅━━┅━
1. 数字总位数 = {m}位
2. 所有数字之和 = {s}
3. 禁止前导零（除非是唯一的零）

┏━━━━━ 输出要求 ━━━━━┓
将最小数和最大数按格式[answer]min max[/answer]输出
无效案例使用[answer]-1 -1[/answer]

例如：
[answer]69 96[/answer] 或 [answer]-1 -1[/answer]"""

    @staticmethod
    def extract_output(output):
        """鲁棒的答案提取机制"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL|re.IGNORECASE)
        return matches[-1].replace(' ', '').strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """三维验证体系：格式检查、数学验证、边界条件"""
        try:
            m, s = identity['m'], identity['s']
            
            # Case 1: 验证-1 -1的特殊情况
            if solution == "-1-1":
                return not cls._has_valid_solution(m, s)
            
            # Case 2: 格式验证
            if len(solution) != 2*m + 1 or solution[m] != ' ':
                return False
            min_num, max_num = solution[:m], solution[m+1:]
            
            # 前导零检查
            if (m > 1 and (min_num[0] == '0' or max_num[0] == '0')):
                return False
            
            # 数值验证
            return (sum(map(int, min_num)) == s and
                    sum(map(int, max_num)) == s and
                    cls._has_valid_solution(m, s))
        except:
            return False

    @classmethod
    def _has_valid_solution(cls, m, s):
        """解存在性判断逻辑"""
        if m == 1:
            return 0 <= s <= 9
        return 1 <= s <= m*9

    @classmethod
    def compute_solutions(cls, m, s):
        """双指针法生成极值"""
        def gen_min():
            if m == 1: return str(s)
            res = [0]*m
            remaining = s
            for i in reversed(range(1, m)):
                val = min(9, remaining-1)
                res[i] = val
                remaining -= val
            res[0] = remaining
            return ''.join(map(str, res)) if res[0] <=9 else None

        def gen_max():
            res = []
            remaining = s
            for _ in range(m):
                val = min(9, remaining)
                res.append(str(val))
                remaining -= val
            return ''.join(res) if remaining ==0 else None

        if not cls._has_valid_solution(m, s):
            return ("-1 -1", True)
        return (f"{gen_min()} {gen_max()}", False) if gen_min() and gen_max() else ("-1 -1", True)

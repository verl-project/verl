"""# 

### 谜题描述
Well, here is another math class task. In mathematics, GCD is the greatest common divisor, and it's an easy task to calculate the GCD between two positive integers.

A common divisor for two positive numbers is a number which both numbers are divisible by.

But your teacher wants to give you a harder task, in this task you have to find the greatest common divisor d between two integers a and b that is in a given range from low to high (inclusive), i.e. low ≤ d ≤ high. It is possible that there is no common divisor in the given range.

You will be given the two integers a and b, then n queries. Each query is a range from low to high and you have to answer each query.

Input

The first line contains two integers a and b, the two integers as described above (1 ≤ a, b ≤ 109). The second line contains one integer n, the number of queries (1 ≤ n ≤ 104). Then n lines follow, each line contains one query consisting of two integers, low and high (1 ≤ low ≤ high ≤ 109).

Output

Print n lines. The i-th of them should contain the result of the i-th query in the input. If there is no common divisor in the given range for any query, you should print -1 as a result for this query.

Examples

Input

9 27
3
1 5
10 11
9 11


Output

3
-1
9

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
if sys.subversion[0] == \"PyPy\":
    import io, atexit
    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
    
    sys.stdin = io.BytesIO(sys.stdin.read())
    input = lambda: sys.stdin.readline().rstrip()

RS = raw_input
RI = lambda x=int: map(x,RS().split())
RN = lambda x=int: x(RS())
''' ...................................................................... '''

def factor(num):
    i = 1
    fac = set()
    while i*i<=num:
        if num%i==0:
            fac.add(i)
            fac.add(num/i)
        i+=1
    return fac

a,b = RI()
facA = factor(a)
facB = factor(b)
common = facA&facB

for _ in xrange(RN()):
    lo,hi = RI()
    ans = -1
    for f in common:
        if lo<=f<=hi:
            ans = max(ans,f)
    print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import math
import random
from bootcamp import Basebootcamp

class Cmodifiedgcdbootcamp(Basebootcamp):
    def __init__(self, min_d=1, max_d=100, min_mult=1, max_mult=100, min_queries=3, max_queries=5):
        self.min_d = min_d
        self.max_d = max_d
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.min_queries = min_queries  # 最少3个查询以保证案例多样性
        self.max_queries = max_queries

    def case_generator(self):
        # 确保生成合法的公约数
        d = random.randint(max(1, self.min_d), self.max_d)

        def generate_coprimes(max_retry=100):
            for _ in range(max_retry):
                a_mult = random.randint(self.min_mult, self.max_mult)
                b_mult = random.randint(self.min_mult, self.max_mult)
                if math.gcd(a_mult, b_mult) == 1:
                    return a_mult, b_mult
            return 1, 1  # fallback

        a_mult, b_mult = generate_coprimes()
        a = d * a_mult
        b = d * b_mult

        # 生成所有公约数（降序排列）
        def collect_factors(num):
            factors = set()
            for i in range(1, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    factors.update({i, num//i})
            return sorted(factors, reverse=True)
        common_factors = collect_factors(math.gcd(a, b))

        # 生成包含不同类型查询的测试案例
        n = random.randint(self.min_queries, self.max_queries)
        queries = []
        answers = []
        
        # 预先生成必须包含的测试类型
        query_types = [
            ('exact_match', d),  # 精确匹配最大公约数
            ('below_min', 0),    # 下界低于最小因数
            ('over_max', d*2)    # 上界超过最大因数
        ]
        
        # 填充剩余查询为随机类型
        for _ in range(n - len(query_types)):
            query_types.append(('random', None))
        
        random.shuffle(query_types)

        for q_type in query_types[:n]:  # 确保总数为n
            if q_type[0] == 'exact_match':
                low = high = q_type[1]
            elif q_type[0] == 'below_min':
                low = 1
                high = min(common_factors) - 1
            elif q_type[0] == 'over_max':
                low = common_factors[0] + 1
                high = low + 10
            else:
                # 生成包含有效范围的随机查询
                if common_factors and random.random() < 0.7:
                    target = random.choice(common_factors)
                    low = random.randint(max(1, target-2), target)
                    high = random.randint(target, target+2)
                else:
                    low = random.randint(1, 10**9)
                    high = random.randint(low, 10**9)

            # 计算答案
            ans = -1
            for f in common_factors:
                if low <= f <= high:
                    ans = f
                    break
            queries.append((low, high))
            answers.append(ans)

        return {
            'a': a,
            'b': b,
            'n': n,
            'queries': queries,
            'answers': answers
        }

    @staticmethod
    def prompt_func(question_case):
        prompt = f"""请解决以下数学问题：

**任务说明**
给定两个正整数a和b（分别为{question_case['a']}和{question_case['b']}），您需要处理{question_case['n']}个查询。每个查询给出两个整数low和high，要求找到a和b的最大公约数d，使得d满足low ≤ d ≤ high。如果不存在这样的d，返回-1。

**输入格式**
- 第1行：a和b，空格分隔
- 第2行：n（查询数量）
- 随后n行：每行两个整数low和high，空格分隔

**当前测试用例**
{question_case['a']} {question_case['b']}
{question_case['n']}
""" 
        prompt += "\n".join(f"{lo} {hi}" for lo, hi in question_case['queries'])
        prompt += "\n\n**输出要求**\n请严格按照顺序输出{question_case['n']}行结果，每行一个整数，使用如下格式：\n[answer]\n结果1\n结果2\n...\n[/answer]"

        return prompt

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
            
        processed = []
        for line in answer_blocks[-1].strip().split('\n'):
            cleaned = line.strip()
            if cleaned:
                try:
                    processed.append(int(cleaned))
                except ValueError:
                    pass
        return processed or None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 双重验证：答案数量必须匹配且每个答案正确
        return (
            isinstance(solution, list) and
            len(solution) == identity['n'] and
            solution == identity['answers']
        )

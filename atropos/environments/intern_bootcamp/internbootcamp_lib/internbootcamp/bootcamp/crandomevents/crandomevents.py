"""# 

### 谜题描述
Ron is a happy owner of a permutation a of length n.

A permutation of length n is an array consisting of n distinct integers from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a permutation, but [1,2,2] is not a permutation (2 appears twice in the array) and [1,3,4] is also not a permutation (n=3 but there is 4 in the array).

<image>

Ron's permutation is subjected to m experiments of the following type: (r_i, p_i). This means that elements in range [1, r_i] (in other words, the prefix of length r_i) have to be sorted in ascending order with the probability of p_i. All experiments are performed in the same order in which they are specified in the input data.

As an example, let's take a look at a permutation [4, 2, 1, 5, 3] and an experiment (3, 0.6). After such an experiment with the probability of 60\% the permutation will assume the form [1, 2, 4, 5, 3] and with a 40\% probability it will remain unchanged.

You have to determine the probability of the permutation becoming completely sorted in ascending order after m experiments.

Input

Each test contains one or more test cases. The first line contains the number of test cases t (1 ≤ t ≤ 100).

The first line of each test case contains two integers n and m (1 ≤ n, m ≤ 10^5) — the length of the permutation and the number of experiments, respectively.

The second line of each test case contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ n) — contents of the permutation.

The following m lines of each test case each contain an integer r_i and a real number p_i (1 ≤ r_i ≤ n, 0 ≤ p_i ≤ 1) — the length of the prefix and the probability of it being sorted. All probabilities are given with at most 6 decimal places.

It is guaranteed that the sum of n and the sum of m does not exceed 10^5 (∑ n, ∑ m ≤ 10^5).

Output

For each test case, print a single number — the probability that after all experiments the permutation becomes sorted in ascending order. Your answer will be considered correct if its absolute or relative error does not exceed 10^{-6}.

Formally, let your answer be a, and the jury's answer be b. Your answer is accepted if and only if \frac{|a - b|}{max{(1, |b|)}} ≤ 10^{-6}.

Example

Input


4
4 3
4 3 2 1
1 0.3
3 1
4 0.6
5 3
4 2 1 3 5
3 0.8
4 0.6
5 0.3
6 5
1 3 2 4 5 6
4 0.9
5 0.3
2 0.4
6 0.7
3 0.5
4 2
1 2 3 4
2 0.5
4 0.1


Output


0.600000
0.720000
0.989500
1.000000

Note

Explanation of the first test case: It can be demonstrated that whether the final permutation is sorted or not depends solely on sorting being performed in the (4, 0.6) experiment.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from functools import reduce
from operator import mul

rints = lambda: [int(x) for x in stdin.readline().split()]
rfloats = lambda: [float(x) for x in stdin.readline().split()]
rfloat_2d = lambda n: [rfloats() for _ in range(n)]
out = []

for _ in range(int(input())):
    n, m = rints()
    a, qur, mem = rints(), rfloat_2d(m), [0] * n
    mi, ma, all = n + 1, 0, []

    if sorted(a) == a:
        out.append('1')
        continue

    for i in range(n - 1, -1, -1):
        mem[i] = 1
        if a[i] != i + 1:
            break

    for r, p in qur:
        if mem[int(r) - 1]:
            all.append(1 - p)

    try:
        out.append('%.6f' % (1 - reduce(mul, all)))
    except:
        out.append('0.000000')

print('\n'.join(out))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from functools import reduce
from operator import mul
from bootcamp import Basebootcamp

class Crandomeventsbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=5, **params):
        self.max_n = min(max_n, 100000)
        self.max_m = min(max_m, 100000)
        self.params = params

    def case_generator(self):
        # 控制案例类型分布
        case_type = random.choice(['sorted', 'unsorted', 'boundary'])
        
        # 生成基本参数
        n = random.randint(1, self.max_n)
        m = random.randint(0, self.max_m)
        
        # 生成排列逻辑
        if case_type == 'sorted':
            a = list(range(1, n+1))
        else:
            sorted_arr = list(range(1, n+1))
            # 生成最大正确后缀长度
            k = 0
            if n > 0:
                k = random.randint(0, n)
            a = sorted_arr.copy()
            if k < n:
                # 搅乱前 n-k 个元素
                prefix = a[:n-k]
                random.shuffle(prefix)
                a = prefix + a[n-k:]
        
        # 实际检查排序状态
        is_sorted = a == sorted(a)
        
        # 生成实验数据
        experiments = []
        applicable_probs = []
        
        # 计算实际有效后缀长度
        last_wrong = n
        for i in reversed(range(n)):
            if a[i] != i+1:
                last_wrong = i
                break
        
        for _ in range(m):
            # 智能生成有效的r值
            if random.random() < 0.7 and last_wrong < n:
                r = random.randint(last_wrong+1, n)
            else:
                r = random.randint(1, n)
            p = round(random.uniform(0, 1), 6)
            experiments.append((r, p))
            
            # 判断该实验是否可能影响最终结果
            if r > last_wrong:
                applicable_probs.append(1 - p)

        # 计算正确概率
        if m == 0:
            prob = 1.0 if is_sorted else 0.0
        else:
            if is_sorted:
                prob = 1.0
            else:
                try:
                    total_prob = 1.0 - reduce(mul, applicable_probs, 1.0)
                except:
                    total_prob = 0.0
                # 四舍五入处理
                prob = round(total_prob, 6)
                prob = max(0.0, min(1.0, prob))

        return {
            'n': n,
            'm': m,
            'a': a,
            'experiments': experiments,
            'correct_answer': prob
        }

    @staticmethod
    def prompt_func(question_case):
        exp_list = "\n".join(
            f"{r} {p:.6f}" 
            for r, p in question_case['experiments']
        )
        return f"""## Permutation Probability Problem

Given a permutation of {question_case['n']} numbers: {' '.join(map(str, question_case['a']))}
After applying {question_case['m']} experiments in order:

{exp_list}

Calculate the final probability that the permutation becomes fully sorted.

Output Requirements:
1. Answer must contain exactly 6 decimal places
2. Format as [answer]<result>[/answer]
3. Use standard decimal notation (no scientific notation)

Example:
[answer]0.123456[/answer]"""

    @staticmethod
    def extract_output(output):
        # 支持多格式匹配
        patterns = [
            r'\[answer\]\s*(\d+\.\d{6})\s*\[/answer\]',  # 标准格式
            r'answer\s*:\s*(\d+\.\d{6})',               # 无标签格式
            r'\\boxed{(\d+\.\d{6})}'                    # LaTeX格式
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                try:
                    return round(float(matches[-1]), 6)
                except:
                    continue
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        expected = identity['correct_answer']
        
        # 统一使用题目允许的误差标准
        if expected == 0.0:
            return solution < 1e-6  # 允许接近0的值
        elif expected == 1.0:
            return (1.0 - solution) < 1e-6
        
        abs_error = abs(solution - expected)
        rel_error = abs_error / max(1e-6, abs(expected))
        return abs_error < 1e-6 or rel_error < 1e-6

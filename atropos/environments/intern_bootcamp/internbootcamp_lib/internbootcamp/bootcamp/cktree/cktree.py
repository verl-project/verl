"""# 

### 谜题描述
Quite recently a creative student Lesha had a lecture on trees. After the lecture Lesha was inspired and came up with the tree of his own which he called a k-tree.

A k-tree is an infinite rooted tree where:

  * each vertex has exactly k children; 
  * each edge has some weight; 
  * if we look at the edges that goes from some vertex to its children (exactly k edges), then their weights will equal 1, 2, 3, ..., k. 



The picture below shows a part of a 3-tree.

<image>

As soon as Dima, a good friend of Lesha, found out about the tree, he immediately wondered: \"How many paths of total weight n (the sum of all weights of the edges in the path) are there, starting from the root of a k-tree and also containing at least one edge of weight at least d?\".

Help Dima find an answer to his question. As the number of ways can be rather large, print it modulo 1000000007 (109 + 7). 

Input

A single line contains three space-separated integers: n, k and d (1 ≤ n, k ≤ 100; 1 ≤ d ≤ k).

Output

Print a single integer — the answer to the problem modulo 1000000007 (109 + 7). 

Examples

Input

3 3 2


Output

3


Input

3 3 3


Output

1


Input

4 3 2


Output

6


Input

4 5 2


Output

7

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
MOD = 1000000007

cache = {}

def ways(curr_weight, target_weight, k, d, d_count):
    if curr_weight == target_weight and d_count > 0:
        return 1
    elif curr_weight > target_weight:
        return 0
    elif (curr_weight, d_count) not in cache:
        ans = 0
        for w in xrange(1, k+1):
            ans = (ans + ways(curr_weight+w, target_weight, k, d, d_count + int(w >= d))) % MOD
        cache[(curr_weight, d_count)] = ans
    return cache[(curr_weight, d_count)]

n, k, d = tuple(map(int, raw_input().split()))
print ways(0, n, k, d, 0)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Cktreebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100, min_k=1, max_k=100):
        self.min_n = min_n
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k

    def case_generator(self):
        # 确保生成的k >=1 且 <=100，d在合法范围内
        k = random.randint(max(1, self.min_k), min(100, self.max_k))
        d = random.randint(1, k)
        # 确保n不超过k的理论可能范围
        max_feasible_n = min(self.max_n, k * 10)  # 合理限制最大n
        n = random.randint(max(1, self.min_n), max_feasible_n)
        return {'n': n, 'k': k, 'd': d}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        d = question_case['d']
        prompt = f"""在k-tree中，每个节点有k个子节点，子节点边的权重分别为1到{k}。请计算从根出发总权重为{n}且至少包含一条权重≥{d}的路径数目（模10^9+7）。
        
输入参数：n={n}, k={k}, d={d}
答案格式：[answer]整数[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 增强正则匹配鲁棒性
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) % MOD if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n, k, d = identity['n'], identity['k'], identity['d']
            correct = cls.calculate_answer(n, k, d)
            # 统一取模比较（处理负数和大数情况）
            return int(solution) % MOD == correct
        except:
            return False
    
    @staticmethod
    def calculate_answer(n, k, d):
        # 使用动态规划优化空间复杂度
        dp_total = [0] * (n + 1)
        dp_total[0] = 1
        for i in range(n + 1):
            for j in range(1, k + 1):
                if i + j <= n:
                    dp_total[i + j] = (dp_total[i + j] + dp_total[i]) % MOD

        if d == 1:
            return dp_total[n] % MOD

        dp_no = [0] * (n + 1)
        dp_no[0] = 1
        for i in range(n + 1):
            for j in range(1, d):
                if i + j <= n:
                    dp_no[i + j] = (dp_no[i + j] + dp_no[i]) % MOD

        return (dp_total[n] - dp_no[n]) % MOD

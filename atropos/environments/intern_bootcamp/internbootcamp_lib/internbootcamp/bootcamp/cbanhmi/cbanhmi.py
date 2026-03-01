"""# 

### 谜题描述
JATC loves Banh-mi (a Vietnamese food). His affection for Banh-mi is so much that he always has it for breakfast. This morning, as usual, he buys a Banh-mi and decides to enjoy it in a special way.

First, he splits the Banh-mi into n parts, places them on a row and numbers them from 1 through n. For each part i, he defines the deliciousness of the part as x_i ∈ \{0, 1\}. JATC's going to eat those parts one by one. At each step, he chooses arbitrary remaining part and eats it. Suppose that part is the i-th part then his enjoyment of the Banh-mi will increase by x_i and the deliciousness of all the remaining parts will also increase by x_i. The initial enjoyment of JATC is equal to 0.

For example, suppose the deliciousness of 3 parts are [0, 1, 0]. If JATC eats the second part then his enjoyment will become 1 and the deliciousness of remaining parts will become [1, \\_, 1]. Next, if he eats the first part then his enjoyment will become 2 and the remaining parts will become [\\_, \\_, 2]. After eating the last part, JATC's enjoyment will become 4.

However, JATC doesn't want to eat all the parts but to save some for later. He gives you q queries, each of them consisting of two integers l_i and r_i. For each query, you have to let him know what is the maximum enjoyment he can get if he eats all the parts with indices in the range [l_i, r_i] in some order.

All the queries are independent of each other. Since the answer to the query could be very large, print it modulo 10^9+7.

Input

The first line contains two integers n and q (1 ≤ n, q ≤ 100 000).

The second line contains a string of n characters, each character is either '0' or '1'. The i-th character defines the deliciousness of the i-th part.

Each of the following q lines contains two integers l_i and r_i (1 ≤ l_i ≤ r_i ≤ n) — the segment of the corresponding query.

Output

Print q lines, where i-th of them contains a single integer — the answer to the i-th query modulo 10^9 + 7.

Examples

Input

4 2
1011
1 4
3 4


Output

14
3


Input

3 2
111
1 2
3 3


Output

3
1

Note

In the first example: 

  * For query 1: One of the best ways for JATC to eats those parts is in this order: 1, 4, 3, 2. 
  * For query 2: Both 3, 4 and 4, 3 ordering give the same answer. 



In the second example, any order of eating parts leads to the same answer.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, q = map(int, raw_input().split())
l = raw_input()
cnt1, cnt0 = [0]*(n+1), [0]*(n+1)
mod = 10**9 + 7
for i in range(len(l)):
	if l[i] == '1':
		cnt1[i+1] = cnt1[i] + 1
		cnt0[i+1] = cnt0[i]
	else:
		cnt0[i+1] = cnt0[i] + 1
		cnt1[i+1] = cnt1[i]
pow2 = [1]
for i in range(1, 10**5 + 10):
	pow2.append((2*pow2[-1])%mod)
for i in range(q):
	l, r = map(int, raw_input().split())
	ones = cnt1[r] - cnt1[l-1]
	zeroes = cnt0[r] - cnt0[l-1]
	t1 = (pow2[ones] - 1)%mod
	t2 = (((pow2[ones] - 1)%mod)*(pow2[zeroes] - 1))%mod
	print((t1+t2)%mod)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Cbanhmibootcamp(Basebootcamp):
    def __init__(self, n=None, q=2, min_n=4, max_n=10):
        """
        参数说明：
        n    - 固定字符串长度（可选）
        q    - 查询数量（默认2）
        min_n - 最小随机长度（当n未指定时生效）
        max_n - 最大随机长度（当n未指定时生效）
        """
        if n is not None:
            self.n = n
        else:
            self.n = random.randint(min_n, max_n)
        self.q = q
    
    def case_generator(self):
        """生成具有边界覆盖的测试案例"""
        # 生成字符串（20%概率全0/全1）
        if random.random() < 0.2:
            s = '0' * self.n if random.choice([True, False]) else '1' * self.n
        else:
            s = ''.join(random.choices(['0','1'], weights=[7,3], k=self.n))
        
        # 预处理前缀和
        cnt1 = [0]*(self.n+1)
        cnt0 = [0]*(self.n+1)
        for i in range(1, self.n+1):
            cnt1[i] = cnt1[i-1] + (1 if s[i-1] == '1' else 0)
            cnt0[i] = cnt0[i-1] + (1 if s[i-1] == '0' else 0)
        
        # 生成多样化查询
        queries = []
        answers = []
        for _ in range(self.q):
            # 30%概率生成全区间查询
            if random.random() < 0.3:
                l, r = 1, self.n
            else:
                l = random.randint(1, self.n)
                r = random.randint(l, self.n)
            
            ones = cnt1[r] - cnt1[l-1]
            zeros = cnt0[r] - cnt0[l-1]
            
            # 动态计算答案
            t1 = (pow(2, ones, MOD) - 1) % MOD
            t2 = ((pow(2, ones, MOD)-1) * (pow(2, zeros, MOD)-1)) % MOD
            ans = (t1 + t2) % MOD
            
            queries.append({'l': l, 'r': r})
            answers.append(ans)
        
        return {
            'n': self.n,
            'q': self.q,
            's': s,
            'queries': queries,
            'answers': answers
        }
    
    @staticmethod
    def prompt_func(case) -> str:
        """生成结构化问题描述"""
        prompt = [
            f"Banh-mi问题求解（n={case['n']}, q={case['q']}）\n\n",
            "字符串：", case['s'], "\n\n",
            "查询列表（格式：l r）：\n"
        ]
        for q in case['queries']:
            prompt.append(f"{q['l']} {q['r']}\n")
        prompt.append("\n请逐行输出每个查询的结果，用[answer]包裹，如：\n[answer]答案[/answer]")
        return ''.join(prompt)
    
    @staticmethod
    def extract_output(output):
        """鲁棒性答案提取"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return [int(m) for m in matches] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """容错验证逻辑"""
        try:
            expected = identity['answers']
            return solution[-len(expected):] == expected
        except:
            return False

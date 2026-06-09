"""# 

### 谜题描述
Zookeeper is buying a carton of fruit to feed his pet wabbit. The fruits are a sequence of apples and oranges, which is represented by a binary string s_1s_2… s_n of length n. 1 represents an apple and 0 represents an orange.

Since wabbit is allergic to eating oranges, Zookeeper would like to find the longest contiguous sequence of apples. Let f(l,r) be the longest contiguous sequence of apples in the substring s_{l}s_{l+1}… s_{r}. 

Help Zookeeper find ∑_{l=1}^{n} ∑_{r=l}^{n} f(l,r), or the sum of f across all substrings.

Input

The first line contains a single integer n (1 ≤ n ≤ 5 ⋅ 10^5).

The next line contains a binary string s of length n (s_i ∈ \{0,1\}) 

Output

Print a single integer: ∑_{l=1}^{n} ∑_{r=l}^{n} f(l,r). 

Examples

Input


4
0110


Output


12


Input


7
1101001


Output


30


Input


12
011100011100


Output


156

Note

In the first test, there are ten substrings. The list of them (we let [l,r] be the substring s_l s_{l+1} … s_r):

  * [1,1]: 0 
  * [1,2]: 01 
  * [1,3]: 011 
  * [1,4]: 0110 
  * [2,2]: 1 
  * [2,3]: 11 
  * [2,4]: 110 
  * [3,3]: 1 
  * [3,4]: 10 
  * [4,4]: 0 



The lengths of the longest contiguous sequence of ones in each of these ten substrings are 0,1,2,2,1,2,2,1,1,0 respectively. Hence, the answer is 0+1+2+2+1+2+2+1+1+0 = 12.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
def main():
    n = int(stdin.readline())
    s = stdin.readline().strip()
    r = [0] * n
    g = [[] for _ in xrange(n + 2)]
    a = [0] * n
    i = n - 1
    while i >= 0:
        if s[i] == '1':
            j = i
            while j >= 0 and s[j] == '1':
                r[j] = i + 1
                j -= 1
            while i > j:
                x = i - j
                if g[x+1]:
                    a[i] = x * (g[x+1][-1] - i) + a[g[x+1][-1]]
                else:
                    a[i] = x * (n - i)
                g[x].append(i)
                i -= 1
        else:
            i -= 1
    ans = c = 0
    for i, t in enumerate(r):
        c += 1
        if s[i] == '0':
            continue
        b = t - i
        if i == 0 or s[i-1] == '0':
            for j in xrange(1, b + 1):
                g[j].pop()
        u = b * (b + 1) / 2
        if g[b+1]:
            x = g[b+1][-1]
            u += (x - t) * b + a[x]
        else:
            u += (n - t) * b
        ans += c * u
        c = 0
    print ans
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_answer(n, s):
    if '1' not in s:  # 快速返回全0情况
        return 0
    
    r = [0] * n
    g = [[] for _ in range(n + 2)]
    a = [0] * n
    i = n - 1
    while i >= 0:
        if s[i] == '1':
            j = i
            while j >= 0 and s[j] == '1':
                r[j] = i + 1
                j -= 1
            while i > j:
                x = i - j
                if g[x+1]:
                    a[i] = x * (g[x+1][-1] - i) + a[g[x+1][-1]]
                else:
                    a[i] = x * (n - i)
                g[x].append(i)
                i -= 1
        else:
            i -= 1
    ans = 0
    c = 0
    for i in range(n):
        c += 1
        if s[i] == '0':
            continue
        t = r[i]
        b = t - i
        if i == 0 or s[i-1] == '0':
            for j in range(1, b+1):
                if g[j]:
                    g[j].pop()
        u = b * (b + 1) // 2
        if g[b+1]:
            x = g[b+1][-1]
            u += (x - t) * b + a[x]
        else:
            u += (n - t) * b
        ans += c * u
        c = 0
    return ans

class Ffruitsequencesbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=12, p_1=0.5, edge_case_prob=0.2):
        self.n_min = max(1, n_min)
        self.n_max = max(self.n_min, n_max)
        self.p_1 = max(0.0, min(1.0, p_1))
        self.edge_case_prob = max(0.0, min(1.0, edge_case_prob))

    def case_generator(self):
        # 生成策略优化
        if random.random() < self.edge_case_prob:
            n = random.randint(self.n_min, self.n_max)
            choice = random.choice(['all_zero', 'all_one', 'alternating'])
            if choice == 'all_zero':
                s = '0' * n
            elif choice == 'all_one':
                s = '1' * n
            else:  # 010101...模式
                s = ''.join(['01'[i%2] for i in range(n)])
        else:
            n = random.randint(self.n_min, self.n_max)
            s = ''.join(random.choices(['0','1'], 
                         weights=[1-self.p_1, self.p_1], k=n))
        
        return {
            'n': n,
            's': s,
            'correct_answer': compute_answer(n, s)
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['s']
        return f"""根据以下条件计算所有子串的最长连续1长度之和：

输入格式：
n = {n}
s = {s}

计算要求：
1. 遍历所有可能的子串s[l..r] (1 ≤ l ≤ r ≤ n)
2. 对每个子串找出最长连续的1的个数
3. 求所有子串的对应数值之和

请将最终答案用[answer]标签包裹，例如：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强匹配模式
        matches = re.findall(
            r'\[answer\][\s\n]*(\d+)[\s\n]*\[/answer\]', 
            output, 
            re.IGNORECASE
        )
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 严格类型检查
        if not isinstance(solution, int):
            return False
        return solution == identity['correct_answer']

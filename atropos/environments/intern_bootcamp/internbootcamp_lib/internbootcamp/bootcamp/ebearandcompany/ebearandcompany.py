"""# 

### 谜题描述
Bear Limak prepares problems for a programming competition. Of course, it would be unprofessional to mention the sponsor name in the statement. Limak takes it seriously and he is going to change some words. To make it still possible to read, he will try to modify each word as little as possible.

Limak has a string s that consists of uppercase English letters. In one move he can swap two adjacent letters of the string. For example, he can transform a string \"ABBC\" into \"BABC\" or \"ABCB\" in one move.

Limak wants to obtain a string without a substring \"VK\" (i.e. there should be no letter 'V' immediately followed by letter 'K'). It can be easily proved that it's possible for any initial string s.

What is the minimum possible number of moves Limak can do?

Input

The first line of the input contains an integer n (1 ≤ n ≤ 75) — the length of the string.

The second line contains a string s, consisting of uppercase English letters. The length of the string is equal to n.

Output

Print one integer, denoting the minimum possible number of moves Limak can do, in order to obtain a string without a substring \"VK\".

Examples

Input

4
VKVK


Output

3


Input

5
BVVKV


Output

2


Input

7
VVKEVKK


Output

3


Input

20
VKVKVVVKVOVKVQKKKVVK


Output

8


Input

5
LIMAK


Output

0

Note

In the first sample, the initial string is \"VKVK\". The minimum possible number of moves is 3. One optimal sequence of moves is:

  1. Swap two last letters. The string becomes \"VKKV\".
  2. Swap first two letters. The string becomes \"KVKV\".
  3. Swap the second and the third letter. The string becomes \"KKVV\". Indeed, this string doesn't have a substring \"VK\".



In the second sample, there are two optimal sequences of moves. One is \"BVVKV\" → \"VBVKV\" → \"VVBKV\". The other is \"BVVKV\" → \"BVKVV\" → \"BKVVV\".

In the fifth sample, no swaps are necessary.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
a, b, c = [], [], []
s = raw_input()
from collections import defaultdict
dp = defaultdict(lambda : float(\"inf\"))
def count(a, st, x):
    ret = 0
    i = st
    while i < len(a) and a[i] < x:
        ret += 1
        i += 1
    return ret
for i in xrange(len(s)):
    if s[i] == \"V\":
        a.append(i)
    elif s[i] == \"K\":
        b.append(i)
    else :
        c.append(i)
dp[(0,0,0,0)] = 0
for i in xrange(len(a)+1):
    for j in xrange(len(b)+1):
        for k in xrange(len(c)+1):
            for p in xrange(2):
                if i < len(a):
                    dp[(i+1,j,k,1)] = min(dp[(i+1,j,k,1)], dp[(i,j,k,p)] + count(a,i,a[i])+count(b,j,a[i])+count(c,k,a[i]))
                if j < len(b) and p == 0:
                    dp[(i,j+1,k,0)] = min(dp[(i,j+1,k,0)], dp[(i,j,k,p)]+count(a,i,b[j])+count(b,j,b[j])+count(c,k,b[j]))
                if k < len(c):
                    dp[(i,j,k+1,0)]=min(dp[(i,j,k+1,0)],dp[(i,j,k,p)]+count(a,i,c[k])+count(b,j,c[k])+count(c,k,c[k]))
print min(dp[(len(a),len(b),len(c),0)],dp[(len(a),len(b),len(c),1)])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import string
from collections import defaultdict
import re

def compute_min_swaps(n, s):
    a, b, c = [], [], []
    for i in range(n):
        char = s[i]
        if char == 'V':
            a.append(i)
        elif char == 'K':
            b.append(i)
        else:
            c.append(i)
    
    def count(arr, st, x):
        ret = 0
        i = st
        while i < len(arr) and arr[i] < x:
            ret += 1
            i += 1
        return ret
    
    dp = defaultdict(lambda: float('inf'))
    dp[(0, 0, 0, 0)] = 0
    
    for i in range(len(a)+1):
        for j in range(len(b)+1):
            for k in range(len(c)+1):
                for p in range(2):
                    current_key = (i, j, k, p)
                    current_val = dp[current_key]
                    if current_val == float('inf'):
                        continue
                    
                    # Place V
                    if i < len(a):
                        cost = count(a, i, a[i]) + count(b, j, a[i]) + count(c, k, a[i])
                        new_key = (i+1, j, k, 1)
                        dp[new_key] = min(dp[new_key], current_val + cost)
                    
                    # Place K (only if previous was not V)
                    if j < len(b) and p == 0:
                        cost = count(a, i, b[j]) + count(b, j, b[j]) + count(c, k, b[j])
                        new_key = (i, j+1, k, 0)
                        dp[new_key] = min(dp[new_key], current_val + cost)
                    
                    # Place other characters
                    if k < len(c):
                        cost = count(a, i, c[k]) + count(b, j, c[k]) + count(c, k, c[k])
                        new_key = (i, j, k+1, 0)
                        dp[new_key] = min(dp[new_key], current_val + cost)
    
    return min(dp[(len(a), len(b), len(c), 0)], dp[(len(a), len(b), len(c), 1)])

class Ebearandcompanybootcamp(Basebootcamp):
    def __init__(self, min_length=1, max_length=75):
        self.min_length = min_length
        self.max_length = max_length
    
    def case_generator(self):
        # 保证至少有一定概率生成包含VK的情况
        n = random.randint(self.min_length, self.max_length)
        # 提高生成包含V和K的概率
        population = list(string.ascii_uppercase) + ['V', 'K']*3
        s = ''.join(random.choices(population, k=n))
        correct_answer = compute_min_swaps(n, s)
        return {
            'n': n,
            's': s,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""Bear Limak需要修改字符串以避免出现"VK"子串。每次只能交换相邻字符，求最少交换次数。

输入格式：
第一行：整数n（1 ≤ n ≤ 75）
第二行：由大写字母组成的字符串

当前问题：
{question_case['n']}
{question_case['s']}

请计算最小交换次数，并将整数答案包裹在[answer]和[/answer]标签中。例如：[answer]3[/answer]

注意：
1. 必须读取两行标准输入
2. 最终答案必须是整数形式"""

    @staticmethod
    def extract_output(output):
        # 加强提取的鲁棒性，允许数字前后的空格
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

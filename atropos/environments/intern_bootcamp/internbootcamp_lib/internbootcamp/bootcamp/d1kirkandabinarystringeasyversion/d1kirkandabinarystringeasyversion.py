"""# 

### 谜题描述
The only difference between easy and hard versions is the length of the string. You can hack this problem only if you solve both problems.

Kirk has a binary string s (a string which consists of zeroes and ones) of length n and he is asking you to find a binary string t of the same length which satisfies the following conditions:

  * For any l and r (1 ≤ l ≤ r ≤ n) the length of the longest non-decreasing subsequence of the substring s_{l}s_{l+1} … s_{r} is equal to the length of the longest non-decreasing subsequence of the substring t_{l}t_{l+1} … t_{r};
  * The number of zeroes in t is the maximum possible.



A non-decreasing subsequence of a string p is a sequence of indices i_1, i_2, …, i_k such that i_1 < i_2 < … < i_k and p_{i_1} ≤ p_{i_2} ≤ … ≤ p_{i_k}. The length of the subsequence is k.

If there are multiple substrings which satisfy the conditions, output any.

Input

The first line contains a binary string of length not more than 2\: 000.

Output

Output a binary string which satisfied the above conditions. If there are many such strings, output any of them.

Examples

Input


110


Output


010


Input


010


Output


010


Input


0001111


Output


0000000


Input


0111001100111011101000


Output


0011001100001011101000

Note

In the first example: 

  * For the substrings of the length 1 the length of the longest non-decreasing subsequnce is 1; 
  * For l = 1, r = 2 the longest non-decreasing subsequnce of the substring s_{1}s_{2} is 11 and the longest non-decreasing subsequnce of the substring t_{1}t_{2} is 01; 
  * For l = 1, r = 3 the longest non-decreasing subsequnce of the substring s_{1}s_{3} is 11 and the longest non-decreasing subsequnce of the substring t_{1}t_{3} is 00; 
  * For l = 2, r = 3 the longest non-decreasing subsequnce of the substring s_{2}s_{3} is 1 and the longest non-decreasing subsequnce of the substring t_{2}t_{3} is 1; 



The second example is similar to the first one.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function

def main():
    s = list(map(int, input()))
    n = len(s)
    t = [0] * n

    st = []
    for i in range(n):
        if s[i] == 1:
            st.append(i)
        else:
            if st:
                v = st.pop()
                t[v] = 1

    print(*t, sep='')


INF = float('inf')
MOD = 10**9 + 7

import os, sys
from atexit import register
from io import BytesIO
import itertools

if sys.version_info[0] < 3:
    input = raw_input
    range = xrange

    filter = itertools.ifilter
    map = itertools.imap
    zip = itertools.izip

if \"LOCAL_\" in os.environ:
    debug_print = print
else:
    sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
    sys.stdout = BytesIO()
    register(lambda: os.write(1, sys.stdout.getvalue()))

    input = lambda: sys.stdin.readline().rstrip('\r\n')
    debug_print = lambda *x, **y: None


def input_as_list():
    return list(map(int, input().split()))

def array_of(f, *dim):
    return [array_of(f, *dim[1:]) for _ in range(dim[0])] if dim else f()

main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class D1kirkandabinarystringeasyversionbootcamp(Basebootcamp):
    def __init__(self, min_length=3, max_length=2000):
        """
        初始化训练场参数，设置生成字符串的最小和最大长度。
        添加特殊场景生成概率参数，增强测试覆盖率
        """
        self.min_length = min_length
        self.max_length = max_length
        self.special_case_prob = 0.3  # 30%概率生成特殊场景案例
    
    def case_generator(self):
        """
        生成二进制字符串s，包含随机、全0、全1、交替模式等多种场景
        根据算法生成对应的正确解t
        """
        n = random.randint(self.min_length, self.max_length)
        
        # 30%概率生成特殊模式字符串
        if random.random() < self.special_case_prob:
            pattern = random.choice(['all_zero', 'all_one', 'alternate'])
            if pattern == 'all_zero':
                s = '0' * n
            elif pattern == 'all_one':
                s = '1' * n
            else:  # alternate模式
                s = ('01' * (n//2 + 1))[:n]
        else:
            s = ''.join(random.choices(['0', '1'], k=n))
        
        t = self.compute_t(s)
        return {"s": s, "t": t}
    
    @staticmethod
    def compute_t(s_str):
        """
        根据参考算法计算给定二进制字符串s的正确解t。
        添加类型转换确保处理字节型数据
        """
        s = list(map(int, s_str))
        n = len(s)
        t = [0] * n
        stack = []
        for i in range(n):
            if s[i] == 1:
                stack.append(i)
            else:
                if stack:
                    idx = stack.pop()
                    t[idx] = 1
        return ''.join(map(str, t))
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        增强问题描述的规则说明，补充边界条件说明
        """
        s = question_case['s']
        prompt = f"""你是一个二进制字符串处理专家，请解决以下问题：

给定一个二进制字符串s，构造一个新的二进制字符串t，满足下列条件：
1. 对于所有可能的子区间[l, r] (1 ≤ l ≤ r ≤ n)，s的子串s[l..r]的最长非递减子序列(LNDS)长度必须严格等于t对应子串t[l..r]的LNDS长度
2. 在满足条件1的前提下，t中0的数量要尽可能多

附加规则说明：
- 子序列不需要连续，但元素必须保持原始顺序
- 当s全为0时，t必须等于s（此时已是最优解）
- 当s全为1时，t可以是全0（观察示例验证此情况）
- 对于形如"01"交替的字符串，需要确保每个1的位置被正确处理

输入s为：{s}

你的输出必须是长度与s相同（{len(s)}位）的二进制字符串。请将最终答案放在[answer]和[/answer]标签之间，例如：[answer]010[/answer]。答案必须完全匹配正则表达式：^[01]+$"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        强化提取逻辑，增加格式校验和非法字符过滤
        """
        # 匹配最后一个有效答案块
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        
        # 提取并清理答案
        raw_answer = matches[-1].strip().replace('\n', '').replace(' ', '')
        
        # 过滤非法字符
        filtered = re.sub(r'[^01]', '', raw_answer)
        return filtered if len(filtered) == len(raw_answer) and filtered else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        增加solution格式校验，确保与s等长
        """
        if not solution or len(solution) != len(identity['s']):
            return False
        if not re.match('^[01]+$', solution):
            return False
        return solution == identity['t']

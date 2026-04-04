"""# 

### 谜题描述
The Travelling Salesman spends a lot of time travelling so he tends to get bored. To pass time, he likes to perform operations on numbers. One such operation is to take a positive integer x and reduce it to the number of bits set to 1 in the binary representation of x. For example for number 13 it's true that 1310 = 11012, so it has 3 bits set and 13 will be reduced to 3 in one operation.

He calls a number special if the minimum number of operations to reduce it to 1 is k.

He wants to find out how many special numbers exist which are not greater than n. Please help the Travelling Salesman, as he is about to reach his destination!

Since the answer can be large, output it modulo 109 + 7.

Input

The first line contains integer n (1 ≤ n < 21000).

The second line contains integer k (0 ≤ k ≤ 1000).

Note that n is given in its binary representation without any leading zeros.

Output

Output a single integer — the number of special numbers not greater than n, modulo 109 + 7.

Examples

Input

110
2


Output

3


Input

111111011
2


Output

169

Note

In the first sample, the three special numbers are 3, 5 and 6. They get reduced to 2 in one operation (since there are two set bits in each of 3, 5 and 6) and then to 1 in one more operation (since there is only one set bit in 2).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# -*- coding: utf-8 -*-
MOD = int(1e+9 + 7)


def inv(val): return pow(val, MOD - 2, MOD)


def factorial(n):
    try:
        return factorial.cache[n]
    except IndexError:
        last = len(factorial.cache)
        while last <= n:
            factorial.cache.append(factorial.cache[-1] * last % MOD)
            last += 1
        return factorial.cache[n]
factorial.cache = [1]


def choose(n, k):
    return factorial(n) * inv(factorial(k) * factorial(n-k)) % MOD


def numbers_with_bits(length, n):
    nxt = 0
    res = 0
    while nxt != -1 and length > 0:
        top = len(n) - 1 - nxt

        if(length <= top):
            res += choose(top, length)
            res %= MOD

        nxt = n.find('1', nxt + 1)
        length -= 1

    if length == 0:
        res += 1

    return res


def bits(n): return bin(n).count('1')


def solve(n, k):
    if k == 0:
        return 1
    elif k == 1:
        return len(n) - 1  # exclude the 1
    elif k > 5:
        return 0

    ans = 0
    f = [-1, 0]
    for i in xrange(2, 1001):
        f.append(f[bits(i)] + 1)
        if f[i] == k - 1:
            ans += numbers_with_bits(i, n)
            ans %= MOD
    return ans


#for i in xrange(20, 30):
    #for j in xrange(100, 150):
        #assert j * (Residue(i) / j) == i
        #k = mod_div(i, j)
        #assert 0 < k < MOD
        #assert j * k % MOD == i
#assert numbers_with_bits(1, '10') == 2
#assert numbers_with_bits(2, '101') == 2
#assert numbers_with_bits(3, '1100') == 2
#assert numbers_with_bits(1, '1') == 1
#try:
print solve(n=raw_input(), k=int(raw_input()))
#except Exception as e:
    #print \"ERROR:\", e
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ctravellingsalesmanandspecialnumbersbootcamp(Basebootcamp):
    _MOD = 10**9 + 7
    _factorial_cache = [1]  # 类级阶乘缓存

    def __init__(self, max_n_length=1000, **kwargs):
        super().__init__(**kwargs)
        self.max_n_length = max_n_length

    def case_generator(self):
        # 生成有效的二进制数字（保证多样性）
        length = random.randint(1, self.max_n_length)
        if random.random() < 0.2:  # 20%概率生成边界案例
            if random.choice([True, False]):
                n = '1' + '0' * (length-1)  # 最小有效值
            else:
                n = '1' * length          # 最大有效值
        else:
            n = ['1'] + [random.choice('01') for _ in range(length-1)]
            n = ''.join(n).lstrip('0') or '0'
            if not n: n = '0'
        
        k = random.randint(0, 1000)
        return {'n': n if n != '0' else '1', 'k': k}

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""## 数字变换谜题

**规则说明**：
1. 操作定义：将数字转换为二进制，统计其中1的个数作为新数字
2. 特殊数字：将数字用最少的操作次数变为1，需要正好k次操作
3. 输入约束：n为二进制字符串（1 ≤ n < 2^1000），k为整数（0 ≤ k ≤ 1000）
4. 输出要求：返回符合条件的数字个数（模{10**9+7}）

**输入样例**：
n = 110
k = 2
→ 输出：[answer]3[/answer]

**当前问题**：
n = {question_case['n']}
k = {question_case['k']}

**答案格式**：
将最终结果放在[answer]标签内，例如：[answer]0[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches: return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n_str = identity['n']
            k = identity['k']
            return solution == cls._solve(n_str, k)
        except:
            return False

    # 核心算法实现
    @classmethod
    def _solve(cls, n, k):
        # 特殊值处理
        if k == 0: return 1 if n == '1' else 0
        if k == 1: return max(0, len(n)-1)
        if k > 1000: return 0
        
        dp = [0] * 1002
        dp[1] = 0
        
        total = 0
        for x in range(2, 1001):
            bits = bin(x).count('1')
            dp[x] = dp[bits] + 1
            if dp[x] == k-1:
                total = (total + cls._count_comb(x, n)) % cls._MOD
        return total

    @classmethod
    def _count_comb(cls, target_bits, n_str):
        res = 0
        pos = 0
        remaining = target_bits
        n_len = len(n_str)
        
        while remaining > 0:
            next_1 = n_str.find('1', pos)
            if next_1 == -1: break
            
            available = n_len - next_1 - 1
            if remaining-1 <= available:
                res += cls._comb(available, remaining-1)
                res %= cls._MOD
            
            remaining -= 1
            pos = next_1 + 1
        
        if remaining == 0:
            res = (res + 1) % cls._MOD  # 包含等于原数的情况
        return res

    @classmethod
    def _comb(cls, n, k):
        if k < 0 or k > n: return 0
        return cls._factorial(n) * pow(cls._factorial(k)*cls._factorial(n-k), cls._MOD-2, cls._MOD) % cls._MOD

    @classmethod
    def _factorial(cls, n):
        while len(cls._factorial_cache) <= n:
            cls._factorial_cache.append(cls._factorial_cache[-1] * len(cls._factorial_cache) % cls._MOD)
        return cls._factorial_cache[n]

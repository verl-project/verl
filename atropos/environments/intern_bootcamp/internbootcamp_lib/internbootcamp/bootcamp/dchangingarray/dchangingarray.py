"""# 

### 谜题描述
At a break Vanya came to the class and saw an array of n k-bit integers a_1, a_2, …, a_n on the board. An integer x is called a k-bit integer if 0 ≤ x ≤ 2^k - 1. 

Of course, Vanya was not able to resist and started changing the numbers written on the board. To ensure that no one will note anything, Vanya allowed himself to make only one type of changes: choose an index of the array i (1 ≤ i ≤ n) and replace the number a_i with the number \overline{a_i}. We define \overline{x} for a k-bit integer x as the k-bit integer such that all its k bits differ from the corresponding bits of x. 

Vanya does not like the number 0. Therefore, he likes such segments [l, r] (1 ≤ l ≤ r ≤ n) such that a_l ⊕ a_{l+1} ⊕ … ⊕ a_r ≠ 0, where ⊕ denotes the [bitwise XOR operation](https://en.wikipedia.org/wiki/Bitwise_operation#XOR). Determine the maximum number of segments he likes Vanya can get applying zero or more operations described above.

Input

The first line of the input contains two integers n and k (1 ≤ n ≤ 200 000, 1 ≤ k ≤ 30).

The next line contains n integers a_1, a_2, …, a_n (0 ≤ a_i ≤ 2^k - 1), separated by spaces — the array of k-bit integers.

Output

Print one integer — the maximum possible number of segments with XOR not equal to 0 that can be obtained by making several (possibly 0) operations described in the statement.

Examples

Input

3 2
1 3 0


Output

5

Input

6 3
1 4 4 7 3 4


Output

19

Note

In the first example if Vasya does not perform any operations, he gets an array that has 5 segments that Vanya likes. If he performs the operation with i = 2, he gets an array [1, 0, 0], because \overline{3} = 0 when k = 2. This array has 3 segments that Vanya likes. Also, to get an array with 5 segments that Vanya likes, he can perform two operations with i = 3 and with i = 2. He then gets an array [1, 0, 3]. It can be proven that he can't obtain 6 or more segments that he likes.

In the second example, to get 19 segments that Vanya likes, he can perform 4 operations with i = 3, i = 4, i = 5, i = 6 and get an array [1, 4, 3, 0, 4, 3].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict as ddic, deque, Counter
from itertools import combinations, permutations, product
import bisect, heapq
YES, NO = \"YES\", \"NO\"

rr = raw_input
rrm = lambda: map(int, rr().split())

def solve(N, K, A):
    def flip(x):
        return (1<<K) - 1 - x
    
    P = [0]
    for x in A:
        P.append(P[-1] ^ x)

    count = Counter(P)
    tot = N*(N+1)/2
    for k in count:
        v = count[k]
        k2 = flip(k)
        if k > k2 and k2 in count: continue
        
        v2 = count.get(k2, 0)
        
        if k != k2:
            vs = v + v2
        else:
            vs = v

        w1 = vs / 2
        w2 = (vs+1) / 2
        tot -= w1 * (w1 - 1) / 2
        tot -= w2 * (w2 - 1) / 2
        
    return tot            

\"\"\"
def brute(N, K, A):
    def flip(x):
        return (1<<K) - 1 - x
    def xor(A):
        ans = 0
        for x in A:
            ans ^= x
        return ans

    ans = 0
    for cand in product(*[[x, flip(x)] for x in A]):
        bns = 0
        for i in xrange(len(cand)):
            for j in xrange(i+1,len(cand)+1):
                z = xor(cand[i:j])
                if z != 0:
                    bns += 1
        ans = max(ans, bns)
        
    return ans
\"\"\"

N, K = rrm()
print solve(N, K, rrm())
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import Counter
from bootcamp import Basebootcamp

class Dchangingarraybootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, k_min=1, k_max=5):
        self.n_min = n_min  # 允许生成n=1的边界情况
        self.n_max = n_max
        self.k_min = k_min
        self.k_max = k_max

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        k = random.randint(self.k_min, self.k_max)
        max_val = (1 << k) - 1
        
        # 允许生成全零数组
        a = [random.randint(0, max_val) for _ in range(n)]
        return {'n': n, 'k': k, 'a': a}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        a = ' '.join(map(str, question_case['a']))
        return f"""## 异或子数组最大值问题

**问题描述**
给定一个包含{n}个{k}位整数的数组，每个元素可以被替换为它的补码（所有二进制位取反）。请找出通过任意次数的操作后，可以得到的最多非零异或连续子数组的数量。

**输入格式**
第一行：n k
第二行：a_1 a_2 ... a_n

**当前测试案例**
{n} {k}
{a}

**输出要求**
将最终答案放在[answer]和[/answer]标签之间，例如：[answer]42[/answer]

**注意**
1. 补码定义：k位整数的补码是所有位取反后的结果
2. 子数组的异或值为所有元素按位异或的结果
3. 需要最大化满足异或值非零的子数组数量"""

    @staticmethod
    def extract_output(output):
        # 使用更鲁棒的正则匹配数字
        matches = re.findall(r'\[answer\][^\d]*(\d+)[^\d]*\[/answer\]', output)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = cls.solve(identity['n'], identity['k'], identity['a'])
        return solution == expected

    @staticmethod
    def solve(n, k, a):
        def flip(x):
            return (1 << k) - 1 - x

        prefix = [0]
        for num in a:
            prefix.append(prefix[-1] ^ num)

        cnt = Counter(prefix)
        total = n * (n + 1) // 2
        processed = set()

        for key in cnt:
            if key in processed:
                continue
            
            complement = flip(key)
            # 处理键对避免重复计算
            if key > complement and complement in cnt:
                processed.update([key, complement])
                continue

            # 计算当前键和补码的总出现次数
            current_count = cnt[key]
            complement_count = cnt.get(complement, 0) if key != complement else 0

            # 合并相同补码的情况
            total_pairs = current_count + complement_count if key != complement else current_count

            # 最优分割策略
            max_half = (total_pairs + 1) // 2
            min_half = total_pairs // 2
            total -= max_half * (max_half - 1) // 2
            total -= min_half * (min_half - 1) // 2

            processed.update({key, complement})

        return total

# 增强的单元测试
if __name__ == "__main__":
    def test_case(n, k, a, expect):
        res = Dchangingarraybootcamp.solve(n, k, a)
        assert res == expect, f"""
Test Failed:
n={n}, k={k}, a={a}
Expected: {expect}
Actual:   {res}"""

    # 官方样例
    test_case(3, 2, [1, 3, 0], 5)
    test_case(6, 3, [1, 4, 4, 7, 3, 4], 19)
    
    # 新增边界测试
    test_case(1, 1, [0], 1)    # 翻转后为[1]
    test_case(1, 1, [1], 1)    # 保持原值
    test_case(2, 1, [0, 0], 2) # 最佳方案[1,1] → 2个单元素
    test_case(3, 2, [0, 0, 0], 4)  # 最佳方案补码全翻转为3 → 前缀序列[0,3,0,3]
    
    print("所有测试用例通过验证")

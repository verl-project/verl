"""# 

### 谜题描述
Petya loves lucky numbers very much. Everybody knows that lucky numbers are positive integers whose decimal record contains only the lucky digits 4 and 7. For example, numbers 47, 744, 4 are lucky and 5, 17, 467 are not.

Petya has sequence a consisting of n integers.

The subsequence of the sequence a is such subsequence that can be obtained from a by removing zero or more of its elements.

Two sequences are considered different if index sets of numbers included in them are different. That is, the values ​of the elements ​do not matter in the comparison of subsequences. In particular, any sequence of length n has exactly 2n different subsequences (including an empty subsequence).

A subsequence is considered lucky if it has a length exactly k and does not contain two identical lucky numbers (unlucky numbers can be repeated any number of times).

Help Petya find the number of different lucky subsequences of the sequence a. As Petya's parents don't let him play with large numbers, you should print the result modulo prime number 1000000007 (109 + 7).

Input

The first line contains two integers n and k (1 ≤ k ≤ n ≤ 105). The next line contains n integers ai (1 ≤ ai ≤ 109) — the sequence a. 

Output

On the single line print the single number — the answer to the problem modulo prime number 1000000007 (109 + 7).

Examples

Input

3 2
10 10 10


Output

3


Input

4 2
4 4 7 7


Output

4

Note

In the first sample all 3 subsequences of the needed length are considered lucky.

In the second sample there are 4 lucky subsequences. For them the sets of indexes equal (the indexation starts from 1): {1, 3}, {1, 4}, {2, 3} and {2, 4}.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
N, K = map(int, raw_input().split())
ll = map(int, raw_input().split())

MOD = 10**9+7
rest = 0
lk = {}
def islucky(x):
    global lk, rest

    if x in lk:
        lk[x] += 1
        return

    result = True
    for ch in str(x):
        if ch != \"4\" and ch != \"7\":
            result = False
            break

    if result:
        if x not in lk:
            lk[x] = 0

        lk[x] += 1
    else:
        rest += 1

for elem in ll:
    islucky(elem)

llk = lk.values()

dp = {}
def solve(ind, need):
    global MOD

    if need == 0:
        return 1

    if ind < 0 or need < 0 or ind+1 < need:
        return 0

    if (ind, need) in dp:
        return dp[(ind, need)]

    dp[(ind, need)] = solve(ind-1, need) + solve(ind-1, need-1) * llk[ind]
    dp[(ind, need)] %= MOD

    return dp[(ind, need)]


facts = [1 for _ in xrange(N+5)]
for i in xrange(2, len(facts)):
    facts[i] = facts[i-1] * i
    facts[i] %= MOD

def comber(a, b):
    global MOD, facts

    if b == 0:
        return 1

    if b > a or a < 0 or b < 0:
        return 0

    return (facts[a] * pow(facts[b], MOD-2, MOD) * pow(facts[a-b], MOD-2, MOD)) % MOD

ans = 0
for i in xrange(len(lk)+1):
    ans += solve(len(lk)-1, i) * comber(rest, K-i)
    ans %= MOD
    #print \"==\", i, ans

print ans
#print dp
#print lk, llk, rest
#print facts
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

def calculate_answer(n, k, a):
    rest = 0
    lk = {}

    def islucky(x):
        nonlocal rest
        s = str(x)
        for c in s:
            if c not in {'4', '7'}:
                rest += 1
                return False
        lk[x] = lk.get(x, 0) + 1
        return True

    for elem in a:
        islucky(elem)

    llk = list(lk.values())
    m = len(llk)
    dp = {}

    def solve(ind, need):
        if need == 0:
            return 1
        if ind < 0 or need < 0 or ind + 1 < need:
            return 0
        if (ind, need) in dp:
            return dp[(ind, need)]
        res = (solve(ind-1, need) + solve(ind-1, need-1) * llk[ind]) % MOD
        dp[(ind, need)] = res
        return res

    facts = [1] * (n + 5)
    for i in range(2, len(facts)):
        facts[i] = (facts[i-1] * i) % MOD

    def comber(a_num, b_num):
        if b_num == 0:
            return 1
        if b_num > a_num or a_num < 0 or b_num < 0:
            return 0
        numerator = facts[a_num]
        denominator = (facts[b_num] * facts[a_num - b_num]) % MOD
        return (numerator * pow(denominator, MOD-2, MOD)) % MOD

    ans = 0
    max_i = min(m, k)
    for i in range(0, max_i + 1):
        needed = k - i
        if needed < 0 or needed > rest:
            continue
        way_lucky = solve(m-1, i) if m > 0 else (0 if i > 0 else 1)
        way_non_lucky = comber(rest, needed)
        ans = (ans + way_lucky * way_non_lucky) % MOD
    return ans

class Cluckysubsequencebootcamp(Basebootcamp):
    def __init__(self, max_lucky_types=3, min_n=1, max_n=10, max_rest_count=5):
        self.max_lucky_types = max_lucky_types
        self.min_n = min_n
        self.max_n = max_n
        self.max_rest_count = max_rest_count
        self.max_lucky_count = max(1, max_n // 2)  # 新增参数控制单个幸运数最大数量
    
    @staticmethod
    def is_lucky(x):
        s = str(x)
        return all(c in {'4', '7'} for c in s)
    
    def generate_lucky_number(self):
        return int(''.join(random.choice(['4', '7']) for _ in range(random.randint(1, 4))))
    
    def generate_non_lucky_number(self):
        while True:
            num = random.randint(1, 10**9)
            s = list(str(num))
            if any(c not in {'4', '7'} for c in s):
                return num
            # 强制修改最后一位为非幸运数字
            s[-1] = random.choice(['0', '1', '2', '3', '5', '6', '8', '9'])
            return int(''.join(s))
    
    def case_generator(self):
        for _ in range(1000):  # 防止无限循环
            m = random.randint(0, min(self.max_lucky_types, self.max_n))
            lucky_numbers = list({self.generate_lucky_number() for _ in range(m)})
            m = len(lucky_numbers)  # 实际生成的唯一幸运数数量
            
            # 生成counts并确保总和不超过max_n
            counts = []
            sum_c = 0
            for _ in range(m):
                max_possible = min(self.max_lucky_count, self.max_n - sum_c)
                if max_possible < 1:
                    break
                cnt = random.randint(1, max_possible)
                counts.append(cnt)
                sum_c += cnt
                if sum_c >= self.max_n:
                    break
            
            # 计算rest_count范围
            rest_min = max(0, self.min_n - sum_c)
            rest_max = min(
                self.max_rest_count,
                self.max_n - sum_c,
                self.max_n - rest_min  # 确保总和不超过max_n
            )
            
            if rest_min > rest_max:
                continue
            
            rest_count = random.randint(rest_min, rest_max)
            n = sum_c + rest_count
            
            # 最终校验
            if not (self.min_n <= n <= self.max_n):
                continue
            if not (1 <= rest_count <= self.max_rest_count):
                continue
            
            k = random.randint(1, n)
            
            # 构建数组
            a = []
            for num, cnt in zip(lucky_numbers, counts):
                a += [num] * cnt
            a += [self.generate_non_lucky_number() for _ in range(rest_count)]
            random.shuffle(a)
            
            try:
                expected = calculate_answer(n, k, a)
                return {
                    'n': n,
                    'k': k,
                    'a': a,
                    'expected': expected
                }
            except:
                continue
        raise ValueError("Failed to generate valid case after 1000 attempts")
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        a = question_case['a']
        problem_desc = (
            "Petya loves lucky numbers. Lucky numbers are positive integers that only contain digits 4 and 7. "
            "You need to find the number of distinct lucky subsequences of length exactly k. A subsequence is lucky "
            "if it contains no duplicate lucky numbers (non-lucky numbers can be repeated). Subsequences are "
            "considered different if their indices are different. Output the answer modulo 1,000,000,007.\n\n"
            "Input format:\n- First line: n and k (1 ≤ k ≤ n ≤ 1e5)\n- Second line: n integers a_1 to a_n (1 ≤ a_i ≤ 1e9)\n\n"
            "Your task:\n"
            f"n = {n}, k = {k}\n"
            f"Array a: {a}\n\n"
            "Compute the result and put your final answer within [answer] and [/answer]."
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip()) % MOD
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('expected')
        if expected is None:
            return False
        return (solution % MOD) == (expected % MOD)

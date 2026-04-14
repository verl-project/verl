"""# 

### 谜题描述
An array b is called to be a subarray of a if it forms a continuous subsequence of a, that is, if it is equal to a_l, a_{l + 1}, …, a_r for some l, r.

Suppose m is some known constant. For any array, having m or more elements, let's define it's beauty as the sum of m largest elements of that array. For example: 

  * For array x = [4, 3, 1, 5, 2] and m = 3, the 3 largest elements of x are 5, 4 and 3, so the beauty of x is 5 + 4 + 3 = 12.
  * For array x = [10, 10, 10] and m = 2, the beauty of x is 10 + 10 = 20.



You are given an array a_1, a_2, …, a_n, the value of the said constant m and an integer k. Your need to split the array a into exactly k subarrays such that:

  * Each element from a belongs to exactly one subarray.
  * Each subarray has at least m elements.
  * The sum of all beauties of k subarrays is maximum possible.

Input

The first line contains three integers n, m and k (2 ≤ n ≤ 2 ⋅ 10^5, 1 ≤ m, 2 ≤ k, m ⋅ k ≤ n) — the number of elements in a, the constant m in the definition of beauty and the number of subarrays to split to.

The second line contains n integers a_1, a_2, …, a_n (-10^9 ≤ a_i ≤ 10^9).

Output

In the first line, print the maximum possible sum of the beauties of the subarrays in the optimal partition.

In the second line, print k-1 integers p_1, p_2, …, p_{k-1} (1 ≤ p_1 < p_2 < … < p_{k-1} < n) representing the partition of the array, in which:

  * All elements with indices from 1 to p_1 belong to the first subarray.
  * All elements with indices from p_1 + 1 to p_2 belong to the second subarray.
  * ….
  * All elements with indices from p_{k-1} + 1 to n belong to the last, k-th subarray.



If there are several optimal partitions, print any of them.

Examples

Input


9 2 3
5 2 5 2 4 1 1 3 2


Output


21
3 5 

Input


6 1 4
4 1 3 2 2 3


Output


12
1 3 5 

Input


2 1 2
-1000000000 1000000000


Output


0
1 

Note

In the first example, one of the optimal partitions is [5, 2, 5], [2, 4], [1, 1, 3, 2].

  * The beauty of the subarray [5, 2, 5] is 5 + 5 = 10. 
  * The beauty of the subarray [2, 4] is 2 + 4 = 6. 
  * The beauty of the subarray [1, 1, 3, 2] is 3 + 2 = 5. 



The sum of their beauties is 10 + 6 + 5 = 21.

In the second example, one optimal partition is [4], [1, 3], [2, 2], [3].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
from sys import stdin, stdout


def write(x):
    stdout.write(str(x) + \"\n\")


n, m, k = map(int, stdin.readline().split())
a = map(int, stdin.readline().split())

asort = sorted(a, reverse=True)
cut = m * k - 1
out = []
currentin = 0
for i in xrange(len(a)):
    if a[i] >= asort[cut]:
        if a[i] == asort[cut]:
            cut -= 1
        currentin += 1
        if currentin == m:
            out.append(i + 1)
            currentin = 0
            if len(out) + 1 == k:
                break

write(sum(asort[:m * k]))
print(*out)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Byetanotherarraypartitioningtaskbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_m=3, max_k=4):
        self.max_n = max_n
        self.max_m = max_m
        self.max_k = max_k
    
    def case_generator(self):
        for _ in range(1000):  # 防止无限循环，最多尝试1000次
            m = random.randint(1, self.max_m)
            k = random.randint(2, self.max_k)
            if m * k > self.max_n:
                continue
            n = random.randint(m * k, self.max_n)
            a_top = [random.randint(50, 100) for _ in range(m * k)]
            a_rest = [random.randint(-100, 0) for _ in range(n - m * k)]
            a = a_top + a_rest
            random.shuffle(a)
            sum_answer, partitions = self.generate_solution(a, m, k)
            if len(partitions) == k - 1:
                return {
                    'n': n,
                    'm': m,
                    'k': k,
                    'a': a,
                    'sum_answer': sum_answer,
                    'partitions': partitions
                }
        raise ValueError("Unable to generate valid test case after 1000 attempts")
    
    @staticmethod
    def generate_solution(a, m, k):
        asort = sorted(a, reverse=True)
        total_sum = sum(asort[:m * k])
        cut = m * k - 1
        out = []
        currentin = 0
        for i in range(len(a)):
            # 防止cut越界
            if cut >= 0 and a[i] >= asort[cut]:
                if a[i] == asort[cut]:
                    cut -= 1
                currentin += 1
                if currentin == m:
                    out.append(i + 1)
                    currentin = 0
                    if len(out) == k - 1:
                        break
        return total_sum, out
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']} {question_case['k']}",
            ' '.join(map(str, question_case['a']))
        ]
        input_str = '\n'.join(input_lines)
        return f"""You are a competitive programmer. Solve the subarray partition problem by splitting the array into exactly {question_case['k']} subarrays, each with at least {question_case['m']} elements. The beauty of each subarray is the sum of its {question_case['m']} largest elements. Find the maximum total beauty and the partition points.

Input:
{input_str}

Output the maximum sum on the first line and the partition points (k-1 integers) on the second line. Enclose your answer within [answer] and [/answer]. For example:
[answer]
42
1 3 5
[/answer]
"""
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            sum_answer = int(lines[0])
            partitions = list(map(int, lines[1].split()))
        except:
            return None
        return (sum_answer, partitions)
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != 2:
            return False
        sum_user, partitions = solution
        k = identity['k']
        n = identity['n']
        m = identity['m']
        # 检查总和是否正确
        if sum_user != identity['sum_answer']:
            return False
        # 检查分割点数量
        if len(partitions) != k - 1:
            return False
        # 检查分割点是否递增且在合理范围
        prev = 0
        for p in partitions:
            if p <= prev or p < 1 or p >= n:
                return False
            prev = p
        # 检查每个子数组长度至少m
        current_start = 0
        for p in partitions + [n]:
            sub_length = p - current_start
            if sub_length < m:
                return False
            current_start = p
        # 检查实际总和是否匹配
        current_start = 0
        total = 0
        a = identity['a']
        for p in partitions + [n]:
            sub = a[current_start:p]
            sorted_sub = sorted(sub, reverse=True)
            total += sum(sorted_sub[:m])
            current_start = p
        return total == sum_user

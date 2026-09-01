"""# 

### 谜题描述
Pikachu had an array with him. He wrote down all the non-empty subsequences of the array on paper. Note that an array of size n has 2n - 1 non-empty subsequences in it. 

Pikachu being mischievous as he always is, removed all the subsequences in which Maximum_element_of_the_subsequence -  Minimum_element_of_subsequence ≥ d

Pikachu was finally left with X subsequences. 

However, he lost the initial array he had, and now is in serious trouble. He still remembers the numbers X and d. He now wants you to construct any such array which will satisfy the above conditions. All the numbers in the final array should be positive integers less than 1018. 

Note the number of elements in the output array should not be more than 104. If no answer is possible, print  - 1.

Input

The only line of input consists of two space separated integers X and d (1 ≤ X, d ≤ 109).

Output

Output should consist of two lines.

First line should contain a single integer n (1 ≤ n ≤ 10 000)— the number of integers in the final array.

Second line should consist of n space separated integers — a1, a2, ... , an (1 ≤ ai < 1018).

If there is no answer, print a single integer -1. If there are multiple answers, print any of them.

Examples

Input

10 5


Output

6
5 50 7 15 6 100

Input

4 2


Output

4
10 100 1000 10000

Note

In the output of the first example case, the remaining subsequences after removing those with Maximum_element_of_the_subsequence -  Minimum_element_of_subsequence ≥ 5 are [5], [5, 7], [5, 6], [5, 7, 6], [50], [7], [7, 6], [15], [6], [100]. There are 10 of them. Hence, the array [5, 50, 7, 15, 6, 100] is valid.

Similarly, in the output of the second example case, the remaining sub-sequences after removing those with Maximum_element_of_the_subsequence -  Minimum_element_of_subsequence ≥ 2 are [10], [100], [1000], [10000]. There are 4 of them. Hence, the array [10, 100, 1000, 10000] is valid.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# Team ASML

import math

def make_list(X):
  L = []
  while (X > 0):
    n = int(math.log(X+1, 2))
    X = X - (2**n - 1)
    L.append(n)
  return L

def solve():
  X, d = map(int, raw_input().split())
  L = make_list(X)
  
  result = []
  start = 1
  
  for l in L:
    for e in range(start, start+l):
      result.append(start)
    start = result[-1]+d

  if (len(result) <= 10000) and (result[-1] < 10**18):
    print(len(result))
    print(\" \".join((\"%d\" % i) for i in result))
  else:
    print(-1)

if __name__ == \"__main__\":
  solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Csubsequencecountingbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
    
    def case_generator(self):
        """生成有效的X和d组合，确保数组符合所有约束条件"""
        max_attempts = 100  # 防止无限循环
        
        for _ in range(max_attempts):
            # 生成L列表，确保sum(2^m-1) ≤1e9且元素数目≤1e4
            L = []
            total_elements = 0  # 数组总长度
            X = 0
            
            # 随机生成1-3个分组（保证元素数目可控）
            group_count = random.randint(1, 3)
            for _ in range(group_count):
                # 每组元素数量m的2^m-1值不能超过剩余X空间
                remaining_X = 10**9 - X
                if remaining_X <= 0:
                    break
                
                max_m = min(
                    30,  # 防止生成过大的指数
                    math.floor(math.log2(remaining_X + 1))
                )
                if max_m <= 0: break
                
                m = random.randint(1, max_m)
                term = 2**m - 1
                if X + term > 10**9: continue
                
                L.append(m)
                X += term
                total_elements += m
            
            if not L: continue
            if random.random() < 0.3:  # 30%概率生成可能无解的测试用例
                X = random.randint(X + 1, X * 2)
            
            # 计算可用d的范围
            k = len(L)
            if k == 1:
                d_max = 10**18 - 1
            else:
                d_max = (10**18 - 1) // (k - 1)
            if d_max < 1: continue
            d = random.randint(1, d_max)
            
            # 生成预期解
            array = []
            current = 1
            for m in L:
                array += [current] * m
                current += d
            
            # 有效性检查
            if (len(array) > 10000 or 
                (array and array[-1] >= 10**18) or 
                (X > 10**9)):
                continue
            
            return {
                'X': X,
                'd': d,
                'expected_array': array,
                'subseq_count': sum(2**m-1 for m in L)
            }
        
        # 达到最大尝试次数仍无解
        return {
            'X': 0,
            'd': 1,
            'expected_array': -1,
            'subseq_count': 0
        }

    @staticmethod
    def prompt_func(case):
        X, d = case['X'], case['d']
        return f"""给定两个整数X={X}和d={d}，构造一个正整数数组：
1. 所有元素都是严格递增的正整数
2. 数组长度不超过10000
3. 所有元素必须小于1e18
4. 满足：所有非空子序列中，最大值-最小值<d的子序列数等于X

按以下格式返回答案：
[answer]
n
a1 a2 ... an
[/answer]
如果无解输出-1"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches: return None
        
        content = matches[-1].strip()
        if content == '-1': return -1
        
        try:
            lines = content.split('\n')
            n = int(lines[0])
            arr = list(map(int, lines[1].split()))
            if len(arr) != n: return None
            return arr
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == -1:
            return identity['expected_array'] == -1
        
        # 基本约束检查
        if not (1 <= len(solution) <= 10000):
            return False
        if any(not (1 <= x < 10**18) for x in solution):
            return False
        
        # 分析分组结构
        groups = []
        current_group = [solution[0]]
        for num in solution[1:]:
            if num == current_group[-1]:
                current_group.append(num)
            else:
                if num <= current_group[-1]:
                    return False  # 必须严格递增
                groups.append(current_group)
                current_group = [num]
        groups.append(current_group)
        
        # 检查相邻组差
        d = identity['d']
        for i in range(1, len(groups)):
            if groups[i][0] - groups[i-1][-1] < d:
                return False
        
        # 计算有效子序列数
        valid_count = sum(2**len(g) - 1 for g in groups)
        return valid_count == identity['X']

# 示例用法
if __name__ == "__main__":
    bootcamp = Csubsequencecountingbootcamp()
    test_case = bootcamp.case_generator()
    print(test_case)

"""# 

### 谜题描述
At the children's day, the child came to Picks's house, and messed his house up. Picks was angry at him. A lot of important things were lost, in particular the favorite set of Picks.

Fortunately, Picks remembers something about his set S:

  * its elements were distinct integers from 1 to limit; 
  * the value of <image> was equal to sum; here lowbit(x) equals 2k where k is the position of the first one in the binary representation of x. For example, lowbit(100102) = 102, lowbit(100012) = 12, lowbit(100002) = 100002 (binary representation). 



Can you help Picks and find any set S, that satisfies all the above conditions?

Input

The first line contains two integers: sum, limit (1 ≤ sum, limit ≤ 105).

Output

In the first line print an integer n (1 ≤ n ≤ 105), denoting the size of S. Then print the elements of set S in any order. If there are multiple answers, print any of them.

If it's impossible to find a suitable set, print -1.

Examples

Input

5 5


Output

2
4 5


Input

4 3


Output

3
2 3 1


Input

5 1


Output

-1

Note

In sample test 1: lowbit(4) = 4, lowbit(5) = 1, 4 + 1 = 5.

In sample test 2: lowbit(1) = 1, lowbit(2) = 2, lowbit(3) = 1, 1 + 2 + 1 = 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m=map(int,raw_input().split())
a=[]
for i in range(1,m+1):
    for j in range(15):
        if i&(1<<j)>0:
            a.append([1<<j,i])
            break
a=sorted(a,key=lambda z: z[0])
a.reverse()
b=[]
for i in a:
    if n-i[0]>=0:
        b.append(i[1])
        n-=i[0]
if n==0:
    print len(b)
    for i in b:
        print i,
else:
    print -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Bthechildandsetbootcamp(Basebootcamp):
    def __init__(self, min_limit=1, max_limit=10**5):
        self.min_limit = min_limit
        self.max_limit = max_limit

    def case_generator(self):
        # 50%概率生成有解案例，50%生成无解案例
        if random.random() < 0.5:
            # 生成有解案例
            limit = random.randint(self.min_limit, self.max_limit)
            
            # 收集所有lowbit信息
            candidates = []
            for num in range(1, limit+1):
                lb = num & -num
                candidates.append((lb, num))
            
            # 按lowbit降序排序
            candidates.sort(reverse=True, key=lambda x: x[0])
            
            # 随机选择有效子集
            selected = []
            sum_total = 0
            for lb, num in candidates:
                if random.random() < 0.7:  # 70%概率选择当前元素
                    selected.append(num)
                    sum_total += lb
                if sum_total > 0 and random.random() < 0.3:  # 30%概率停止
                    break
            
            # 确保至少选择一个元素
            if not selected:
                selected.append(candidates[0][1])
                sum_total = candidates[0][0]
            
            return {
                'sum': sum_total,
                'limit': limit,
                '_solution': selected  # 隐藏的解信息用于验证
            }
        else:
            # 生成无解案例
            limit = random.randint(self.min_limit, self.max_limit)
            max_sum = sum(num & -num for num in range(1, limit+1))
            return {
                'sum': max_sum + random.randint(1, 100),
                'limit': limit,
                '_solution': None
            }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        sum_val = question_case['sum']
        limit_val = question_case['limit']
        return f"""Picks需要找到满足以下条件的整数集合：
1. 所有元素都是1到{limit_val}之间的不同整数
2. 所有元素的lowbit之和等于{sum_val}

lowbit定义：数字二进制表示中最后一位1所代表的值，例如：
- lowbit(6) = 2（二进制110）
- lowbit(12) = 4（二进制1100）

如果存在这样的集合，按任意顺序输出元素；否则输出-1。
答案请用[answer]标签包裹，例如：
[answer]
3
1 2 3
[/answer]"""

    @staticmethod
    def extract_output(output):
        match = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not match:
            return None
        content = match[-1].strip()
        if content == '-1':
            return -1
        try:
            parts = list(map(int, content.split()))
            if len(parts) < 1:
                return None
            n = parts[0]
            elements = parts[1:1+n]
            if len(elements) != n:
                return None
            return elements
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证时优先使用隐藏解信息
        if '_solution' in identity:
            if identity['_solution'] is None:
                return solution == -1
            if solution == -1:
                return False
            return set(solution) == set(identity['_solution'])
        
        # 常规验证逻辑
        if solution == -1:
            return not cls.find_solution(identity['sum'], identity['limit'])
        
        sum_total = 0
        limit = identity['limit']
        seen = set()
        for num in solution:
            if not (1 <= num <= limit) or num in seen:
                return False
            seen.add(num)
            sum_total += num & -num
        return sum_total == identity['sum']

    @classmethod
    def find_solution(cls, sum_val, limit):
        # 参考原题给出的解题算法
        candidates = []
        for num in range(1, limit+1):
            lb = num & -num
            candidates.append((lb, num))
        
        candidates.sort(reverse=True, key=lambda x: x[0])
        remaining = sum_val
        selected = []
        for lb, num in candidates:
            if remaining >= lb:
                selected.append(num)
                remaining -= lb
        return selected if remaining == 0 else None

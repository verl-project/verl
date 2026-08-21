"""# 

### 谜题描述
Cowboy Vlad has a birthday today! There are n children who came to the celebration. In order to greet Vlad, the children decided to form a circle around him. Among the children who came, there are both tall and low, so if they stand in a circle arbitrarily, it may turn out, that there is a tall and low child standing next to each other, and it will be difficult for them to hold hands. Therefore, children want to stand in a circle so that the maximum difference between the growth of two neighboring children would be minimal possible.

Formally, let's number children from 1 to n in a circle order, that is, for every i child with number i will stand next to the child with number i+1, also the child with number 1 stands next to the child with number n. Then we will call the discomfort of the circle the maximum absolute difference of heights of the children, who stand next to each other.

Please help children to find out how they should reorder themselves, so that the resulting discomfort is smallest possible.

Input

The first line contains a single integer n (2 ≤ n ≤ 100) — the number of the children who came to the cowboy Vlad's birthday.

The second line contains integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^9) denoting heights of every child.

Output

Print exactly n integers — heights of the children in the order in which they should stand in a circle. You can start printing a circle with any child.

If there are multiple possible answers, print any of them.

Examples

Input


5
2 1 1 3 2


Output


1 2 3 2 1


Input


3
30 10 20


Output


10 20 30

Note

In the first example, the discomfort of the circle is equal to 1, since the corresponding absolute differences are 1, 1, 1 and 0. Note, that sequences [2, 3, 2, 1, 1] and [3, 2, 1, 1, 2] form the same circles and differ only by the selection of the starting point.

In the second example, the discomfort of the circle is equal to 20, since the absolute difference of 10 and 30 is equal to 20.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
l=map(int,raw_input().split())
l.sort()
t=[0]*n
i=0
j=n-1
for k in range(n):
    if k%2==0:
        t[i]=l[k]
        i+=1
    else:
        t[j]=l[k]
        j-=1
if n%2==1:
    t[n/2]=l[n-1]
for i in t:
    print i,
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cbirthdaybootcamp(Basebootcamp):
    def __init__(self, **kwargs):
        self.n_min = kwargs.pop('n_min', 2)
        self.n_max = kwargs.pop('n_max', 100)
        self.min_val = kwargs.pop('min_val', 1)
        self.max_val = kwargs.pop('max_val', 10**9)
        super().__init__(**kwargs)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        heights = [random.randint(self.min_val, self.max_val) for _ in range(n)]
        sorted_heights = sorted(heights)
        t = [0] * n
        i, j = 0, n - 1
        
        for k in range(n):
            if k % 2 == 0:
                t[i] = sorted_heights[k]
                i += 1
            else:
                t[j] = sorted_heights[k]
                j -= 1
        
        # 显式处理奇数情况
        if n % 2 == 1 and n > 1:
            t[n//2] = sorted_heights[-1]
        
        max_diff = max(abs(t[i] - t[(i+1)%n]) for i in range(n))
        return {'n': n, 'heights': heights, 'expected_max_diff': max_diff}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        example = "\n".join([
            "[answer]",
            "1 2 3 2 1" if n == 5 else "10 20 30",
            "[/answer]"
        ])
        return f"""## 题目背景
孩子们需要围成一个圆圈，使得相邻身高差的最大值最小。

## 输入格式
第1行：整数n（人数）
第2行：包含n个整数的列表

## 你的任务
找到任意一个最优排列方案，将结果用空格分隔放在[answer]标签内

当前问题：
n = {n}
身高数据 = {' '.join(map(str, question_case['heights']))}

{example}"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL|re.IGNORECASE)
        if not matches:
            return None
        last_match = matches[-1].strip().replace('\n', ' ')
        try:
            return list(map(int, re.findall(r'-?\d+', last_match)))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 完整性校验
        if not solution or len(solution) != identity['n']:
            return False
        # 元素匹配校验
        if sorted(solution) != sorted(identity['heights']):
            return False
        # 计算真实差异
        current_max = max(
            abs(solution[i] - solution[(i+1)%identity['n']])
            for i in range(identity['n'])
        )
        return current_max == identity['expected_max_diff']

"""# 

### 谜题描述
The Berland University is preparing to celebrate the 256-th anniversary of its founding! A specially appointed Vice Rector for the celebration prepares to decorate the campus. In the center of the campus n ice sculptures were erected. The sculptures are arranged in a circle at equal distances from each other, so they form a regular n-gon. They are numbered in clockwise order with numbers from 1 to n.

The site of the University has already conducted a voting that estimated each sculpture's characteristic of ti — the degree of the sculpture's attractiveness. The values of ti can be positive, negative or zero.

When the university rector came to evaluate the work, he said that this might be not the perfect arrangement. He suggested to melt some of the sculptures so that: 

  * the remaining sculptures form a regular polygon (the number of vertices should be between 3 and n), 
  * the sum of the ti values of the remaining sculptures is maximized. 



Help the Vice Rector to analyze the criticism — find the maximum value of ti sum which can be obtained in this way. It is allowed not to melt any sculptures at all. The sculptures can not be moved.

Input

The first input line contains an integer n (3 ≤ n ≤ 20000) — the initial number of sculptures. The second line contains a sequence of integers t1, t2, ..., tn, ti — the degree of the i-th sculpture's attractiveness ( - 1000 ≤ ti ≤ 1000). The numbers on the line are separated by spaces.

Output

Print the required maximum sum of the sculptures' attractiveness.

Examples

Input

8
1 2 -3 4 -5 5 2 3


Output

14


Input

6
1 -2 3 -4 5 -6


Output

9


Input

6
1 2 3 4 5 6


Output

21

Note

In the first sample it is best to leave every second sculpture, that is, leave sculptures with attractivenesses: 2, 4, 5 и 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
    n = int( raw_input() )
    L = [ int(i) for i in raw_input().split(\" \") ]
    
    maxi = -10**15
    for i in range(1,n+1):
        if n % i == 0:
            for k in range(0,i):
                ptotal = 0
                count = 0
                for j in range(k,n,i):
                    ptotal += L[j]
                    count += 1
                if count >= 3:
                    maxi = max(ptotal,maxi)
    
    print maxi
    
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def calculate_max_sum(n, t_list):
    maxi = -10**15
    # 需要包含所有可能的因数（包括n自身）
    divisors = set()
    for step in range(1, int(n**0.5)+1):
        if n % step == 0:
            divisors.add(step)
            divisors.add(n//step)
    
    for step in sorted(divisors):
        m = n // step
        if m < 3:
            continue
        for k in range(step):
            total = sum(t_list[j] for j in range(k, n, step))
            if total > maxi:
                maxi = total
    return maxi

class Dicesculpturesbootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=20000, t_min=-1000, t_max=1000):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.t_min = t_min
        self.t_max = t_max
    
    def case_generator(self):
        while True:
            n = random.randint(self.n_min, self.n_max)
            t = [random.randint(self.t_min, self.t_max) for _ in range(n)]
            max_sum = calculate_max_sum(n, t)
            # 确保至少存在合法解
            if max_sum != -10**15:
                return {
                    "n": n,
                    "t": t,
                    "max_sum": max_sum
                }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        t_list = question_case["t"]
        t_str = " ".join(map(str, t_list))
        return f"""Berland大学校庆需要装饰{n}个冰雕（编号1~{n}排列成正多边形），各冰雕吸引力值为：{t_str}
            
规则：
1. 融化部分冰雕，剩余冰雕必须构成正多边形
2. 剩余数量k必须是{n}的因数且3 ≤ k ≤ {n}
3. 冰雕间距必须相等（步长m = {n}/k）

请计算最大可能的吸引力总和，答案置于[answer]标签内，如：[answer]21[/answer]。
"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity["max_sum"]

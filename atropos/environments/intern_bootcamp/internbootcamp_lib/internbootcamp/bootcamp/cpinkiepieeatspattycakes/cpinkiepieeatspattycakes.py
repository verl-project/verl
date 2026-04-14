"""# 

### 谜题描述
Pinkie Pie has bought a bag of patty-cakes with different fillings! But it appeared that not all patty-cakes differ from one another with filling. In other words, the bag contains some patty-cakes with the same filling.

Pinkie Pie eats the patty-cakes one-by-one. She likes having fun so she decided not to simply eat the patty-cakes but to try not to eat the patty-cakes with the same filling way too often. To achieve this she wants the minimum distance between the eaten with the same filling to be the largest possible. Herein Pinkie Pie called the distance between two patty-cakes the number of eaten patty-cakes strictly between them.

Pinkie Pie can eat the patty-cakes in any order. She is impatient about eating all the patty-cakes up so she asks you to help her to count the greatest minimum distance between the eaten patty-cakes with the same filling amongst all possible orders of eating!

Pinkie Pie is going to buy more bags of patty-cakes so she asks you to solve this problem for several bags!

Input

The first line contains a single integer T (1 ≤ T ≤ 100): the number of bags for which you need to solve the problem.

The first line of each bag description contains a single integer n (2 ≤ n ≤ 10^5): the number of patty-cakes in it. The second line of the bag description contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ n): the information of patty-cakes' fillings: same fillings are defined as same integers, different fillings are defined as different integers. It is guaranteed that each bag contains at least two patty-cakes with the same filling. 

It is guaranteed that the sum of n over all bags does not exceed 10^5.

Output

For each bag print in separate line one single integer: the largest minimum distance between the eaten patty-cakes with the same filling amongst all possible orders of eating for that bag.

Example

Input


4
7
1 7 1 6 4 4 6
8
1 1 4 6 4 6 4 7
3
3 3 3
6
2 5 2 3 1 4


Output


3
2
0
4

Note

For the first bag Pinkie Pie can eat the patty-cakes in the following order (by fillings): 1, 6, 4, 7, 1, 6, 4 (in this way, the minimum distance is equal to 3).

For the second bag Pinkie Pie can eat the patty-cakes in the following order (by fillings): 1, 4, 6, 7, 4, 1, 6, 4 (in this way, the minimum distance is equal to 2).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
for _ in range(input()):
    n=input()
    a=[int(i) for i in raw_input().split()]
    d={}
    for i in a:
        d[i]=d.get(i,0)+1
    maxx=max(d.values())
    ct=0
    for i in d.values():
        if i==maxx:
            ct+=1
    v=(n-ct)/(maxx-1)
    print v-1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Cpinkiepieeatspattycakesbootcamp(Basebootcamp):
    def __init__(self, max_maxx=5, max_ct=3, max_v=10):
        self.max_maxx = max_maxx    # 最大重复次数上限
        self.max_ct = max_ct        # 相同最大次数填充类型上限
        self.max_v = max_v          # 虚拟参数v的取值范围上限

    def case_generator(self):
        # 保证至少有两次重复
        maxx = random.randint(2, self.max_maxx)
        ct = random.randint(1, self.max_ct)
        
        # 动态调整v的合法取值范围
        min_v = max(0, ct - 1)  # 根据公式推导的最小合法v值
        max_v = max(min_v, self.max_v)  # 保证取值范围有效性
        
        # 生成合法的间隔参数v
        v = random.randint(min_v, max_v)
        
        # 根据题目公式计算总蛋糕数
        n = (maxx - 1) * (v + 1) + ct
        
        # 生成基础重复元素（保证最大重复次数）
        elements = []
        for i in range(ct):
            elements += [i + 1] * maxx  # 填充类型从1开始
        
        # 添加唯一填充元素（保证数值不超过n）
        remaining = n - len(elements)
        if remaining > 0:
            start = ct + 1
            elements += list(range(start, start + remaining))
        
        # 洗牌后输出确保测试案例多样性
        random.shuffle(elements)
        
        return {
            "n": n,
            "a": elements
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        a = question_case['a']
        return f"""
你需要解决Pinkie Pie的蛋糕排列问题。当前袋子包含{n}个蛋糕，填充类型如下：{a}。
相同数字代表相同填充。请找到一种排列顺序，使得相同填充蛋糕之间的最小间隔尽可能大。
输入格式要求：最后将最终答案放在[answer]标签内，例如[answer]3[/answer]。

问题示例：
当蛋糕为[1,1,2]时，最优排列是[1,2,1]，最小间隔为1。
实际题目可能包含多个重复类型，请仔细分析最大重复次数和重复类型数量。
        """.strip()

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return answer_blocks[-1].strip() if answer_blocks else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            cnt = defaultdict(int)
            for num in identity['a']:
                cnt[num] += 1
            
            counts = list(cnt.values())
            maxx = max(counts)
            ct = counts.count(maxx)
            
            # 验证题目约束条件
            if maxx < 2 or identity['n'] != len(identity['a']):
                return False
            
            # 根据题目公式计算结果
            expected = (identity['n'] - ct) // (maxx - 1) - 1
            return int(solution) == expected
        except (ValueError, KeyError):
            return False

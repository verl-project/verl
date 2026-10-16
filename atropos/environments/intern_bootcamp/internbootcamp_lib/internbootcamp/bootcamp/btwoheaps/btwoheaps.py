"""# 

### 谜题描述
Valera has 2·n cubes, each cube contains an integer from 10 to 99. He arbitrarily chooses n cubes and puts them in the first heap. The remaining cubes form the second heap. 

Valera decided to play with cubes. During the game he takes a cube from the first heap and writes down the number it has. Then he takes a cube from the second heap and write out its two digits near two digits he had written (to the right of them). In the end he obtained a single fourdigit integer — the first two digits of it is written on the cube from the first heap, and the second two digits of it is written on the second cube from the second heap.

Valera knows arithmetic very well. So, he can easily count the number of distinct fourdigit numbers he can get in the game. The other question is: how to split cubes into two heaps so that this number (the number of distinct fourdigit integers Valera can get) will be as large as possible?

Input

The first line contains integer n (1 ≤ n ≤ 100). The second line contains 2·n space-separated integers ai (10 ≤ ai ≤ 99), denoting the numbers on the cubes.

Output

In the first line print a single number — the maximum possible number of distinct four-digit numbers Valera can obtain. In the second line print 2·n numbers bi (1 ≤ bi ≤ 2). The numbers mean: the i-th cube belongs to the bi-th heap in your division.

If there are multiple optimal ways to split the cubes into the heaps, print any of them.

Examples

Input

1
10 99


Output

1
2 1 


Input

2
13 24 13 45


Output

4
1 2 2 1 

Note

In the first test case Valera can put the first cube in the first heap, and second cube — in second heap. In this case he obtain number 1099. If he put the second cube in the first heap, and the first cube in the second heap, then he can obtain number 9910. In both cases the maximum number of distinct integers is equal to one.

In the second test case Valera can obtain numbers 1313, 1345, 2413, 2445. Note, that if he put the first and the third cubes in the first heap, he can obtain only two numbers 1324 and 1345.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
a = map(int, raw_input().split())
d = [0]*100
for x in a:
    d[x] += 1
n1 = len(filter(lambda x: x == 1, d))
n2 = len(filter(lambda x: x > 1, d))

print (n2 + n1/2)*(n2 + (n1+1)/2)

nx, k1, k3 = [1]*100, 1, 2
b = [0]*(2*n)
for i in xrange(10, 100):
    if d[i] == 1:
        nx[i] = k1
        k1 = 3-k1
    elif d[i] %2 == 1:
        nx[i] = k3
        k3 = 3-k3

for i in xrange(2*n):
    b[i] = nx[a[i]]
    nx[a[i]] = 3 - nx[a[i]]
print str(b).replace(',','').replace('[','').replace(']','')
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
from collections import defaultdict
import re
import random

class Btwoheapsbootcamp(Basebootcamp):
    def __init__(self, n_range=(1, 5), num_range=(10, 99)):
        self.n_range = n_range
        self.num_range = num_range

    def case_generator(self):
        # 生成至少包含重复数字的合法案例
        n = random.randint(*self.n_range)
        a = []
        # 确保至少有一个重复数
        for _ in range(2*n//2):
            num = random.randint(*self.num_range)
            a.extend([num]*2)
        # 补充剩余数字（如果存在奇数个）
        while len(a) < 2*n:
            a.append(random.randint(*self.num_range))
        random.shuffle(a)
        return {'n': n, 'a': a}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        a_str = ' '.join(map(str, question_case['a']))
        problem = f"""Valera有2·n个立方体，每个立方体上的数字是10到99的整数。现在需要将这些立方体分成两个堆，每个堆各n个，使得可能生成的不同的四位数数目最大。你的任务是找到这样的分堆方法，并输出最大数目和对应的堆分配方案。

输入：
第一行是n的值，即{n}。
第二行是{2*n}个用空格分隔的数字：{a_str}。

输出：
第一行输出一个整数，表示最大可能的四位数数目。
第二行输出{2*n}个用空格分隔的1或2，表示每个立方体属于哪个堆。1表示第一个堆，2表示第二个堆。必须保证每个堆恰好有n个立方体。

请将答案按照严格格式放在[answer]标签内，例如：
[answer]
4
1 2 2 1
[/answer]"""
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            solution = list(map(int, lines[1].split()))
            return solution
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 基础验证
        n = identity['n']
        a = identity['a']
        total = 2 * n
        
        if len(solution) != total:
            return False
        if sum(1 for x in solution if x == 1) != n:
            return False
        
        # 计算理论最大值
        counts = defaultdict(int)
        for num in a:
            counts[num] += 1
            
        n1 = sum(1 for cnt in counts.values() if cnt == 1)
        n2 = sum(1 for cnt in counts.values() if cnt > 1)
        max_val = (n2 + n1 // 2) * (n2 + (n1 + 1) // 2)
        
        # 计算实际四位数数量
        heap1 = [a[i] for i, s in enumerate(solution) if s == 1]
        heap2 = [a[i] for i, s in enumerate(solution) if s == 2]
        actual = len({h1*100 + h2 for h1 in heap1 for h2 in heap2})
        
        return actual == max_val

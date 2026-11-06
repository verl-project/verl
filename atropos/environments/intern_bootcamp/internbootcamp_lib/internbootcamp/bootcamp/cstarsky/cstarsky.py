"""# 

### 谜题描述
The Cartesian coordinate system is set in the sky. There you can see n stars, the i-th has coordinates (xi, yi), a maximum brightness c, equal for all stars, and an initial brightness si (0 ≤ si ≤ c).

Over time the stars twinkle. At moment 0 the i-th star has brightness si. Let at moment t some star has brightness x. Then at moment (t + 1) this star will have brightness x + 1, if x + 1 ≤ c, and 0, otherwise.

You want to look at the sky q times. In the i-th time you will look at the moment ti and you will see a rectangle with sides parallel to the coordinate axes, the lower left corner has coordinates (x1i, y1i) and the upper right — (x2i, y2i). For each view, you want to know the total brightness of the stars lying in the viewed rectangle.

A star lies in a rectangle if it lies on its border or lies strictly inside it.

Input

The first line contains three integers n, q, c (1 ≤ n, q ≤ 105, 1 ≤ c ≤ 10) — the number of the stars, the number of the views and the maximum brightness of the stars.

The next n lines contain the stars description. The i-th from these lines contains three integers xi, yi, si (1 ≤ xi, yi ≤ 100, 0 ≤ si ≤ c ≤ 10) — the coordinates of i-th star and its initial brightness.

The next q lines contain the views description. The i-th from these lines contains five integers ti, x1i, y1i, x2i, y2i (0 ≤ ti ≤ 109, 1 ≤ x1i < x2i ≤ 100, 1 ≤ y1i < y2i ≤ 100) — the moment of the i-th view and the coordinates of the viewed rectangle.

Output

For each view print the total brightness of the viewed stars.

Examples

Input

2 3 3
1 1 1
3 2 0
2 1 1 2 2
0 2 1 4 5
5 1 1 5 5


Output

3
0
3


Input

3 4 5
1 1 2
2 3 0
3 3 1
0 1 1 100 100
1 2 2 4 4
2 2 1 4 7
1 50 50 51 51


Output

3
3
5
0

Note

Let's consider the first example.

At the first view, you can see only the first star. At moment 2 its brightness is 3, so the answer is 3.

At the second view, you can see only the second star. At moment 0 its brightness is 0, so the answer is 0.

At the third view, you can see both stars. At moment 5 brightness of the first is 2, and brightness of the second is 1, so the answer is 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
R=lambda:map(int,raw_input().split())
N=128
n,q,c=R()
c+=1
a=[[[0]*N for _ in range(N)] for _ in range(12)]
for _ in range(n):
  x,y,s=R()
  for t in range(c):
    a[t][x][y]+=(s+t)%c
    
for u in range(c):
  for i in range(N):
    for j in range(1,N):
      a[u][i][j]+=a[u][i][j-1]
    for j in range(N):
      a[u][i][j]+=a[u][i-1][j]

for _ in range(q):
  t,x,y,u,v=R()
  b = a[t%c]
  print b[u][v]-b[x-1][v]-b[u][y-1]+b[x-1][y-1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cstarskybootcamp(Basebootcamp):
    def __init__(self, n=10, q=5, c=5):
        """
        初始化星星闪烁谜题训练场参数
        
        参数:
            n (int): 生成的星星数量，默认10
            q (int): 生成的查询数量，默认5
            c (int): 星星的最大亮度，默认5
        """
        self.n = n
        self.q = q
        self.c = c
    
    def case_generator(self):
        """
        生成包含星星布局和查询的谜题实例
        
        返回:
            dict: 包含完整的谜题实例数据，包含预期答案的可序列化字典
        """
        # 生成星星数据（保证坐标唯一）
        stars = []
        occupied = set()
        while len(stars) < self.n:
            x = random.randint(1, 100)
            y = random.randint(1, 100)
            if (x, y) not in occupied:
                occupied.add((x, y))
                stars.append({
                    'x': x,
                    'y': y,
                    's': random.randint(0, self.c)
                })

        # 生成查询数据
        queries = []
        for _ in range(self.q):
            x1 = random.randint(1, 99)
            x2 = random.randint(x1+1, 100)
            y1 = random.randint(1, 99)
            y2 = random.randint(y1+1, 100)
            queries.append({
                't': random.randint(0, 10**9),
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            })

        # 计算预期答案
        expected = self.calculate_expected(stars, queries, self.c)
        
        return {
            'n': self.n,
            'q': self.q,
            'c': self.c,
            'stars': stars,
            'queries': queries,
            'expected': expected
        }

    @staticmethod
    def calculate_expected(stars, queries, c):
        """
        计算谜题实例的预期答案
        
        参数:
            stars: 星星列表
            queries: 查询列表
            c: 最大亮度值
            
        返回:
            list: 每个查询的正确答案列表
        """
        max_coord = 100
        c_plus_1 = c + 1
        
        # 初始化三维亮度数组
        a = [[[0]*(max_coord+1) for _ in range(max_coord+1)] for _ in range(c_plus_1)]
        
        # 填充基础亮度数据
        for star in stars:
            x, y, s = star['x'], star['y'], star['s']
            for t_mod in range(c_plus_1):
                a[t_mod][x][y] += (s + t_mod) % c_plus_1
        
        # 修正后的前缀和计算（与原参考代码一致）
        for t_mod in range(c_plus_1):
            # 先处理行方向前缀
            for x in range(max_coord+1):
                for y in range(1, max_coord+1):
                    a[t_mod][x][y] += a[t_mod][x][y-1]
            
            # 再处理列方向前缀
            for y in range(max_coord+1):
                for x in range(1, max_coord+1):
                    a[t_mod][x][y] += a[t_mod][x-1][y]
        
        # 处理每个查询
        results = []
        for q in queries:
            t_mod = q['t'] % c_plus_1
            x1, y1 = q['x1'], q['y1']
            x2, y2 = q['x2'], q['y2']
            
            total = (
                a[t_mod][x2][y2] 
                - a[t_mod][x1-1][y2] 
                - a[t_mod][x2][y1-1] 
                + a[t_mod][x1-1][y1-1]
            )
            results.append(total)
        
        return results

    @staticmethod
    def prompt_func(case):
        """（保持原有实现，仅优化格式）"""
        input_lines = [
            f"{case['n']} {case['q']} {case['c']}",
            *[f"{s['x']} {s['y']} {s['s']}" for s in case['stars']],
            *[f"{q['t']} {q['x1']} {q['y1']} {q['x2']} {q['y2']}" for q in case['queries']]
        ]
        
        return f"""你在观察一个动态的星空，每颗星星的亮度会周期性变化。我们需要计算不同时间点上特定区域的亮度总和。

**规则说明：**
1. 有{case['n']}颗星星，最大亮度为c={case['c']}
2. 亮度变化规律：时刻t的亮度为 (初始亮度 + t) mod (c+1)
3. 需要处理{case['q']}次矩形区域查询，区域包含边界
4. 每次查询需要输出对应区域所有星星的亮度总和

**输入数据：**
{chr(10).join(input_lines)}

**输出要求：**
{case['q']}行，每行对应查询结果的整数

请将最终答案按顺序放在[answer]和[/answer]标签之间，例如：
[answer]
42
15
7
[/answer]"""

    @staticmethod
    def extract_output(output):
        """
        改进后的答案提取逻辑，允许存在注释和空行
        """
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
            
        answer_block = matches[-1].strip()
        results = []
        for line in answer_block.split('\n'):
            line = line.strip()
            if line:
                # 提取行中的第一个整数（允许带符号）
                num_match = re.search(r'-?\d+', line)
                if num_match:
                    try:
                        results.append(int(num_match.group()))
                    except:
                        continue
        return results if results else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案正确性（增加长度校验）
        """
        expected = identity['expected']
        return solution == expected and len(solution) == identity['q']

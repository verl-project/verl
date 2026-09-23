"""# 

### 谜题描述
Let's imagine: there is a chess piece billiard ball. Its movements resemble the ones of a bishop chess piece. The only difference is that when a billiard ball hits the board's border, it can reflect from it and continue moving.

More formally, first one of four diagonal directions is chosen and the billiard ball moves in that direction. When it reaches the square located on the board's edge, the billiard ball reflects from it; it changes the direction of its movement by 90 degrees and continues moving. Specifically, having reached a corner square, the billiard ball is reflected twice and starts to move the opposite way. While it moves, the billiard ball can make an infinite number of reflections. At any square of its trajectory the billiard ball can stop and on that the move is considered completed.

<image>

It is considered that one billiard ball a beats another billiard ball b if a can reach a point where b is located.

You are suggested to find the maximal number of billiard balls, that pairwise do not beat each other and that can be positioned on a chessboard n × m in size.

Input

The first line contains two integers n and m (2 ≤ n, m ≤ 106).

Output

Print a single number, the maximum possible number of billiard balls that do not pairwise beat each other.

Please do not use the %lld specificator to read or write 64-bit numbers in C++. It is preferred to use cin (also you may use the %I64d specificator).

Examples

Input

3 4


Output

2

Input

3 3


Output

3

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m=map(int,raw_input().split())
if n<m: m,n=n,m
d = [True]*n
m = 2*m-2
nn = 2*n-2
r = 0
for i in xrange(n):
    if not d[i]: continue
    r+=1
    k=j=i
    while 1:
        d[k]=False
        j+=m
        if j>=nn: j-=nn
        if j==i: break
        k=j if j<n else nn-j        
print r
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cchessboardbilliardbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        # 参数校验与默认设置（防止非法参数）
        self.n_min = max(params.get('n_min', 2), 2)
        self.n_max = min(params.get('n_max', 100), 10**6)
        self.m_min = max(params.get('m_min', 2), 2)
        self.m_max = min(params.get('m_max', 100), 10**6)
        # 确保生成参数的有效性
        if self.n_min > self.n_max or self.m_min > self.m_max:
            raise ValueError("Invalid parameter range")
    
    def case_generator(self):
        """生成符合规范的有效棋盘尺寸"""
        while True:
            a = random.randint(self.n_min, self.n_max)
            b = random.randint(self.m_min, self.m_max)
            # 不进行预交换，保持原始生成顺序
            if 2 <= a <= 10**6 and 2 <= b <= 10**6:
                return {'n': a, 'm': b}
    
    @staticmethod
    def prompt_func(question_case):
        n, m = question_case['n'], question_case['m']
        return f"""## 台球布局问题

你正在设计一个棋盘游戏。棋盘尺寸为 {n} 行 × {m} 列。需要放置尽可能多的台球，满足：

**移动规则**：
- 台球按国际象棋象的斜对角方式移动
- 碰到边缘时以90度反射，角落会反射两次（路径反转）
- 移动轨迹无限延伸，可在任意位置停止

**兼容条件**：
- 两个台球如果存在可达路径即为不兼容
- 需要找到最大兼容台球数

**输入格式**  
两个整数 n m (2 ≤ n, m ≤ 1e6)

**输出格式**  
单一整数答案，用[answer]标签包裹，例如：[answer]5[/answer]

**当前测试案例**  
n = {n}, m = {m}

请计算正确结果："""

    @staticmethod
    def extract_output(output):
        """强化答案提取鲁棒性"""
        matches = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            output.replace(' ', '').lower(),
            re.DOTALL
        )
        if not matches:
            return None
        last_val = matches[-1].strip()
        try:
            return int(round(float(last_val)))  # 兼容浮点格式
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """完全遵循参考算法逻辑的验证"""
        try:
            # 参数交换逻辑修正
            n, m = identity['n'], identity['m']
            if n < m:
                n, m = m, n
            
            # 初始化参数
            d = [True] * n
            transformed_m = 2 * m - 2
            nn = 2 * n - 2
            result = 0
            
            # 完全复制参考算法逻辑
            for i in range(n):
                if not d[i]:
                    continue
                result += 1
                j = k = i
                while True:
                    d[k] = False
                    j += transformed_m
                    if j >= nn:
                        j -= nn
                    if j == i:
                        break
                    k = j if j < n else nn - j
            
            return solution == result
        except Exception as e:
            return False

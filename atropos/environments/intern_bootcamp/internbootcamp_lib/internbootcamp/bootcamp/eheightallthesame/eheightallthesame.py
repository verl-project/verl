"""# 

### 谜题描述
Alice has got addicted to a game called Sirtet recently.

In Sirtet, player is given an n × m grid. Initially a_{i,j} cubes are stacked up in the cell (i,j). Two cells are called adjacent if they share a side. Player can perform the following operations: 

  * stack up one cube in two adjacent cells; 
  * stack up two cubes in one cell. 



Cubes mentioned above are identical in height.

Here is an illustration of the game. States on the right are obtained by performing one of the above operations on the state on the left, and grey cubes are added due to the operation.

<image>

Player's goal is to make the height of all cells the same (i.e. so that each cell has the same number of cubes in it) using above operations. 

Alice, however, has found out that on some starting grids she may never reach the goal no matter what strategy she uses. Thus, she is wondering the number of initial grids such that 

  * L ≤ a_{i,j} ≤ R for all 1 ≤ i ≤ n, 1 ≤ j ≤ m; 
  * player can reach the goal using above operations. 



Please help Alice with it. Notice that the answer might be large, please output the desired value modulo 998,244,353.

Input

The only line contains four integers n, m, L and R (1≤ n,m,L,R ≤ 10^9, L ≤ R, n ⋅ m ≥ 2).

Output

Output one integer, representing the desired answer modulo 998,244,353.

Examples

Input


2 2 1 1


Output


1


Input


1 2 1 2


Output


2

Note

In the first sample, the only initial grid that satisfies the requirements is a_{1,1}=a_{2,1}=a_{1,2}=a_{2,2}=1. Thus the answer should be 1.

In the second sample, initial grids that satisfy the requirements are a_{1,1}=a_{1,2}=1 and a_{1,1}=a_{1,2}=2. Thus the answer should be 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
mod=998244353
n,m,l,r=map(int,raw_input().split())
if n*m%2:
    print pow(r-l+1,n*m,mod)
else:
    n1=(r-l+1)/2
    n2=(r-l+1)-n1
    ans=(pow(n1+n2,n*m,mod)+pow(n2-n1,n*m,mod))%mod
    ans=(ans*499122177)%mod
    print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Eheightallthesamebootcamp(Basebootcamp):
    def __init__(self, **params):
        # 允许生成n=1或m=1但保证n*m>=2的默认参数
        self.n_range = params.get('n_range', (1, 5))  
        self.m_range = params.get('m_range', (1, 5))
        self.L_min = params.get('L_min', 0)
        self.L_max = params.get('L_max', 5)
        self.R_min = params.get('R_min', 0)
        self.R_max = params.get('R_max', 10)
    
    def case_generator(self):
        # 确保生成合法网格尺寸的核心逻辑
        while True:
            n = random.randint(*self.n_range)
            m = random.randint(*self.m_range)
            if n * m >= 2:
                break
        
        # 生成合法L,R参数对
        L = random.randint(self.L_min, self.L_max)
        R = random.randint(max(L, self.R_min), self.R_max)
        
        return {
            'n': n,
            'm': m,
            'L': L,
            'R': R
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        params = {
            'n': question_case['n'],
            'm': question_case['m'],
            'L': question_case['L'],
            'R': question_case['R'],
            'cells': '×'.join([str(question_case['n']), str(question_case['m'])])
        }
        return (
            f"在Eheightallthesame游戏中，Alice面对一个{params['n']}行{params['m']}列的网格（共{params['cells']}格）。"
            f"每个单元格初始立方体数a_{{i,j}}满足{params['L']} ≤ a_{{i,j}} ≤ {params['R']}。\n\n"
            "合法操作：\n"
            "1. 选择两个相邻单元格各+1立方体\n"
            "2. 选择一个单元格+2立方体\n\n"
            "请求满足以下条件的初始配置总数（模998244353）：\n"
            "- 所有初始值在给定范围内\n"
            "- 通过操作可使所有单元格高度相同\n\n"
            "答案请用[answer]标签包裹，例如：[answer]12345[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        # 增强模式：允许数字前后有非数字字符
        matches = re.findall(
            r'\[answer\D*?(\d+)\D*?\[/answer\]', 
            output, 
            flags=re.IGNORECASE|re.DOTALL
        )
        return int(matches[-1]) % 998244353 if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        MOD = 998244353
        n, m, L, R = identity.values()
        k = n * m
        
        # 计算有效解数量的核心算法
        if k % 2 == 1:
            return solution == pow(R - L + 1, k, MOD)
        else:
            cnt = R - L + 1
            half = cnt // 2
            even_choices = half + (cnt % 2)  # 偶数元素个数
            odd_choices = half               # 奇数元素个数
            return solution == (
                (pow(even_choices + odd_choices, k, MOD) + 
                 pow(even_choices - odd_choices, k, MOD)) 
                * 499122177 % MOD  # 乘以模逆元2^-1 mod 998244353
            )

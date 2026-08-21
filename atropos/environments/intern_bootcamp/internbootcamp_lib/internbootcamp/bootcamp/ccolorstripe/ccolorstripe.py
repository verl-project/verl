"""# 

### 谜题描述
A colored stripe is represented by a horizontal row of n square cells, each cell is pained one of k colors. Your task is to repaint the minimum number of cells so that no two neighbouring cells are of the same color. You can use any color from 1 to k to repaint the cells.

Input

The first input line contains two integers n and k (1 ≤ n ≤ 5·105; 2 ≤ k ≤ 26). The second line contains n uppercase English letters. Letter \"A\" stands for the first color, letter \"B\" stands for the second color and so on. The first k English letters may be used. Each letter represents the color of the corresponding cell of the stripe.

Output

Print a single integer — the required minimum number of repaintings. In the second line print any possible variant of the repainted stripe.

Examples

Input

6 3
ABBACC


Output

2
ABCACA


Input

3 2
BBB


Output

1
BAB

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
args = raw_input().split()
n = int(args[0]); k = int(args[1])
s = list(raw_input())
if k > 2:
    s += '_'
    cnt = 0
    for i in range(1, n):
        if s[i] == s[i - 1]:
            s[i] = (set(['A','B','C']) - set([s[i - 1], s[i + 1]])).pop(); cnt += 1
    if s[0] == s[1]:
        if s[0] == 'A':
            s[0] = 'B'
        else:
            s[0] = 'A'
    print cnt
    print ''.join(s[:-1])
else:
    def diff(x, y):
        res = 0
        for i in range(len(x)):
            if x[i] != y[i]:
                res += 1
        return res
    res1 = ('AB' * (n / 2 + 1))[:n]; cnt1 = diff(s, res1)
    res2 = ('BA' * (n / 2 + 1))[:n]; cnt2 = diff(s, res2)
    if cnt1 < cnt2:
        cnt = cnt1; rs = ''.join(res1)
    else:
        cnt = cnt2; rs = ''.join(res2)
    print str(cnt) + '\n' + rs
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

def solve_min_repaint(n, k, s_str):
    if n == 0:
        return 0, ""
    
    s = list(s_str)
    if k > 2:
        modified = False
        for i in range(1, n):
            if s[i] == s[i-1]:
                available = set(string.ascii_uppercase[:k]) - {s[i-1]}
                if i < n-1:
                    available.discard(s[i+1])
                s[i] = sorted(available)[0]
                modified = True
        
        if modified and s[0] == s[1]:
            available = set(string.ascii_uppercase[:k]) - {s[1]}
            if n >= 3:
                available.discard(s[2])
            s[0] = sorted(available)[0]
        
        cnt = sum(1 for a, b in zip(s, s_str) if a != b)
        return cnt, ''.join(s)
    else:
        pattern1 = ['A' if i%2 ==0 else 'B' for i in range(n)]
        pattern2 = ['B' if i%2 ==0 else 'A' for i in range(n)]
        cnt1 = sum(c != sc for c, sc in zip(pattern1, s))
        cnt2 = sum(c != sc for c, sc in zip(pattern2, s))
        if cnt1 <= cnt2:
            return cnt1, ''.join(pattern1)
        return cnt2, ''.join(pattern2)

class Ccolorstripebootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, k_min=2, k_max=4):
        self.n_min = max(n_min, 1)
        self.n_max = n_max
        self.k_min = max(k_min, 2)
        self.k_max = min(k_max, 26)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        k = random.randint(self.k_min, self.k_max)
        
        # 生成具有非连续重复的初始字符串
        colors = string.ascii_uppercase[:k]
        if n == 1:
            original_s = random.choice(colors)
        else:
            original_s = [random.choice(colors)]
            for _ in range(1, n):
                # 保证至少有一个可能的重复
                if len(original_s) < 2 or random.random() < 0.4:
                    original_s.append(original_s[-1])
                else:
                    original_s.append(random.choice([c for c in colors if c != original_s[-1]]))
            original_s = ''.join(original_s)
        
        # 确保问题有解
        min_repaints, correct_s = solve_min_repaint(n, k, original_s)
        return {
            'n': n,
            'k': k,
            'original_s': original_s,
            'min_repaints': min_repaints,
            'correct_s': correct_s
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        original_s = question_case['original_s']
        color_desc = "、".join(string.ascii_uppercase[:k])
        return f"""作为颜色优化专家，请将{n}个单元格的颜色条纹（{original_s}）重新涂色，要求：
⒈ 相邻颜色不能相同
⒉ 只能使用{color_desc}这{k}种颜色
⒊ 修改次数最少

请按以下格式输出答案：
[answer]
修改次数
最终颜色序列
[/answer]

示例（k=3时）：
输入：6 3 ABBACC
回答：
[answer]
2
ABCACA
[/answer]"""

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        return '\n'.join(lines[:2]) if len(lines)>=2 else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
            
        try:
            lines = solution.split('\n')
            if len(lines) < 2:
                return False
            reported = int(lines[0].strip())
            result = lines[1].upper().strip()
            
            # 基础验证
            n = identity['n']
            k = identity['k']
            if len(result) != n:
                return False
            if any(c not in string.ascii_uppercase[:k] for c in result):
                return False
            
            # 相邻验证
            for i in range(n-1):
                if result[i] == result[i+1]:
                    return False
            
            # 修改次数验证
            actual = sum(1 for a, b in zip(identity['original_s'], result) if a != b)
            return actual == reported == identity['min_repaints']
        except:
            return False

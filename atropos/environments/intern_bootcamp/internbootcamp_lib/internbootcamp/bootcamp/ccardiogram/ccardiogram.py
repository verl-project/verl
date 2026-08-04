"""# 

### 谜题描述
In this problem, your task is to use ASCII graphics to paint a cardiogram. 

A cardiogram is a polyline with the following corners:

<image>

That is, a cardiogram is fully defined by a sequence of positive integers a1, a2, ..., an.

Your task is to paint a cardiogram by given sequence ai.

Input

The first line contains integer n (2 ≤ n ≤ 1000). The next line contains the sequence of integers a1, a2, ..., an (1 ≤ ai ≤ 1000). It is guaranteed that the sum of all ai doesn't exceed 1000.

Output

Print max |yi - yj| lines (where yk is the y coordinate of the k-th point of the polyline), in each line print <image> characters. Each character must equal either « / » (slash), « \ » (backslash), « » (space). The printed image must be the image of the given polyline. Please study the test samples for better understanding of how to print a cardiogram.

Note that in this problem the checker checks your answer taking spaces into consideration. Do not print any extra characters. Remember that the wrong answer to the first pretest doesn't give you a penalty.

Examples

Input

5
3 1 2 5 1


Output

     <span class=\"tex-span\"> / </span><span class=\"tex-span\">\</span>     
  <span class=\"tex-span\"> / </span><span class=\"tex-span\">\</span><span class=\"tex-span\"> / </span>  <span class=\"tex-span\">\</span>    
 <span class=\"tex-span\"> / </span>      <span class=\"tex-span\">\</span>   
<span class=\"tex-span\"> / </span>        <span class=\"tex-span\">\</span>  
          <span class=\"tex-span\">\</span><span class=\"tex-span\"> / </span>


Input

3
1 5 1


Output

<span class=\"tex-span\"> / </span><span class=\"tex-span\">\</span>     
  <span class=\"tex-span\">\</span>    
   <span class=\"tex-span\">\</span>   
    <span class=\"tex-span\">\</span>  
     <span class=\"tex-span\">\</span><span class=\"tex-span\"> / </span>

Note

Due to the technical reasons the answers for the samples cannot be copied from the statement. We've attached two text documents with the answers below.

http://assets.codeforces.com/rounds/435/1.txt

http://assets.codeforces.com/rounds/435/2.txt

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
a = map(int,raw_input().split())
L = sum(a)
h = maxh = minh = 1000
A = [[\" \"]*L for i in range(2001)]
i = 0
sign = 1
for ai in a:
    if sign == 1:
        for j in range(ai):
            A[h+j][i+j] = \"/\"
        maxh = max(maxh,h+(ai-1))
        h += ai-1
    else:
        for j in range(ai):
            A[h-j][i+j] = \"\\\"
        minh = min(minh,h-(ai-1))
        h -= ai-1
    i += ai
    sign *= -1
for hi in range(maxh,minh-1,-1):
    print \"\".join(A[hi])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ccardiogrambootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=1000, a_sum_max=1000):
        super().__init__()
        self.n_min = max(n_min, 2)  # 确保n_min ≥2
        self.n_max = min(n_max, 1000)
        self.a_sum_max = min(a_sum_max, 1000)
    
    def case_generator(self):
        while True:
            n = random.randint(self.n_min, self.n_max)
            sum_a = random.randint(n, min(self.a_sum_max, 1000))
            
            def split_integers(total, count):
                if total == 0:
                    return [0] * count
                dividers = sorted(random.sample(range(1, total + count), count - 1))
                return [d - i - 1 for i, d in enumerate([0] + dividers + [total + count])]  # 修正分割算法
    
            parts = split_integers(sum_a - n, n)
            # 确保所有分割部分非负
            if any(p < 0 for p in parts):
                continue
            a = [1 + p for p in parts]
            # 最终校验总和
            if sum(a) == sum_a and all(1 <= ai <= 1000 for ai in a):
                return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        a = question_case['a']
        sum_a = sum(a)
        a_str = ' '.join(map(str, a))
        prompt = f"""作为心电图绘制专家，请根据以下参数生成ASCII图形：

输入参数：
第一行：{n}
第二行：{a_str}

绘制规则：
1. 各线段须严格交替使用'/'和'\\'字符
2. 总行数为最高点与最低点y坐标差+1
3. 每行必须正好包含{sum_a}个字符
4. 行排列顺序为y坐标从高到低

将你的答案按如下格式包裹：
[answer]
第一行图形（例如：  /\\  ）
第二行图形（例如： /  \\ ）
...
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        # 处理转义字符并过滤空行
        solution = '\n'.join([line.replace('\\\\', '\\').strip('\r') 
                             for line in matches[-1].split('\n') 
                             if line.strip()])
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            a = identity['a']
            L = sum(a)
            if L == 0:
                return False
            
            h = 2000  # 提升初始坐标
            max_h = min_h = h
            grid = [[' ']*L for _ in range(4000)]  # 扩展画布
            
            x = 0
            direction = 1  # 1表示上升，-1表示下降
            
            for ai in a:
                if direction == 1:
                    for step in range(ai):
                        y = h + step
                        pos = x + step
                        if 0 <= y < 4000 and pos < L:
                            grid[y][pos] = '/'
                    h += ai - 1
                    max_h = max(max_h, h)
                else:
                    for step in range(ai):
                        y = h - step
                        pos = x + step
                        if 0 <= y < 4000 and pos < L:
                            grid[y][pos] = '\\'
                    h -= ai - 1
                    min_h = min(min_h, h)
                x += ai
                direction *= -1
            
            # 生成正确结果
            correct_lines = []
            for y in range(max_h, min_h - 1, -1):
                correct_lines.append(''.join(grid[y][:L]))
            
            # 处理用户答案
            user_lines = [line.ljust(L)[:L] for line in solution.split('\n') if line]
            
            # 对比每行
            return correct_lines == user_lines
        except Exception:
            return False

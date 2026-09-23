"""# 

### 谜题描述
Vasya and Petya wrote down all integers from 1 to n to play the \"powers\" game (n can be quite large; however, Vasya and Petya are not confused by this fact).

Players choose numbers in turn (Vasya chooses first). If some number x is chosen at the current turn, it is forbidden to choose x or all of its other positive integer powers (that is, x2, x3, ...) at the next turns. For instance, if the number 9 is chosen at the first turn, one cannot choose 9 or 81 later, while it is still allowed to choose 3 or 27. The one who cannot make a move loses.

Who wins if both Vasya and Petya play optimally?

Input

Input contains single integer n (1 ≤ n ≤ 109).

Output

Print the name of the winner — \"Vasya\" or \"Petya\" (without quotes).

Examples

Input

1


Output

Vasya


Input

2


Output

Petya


Input

8


Output

Petya

Note

In the first sample Vasya will choose 1 and win immediately.

In the second sample no matter which number Vasya chooses during his first turn, Petya can choose the remaining number and win.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
sg = [1,2,1,4,3,2,1,5,6,2,1,8,7,5,9,8,7,3,4,7,4,2,1,10,9,3,6,11,12]
i = 2
ans = 1
n = input()
mp = {} 
while i * i <= n:
	if i in mp :
		i += 1
		continue
	t = i
	cnt = 0
	while t <= n:
		mp[t] = 1
		t *= i
		cnt += 1
	ans ^= sg[cnt - 1]
	i += 1
res = n - i + 1
for a in mp: 
	if a >= i : 
		res -= 1
ans ^= res % 2
if ans == 0 : print 'Petya'
else : print 'Vasya'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dgamewithpowersbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10**9):
        self.n_min = n_min
        self.n_max = n_max
    
    def case_generator(self):
        # 生成多样化的测试案例，包括边界值和不同范围的值
        if self.n_max <= 100:
            n = random.randint(self.n_min, self.n_max)
        else:
            # 30% 小值，30% 中等值，40% 大值
            rand_val = random.random()
            if rand_val < 0.3:
                n = random.randint(self.n_min, 100)
            elif rand_val < 0.6:
                n = random.randint(101, 10**5)
            else:
                n = random.randint(10**6, self.n_max)
        
        # 强制加入关键边界值
        if random.random() < 0.2:  # 20% 概率强制使用边界案例
            n = random.choice([1, 2, 8])
            n = min(max(n, self.n_min), self.n_max)
        
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""Vasya和Petya正在玩数字幂游戏。给定n={n}，规则如下：

1. 两人轮流选择数字（Vasya先手）
2. 选择x后，x及其所有正整数次幂将永久禁用
3. 无法选择的玩家败北

请确定最终获胜者，并将答案用[answer]标签包裹，例如：[answer]Vasya[/answer]。"""

    @staticmethod
    def extract_output(output):
        # 增强的答案提取，支持多空格和大小写
        matches = re.findall(r'\[answer\s*](.*?)\[/answer\s*]', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip().capitalize()
        return answer if answer in {'Vasya', 'Petya'} else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 修正后的验证逻辑，包含完整的SG数组
        n = identity['n']
        sg = [
            1,2,1,4,3,2,1,5,6,2,1,8,7,5,9,8,7,3,4,7,4,2,1,10,9,3,6,
            11,12,14,  # 扩展的SG数组元素
            13, 15, 17, 16, 19, 18  # 继续扩展防止越界
        ]
        
        ans = 1
        i = 2
        mp = {}
        
        while i * i <= n:
            if i in mp:
                i += 1
                continue
            t = i
            cnt = 0
            while t <= n:
                mp[t] = 1
                t *= i
                cnt += 1
            # 安全访问SG数组
            ans ^= sg[cnt-1] if (cnt-1) < len(sg) else 0
            i += 1
        
        # 剩余数字计算
        remaining = n - (i - 1)
        for num in mp:
            if num >= i:
                remaining -= 1
        ans ^= remaining % 2
        
        correct_answer = 'Petya' if ans == 0 else 'Vasya'
        return solution == correct_answer

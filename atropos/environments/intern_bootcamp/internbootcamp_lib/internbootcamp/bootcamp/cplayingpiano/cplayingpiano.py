"""# 

### 谜题描述
Little Paul wants to learn how to play piano. He already has a melody he wants to start with. For simplicity he represented this melody as a sequence a_1, a_2, …, a_n of key numbers: the more a number is, the closer it is to the right end of the piano keyboard.

Paul is very clever and knows that the essential thing is to properly assign fingers to notes he's going to play. If he chooses an inconvenient fingering, he will then waste a lot of time trying to learn how to play the melody by these fingers and he will probably not succeed.

Let's denote the fingers of hand by numbers from 1 to 5. We call a fingering any sequence b_1, …, b_n of fingers numbers. A fingering is convenient if for all 1≤ i ≤ n - 1 the following holds:

  * if a_i < a_{i+1} then b_i < b_{i+1}, because otherwise Paul needs to take his hand off the keyboard to play the (i+1)-st note; 
  * if a_i > a_{i+1} then b_i > b_{i+1}, because of the same; 
  * if a_i = a_{i+1} then b_i≠ b_{i+1}, because using the same finger twice in a row is dumb. Please note that there is ≠, not = between b_i and b_{i+1}.



Please provide any convenient fingering or find out that there is none.

Input

The first line contains a single integer n (1 ≤ n ≤ 10^5) denoting the number of notes.

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 2⋅10^5) denoting the positions of notes on the keyboard.

Output

If there is no convenient fingering, print -1. Otherwise, print n numbers b_1, b_2, …, b_n, each from 1 to 5, denoting a convenient fingering, separated by spaces.

Examples

Input


5
1 1 4 2 2


Output


1 4 5 4 5 

Input


7
1 5 7 8 10 3 1


Output


1 2 3 4 5 4 3 

Input


19
3 3 7 9 8 8 8 8 7 7 7 7 5 3 3 3 3 8 8


Output


1 3 4 5 4 5 4 5 4 5 4 5 4 3 5 4 3 5 4 

Note

The third sample test is kinda \"Non stop\" song by Reflex.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin

rints = lambda: [int(x) for x in stdin.readline().split()]
n, a = int(input()), rints()
mem = [[j if not i else 0 for j in range(6)] for i in range(n)]
ans = [-1]

for i in range(1, n):
    for j in range(1, 6):
        for k in range(1, 6):
            if j != k and mem[i - 1][k]:
                if a[i] > a[i - 1] and j > k or a[i] < a[i - 1] and j < k or a[i] == a[i - 1]:
                    mem[i][j] = k

for i in range(1, 6):
    if mem[-1][i]:
        ix, val = n - 1, i
        ans = []

        while ix > -1:
            ans.append(val)
            val = mem[ix][val]
            ix -= 1
        break

print(' '.join(map(str, ans[::-1])))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def generate_solution(n, a):
    if n == 0:
        return -1
    
    # 动态规划表，mem[i][j]表示第i个音符使用手指j时的前驱可行手指
    mem = [{} for _ in range(n)]
    
    # 初始化第一个音符
    for j in range(1, 6):
        mem[0][j] = None  # 第一个音符可用任意手指
    
    # 填充DP表
    for i in range(1, n):
        for curr_finger in range(1, 6):
            valid_prevs = []
            for prev_finger in range(1, 6):
                if prev_finger not in mem[i-1]:
                    continue
                # 检查转移条件
                if (a[i] > a[i-1] and curr_finger > prev_finger) or \
                   (a[i] < a[i-1] and curr_finger < prev_finger) or \
                   (a[i] == a[i-1] and curr_finger != prev_finger):
                    valid_prevs.append(prev_finger)
            
            if valid_prevs:
                mem[i][curr_finger] = valid_prevs[0]  # 记录第一个可行前驱
    
    # 回溯查找可行解
    solution = []
    for last_finger in range(1, 6):
        if last_finger in mem[-1]:
            current = last_finger
            for i in reversed(range(n)):
                solution.append(current)
                current = mem[i][current]
            solution.reverse()
            return solution
    
    return -1

class Cplayingpianobootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'min_notes': 5,
            'max_notes': 20,
            'unsolvable_ratio': 0.4,
            'max_retry': 50
        }
        self.params.update(params)
    
    def case_generator(self):
        for _ in range(self.params['max_retry']):
            n = random.randint(self.params['min_notes'], self.params['max_notes'])
            
            # 生成无解案例的模式
            if random.random() < self.params['unsolvable_ratio']:
                # 模式1: 长序列相同音符
                if n >= 5:
                    a = [random.randint(1,10)] * n
                    solution = generate_solution(n, a)
                    if solution == -1:
                        return {
                            'n': n,
                            'a': a,
                            'has_solution': False
                        }
                
                # 模式2: 交替相邻相同
                a = []
                prev = random.randint(1,10)
                for _ in range(n):
                    a.append(prev)
                    prev = random.choice([prev, prev+1])
                solution = generate_solution(n, a)
                if solution == -1:
                    return {
                        'n': n,
                        'a': a,
                        'has_solution': False
                    }
            
            # 生成常规案例
            a = [random.randint(1, 2*10**5) for _ in range(n)]
            solution = generate_solution(n, a)
            if solution != -1:
                return {
                    'n': n,
                    'a': a,
                    'has_solution': True
                }
        
        # 默认返回合法案例
        a = [3, 3, 2, 2, 1]  # 确保有解的基本案例
        return {
            'n': 5,
            'a': a,
            'has_solution': generate_solution(5, a) != -1
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        a_str = ' '.join(map(str, question_case['a']))
        return f"""请为以下钢琴音符序列设计指法（手指编号1-5）：

音符序列：{a_str}

**规则要求**：
1. 当音高上升时（当前音符 < 下一音符），必须使用升序手指（当前手指 < 下一手指）
2. 当音高下降时（当前音符 > 下一音符），必须使用降序手指（当前手指 > 下一手指）
3. 相同连续音符必须使用不同手指

**输出格式要求**：
如果无解，输出：
[ANSWER]
-1
[/ANSWER]

如果有解，输出类似：
[ANSWER]
1 3 2 4 5
[/ANSWER]"""

    @staticmethod
    def extract_output(output):
        answer_sections = re.findall(r'\[ANSWER\][\s]*([\s\d\-]+)[\s]*\[/ANSWER\]', output)
        if not answer_sections:
            return None
        last_answer = answer_sections[-1].strip()
        # 处理换行和多余空格
        return ' '.join(last_answer.split())

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 处理无解情况
        if solution.strip() == '-1':
            return not identity['has_solution']
        
        # 格式校验
        try:
            fingers = list(map(int, solution.split()))
            if len(fingers) != identity['n']:
                return False
            if any(not 1 <= f <=5 for f in fingers):
                return False
        except ValueError:
            return False
        
        # 规则校验
        a = identity['a']
        for i in range(len(fingers)-1):
            current = fingers[i]
            next_f = fingers[i+1]
            
            # 相同音符检查
            if a[i] == a[i+1] and current == next_f:
                return False
            
            # 上升检查
            if a[i] < a[i+1] and current >= next_f:
                return False
            
            # 下降检查
            if a[i] > a[i+1] and current <= next_f:
                return False
        
        return identity['has_solution']

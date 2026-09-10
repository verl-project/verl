"""# 

### 谜题描述
Jon Snow now has to fight with White Walkers. He has n rangers, each of which has his own strength. Also Jon Snow has his favourite number x. Each ranger can fight with a white walker only if the strength of the white walker equals his strength. He however thinks that his rangers are weak and need to improve. Jon now thinks that if he takes the bitwise XOR of strengths of some of rangers with his favourite number x, he might get soldiers of high strength. So, he decided to do the following operation k times: 

  1. Arrange all the rangers in a straight line in the order of increasing strengths.
  2. Take the bitwise XOR (is written as <image>) of the strength of each alternate ranger with x and update it's strength.

Suppose, Jon has 5 rangers with strengths [9, 7, 11, 15, 5] and he performs the operation 1 time with x = 2. He first arranges them in the order of their strengths, [5, 7, 9, 11, 15]. Then he does the following: 

  1. The strength of first ranger is updated to <image>, i.e. 7.
  2. The strength of second ranger remains the same, i.e. 7.
  3. The strength of third ranger is updated to <image>, i.e. 11.
  4. The strength of fourth ranger remains the same, i.e. 11.
  5. The strength of fifth ranger is updated to <image>, i.e. 13.

The new strengths of the 5 rangers are [7, 7, 11, 11, 13]

Now, Jon wants to know the maximum and minimum strength of the rangers after performing the above operations k times. He wants your help for this task. Can you help him?

Input

First line consists of three integers n, k, x (1 ≤ n ≤ 105, 0 ≤ k ≤ 105, 0 ≤ x ≤ 103) — number of rangers Jon has, the number of times Jon will carry out the operation and Jon's favourite number respectively.

Second line consists of n integers representing the strengths of the rangers a1, a2, ..., an (0 ≤ ai ≤ 103).

Output

Output two integers, the maximum and the minimum strength of the rangers after performing the operation k times.

Examples

Input

5 1 2
9 7 11 15 5


Output

13 7

Input

2 100000 569
605 986


Output

986 605

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from copy import copy

ceil1 = lambda a, b: (a + b - 1) // b
rints = lambda: [int(x) for x in stdin.readline().split()]
n, k, x = rints()
a, mem = rints(), [0] * 1024
for i in a:
    mem[i] += 1

for i in range(k):
    tem, lst = [0] * 1024, 0
    for j in range(1024):
        tem[j ^ x] += ceil1(mem[j] + lst, 2) - ceil1(lst, 2)
        tem[j]+= (mem[j] + lst)// 2 - lst// 2
        lst += mem[j]
    mem = copy(tem)

mi, ma = 0, 0
for i in range(1024):
    if mem[i]:
        mi = i
        break
for i in range(1023, -1, -1):
    if mem[i]:
        ma = i
        break
print('%d %d' % (ma, mi))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Cjonsnowandhisfavouritenumberbootcamp(Basebootcamp):
    def __init__(self, max_n=100000, max_k=100000, max_x=1000, max_strength=1000):
        self.max_n = max_n
        self.max_k = max_k
        self.max_x = max_x
        self.max_strength = max_strength
    
    def case_generator(self):
        # 生成边界用例的概率提升到30%
        if random.random() < 0.3:
            n = random.choice([1, 100000, 1000])
            k = random.choice([0, 100000, 50000])
            x = random.choice([0, 1000])
            strengths = ([random.choice([0, 1000])] * n) if n > 1 else [random.randint(0, 1000)]
        else:
            n = random.randint(1, self.max_n)
            k = random.randint(0, self.max_k)
            x = random.randint(0, self.max_x)
            strengths = [random.randint(0, self.max_strength) for _ in range(n)]
        
        return {
            'n': n,
            'k': k,
            'x': x,
            'strengths': strengths
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        params = question_case
        return f"""Jon Snow需要计算{r'大写的' if params['x'] > 500 else ''}游骑兵部队的最终战力。经过{params['k']}次特殊操作后：
        
**操作规则**
1. 每次操作前按战力升序排列
2. 对奇数位(1st,3rd,5th...)的战士进行XOR运算，使用的值为{params['x']}

**初始数据**
- 战士数量: {params['n']}
- 操作次数: {params['k']}
- XOR值: {params['x']}
- 初始战力: {' '.join(map(str, params['strengths']))}

请输出最终的最大和最小战力，格式示例：
[answer]1024 0[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output)
        if not matches:
            return None
        try:
            values = list(map(int, matches[-1].strip().split()))
            return (values[0], values[1]) if len(values) == 2 else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 使用字典优化大n情况的内存占用
        counter = defaultdict(int)
        for s in identity['strengths']:
            counter[s] += 1

        x_val = identity['x']
        for _ in range(identity['k']):
            new_counter = defaultdict(int)
            accumulated = 0
            
            for key in sorted(counter.keys()):
                count = counter[key]
                if not count:
                    continue

                # 计算当前累积总数
                total = accumulated + count
                
                # 需要异或的数量
                xor_count = (total + 1) // 2 - (accumulated + 1) // 2
                new_counter[key ^ x_val] += xor_count
                
                # 普通数量
                new_counter[key] += count - xor_count
                
                accumulated = total

            counter = new_counter

        # 找到最大最小值
        valid_values = [k for k, v in counter.items() if v > 0]
        if not valid_values:
            return False
        return solution == (max(valid_values), min(valid_values))

"""# 

### 谜题描述
Vector Willman and Array Bolt are the two most famous athletes of Byteforces. They are going to compete in a race with a distance of L meters today.

<image>

Willman and Bolt have exactly the same speed, so when they compete the result is always a tie. That is a problem for the organizers because they want a winner. 

While watching previous races the organizers have noticed that Willman can perform only steps of length equal to w meters, and Bolt can perform only steps of length equal to b meters. Organizers decided to slightly change the rules of the race. Now, at the end of the racetrack there will be an abyss, and the winner will be declared the athlete, who manages to run farther from the starting point of the the racetrack (which is not the subject to change by any of the athletes). 

Note that none of the athletes can run infinitely far, as they both will at some moment of time face the point, such that only one step further will cause them to fall in the abyss. In other words, the athlete will not fall into the abyss if the total length of all his steps will be less or equal to the chosen distance L.

Since the organizers are very fair, the are going to set the length of the racetrack as an integer chosen randomly and uniformly in range from 1 to t (both are included). What is the probability that Willman and Bolt tie again today?

Input

The first line of the input contains three integers t, w and b (1 ≤ t, w, b ≤ 5·1018) — the maximum possible length of the racetrack, the length of Willman's steps and the length of Bolt's steps respectively.

Output

Print the answer to the problem as an irreducible fraction <image>. Follow the format of the samples output.

The fraction <image> (p and q are integers, and both p ≥ 0 and q > 0 holds) is called irreducible, if there is no such integer d > 1, that both p and q are divisible by d.

Examples

Input

10 3 2


Output

3/10


Input

7 1 2


Output

3/7

Note

In the first sample Willman and Bolt will tie in case 1, 6 or 7 are chosen as the length of the racetrack.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
[t,w,b]=raw_input().split(' ');
t=int(t);
w=int(w);
b=int(b);
j=w;
k=b;
l=0;
while (k):
       l=j%k;
       j=k;
       k=l;
j=(w*b)/j;
h=t%j;
j=t/j;
j+=1;
if (w<b):
    l=w;
else:
    l=b;
j=j*l;
j-=1;
if (h<l-1):
     j-=(l-1-h);
w=j;
k=t;
l=0;
while (k):
       l=j%k;
       j=k;
       k=l;
list1=[w/j,t/j];
print \"{}/{}\".format(*list1);
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from math import gcd
from bootcamp import Basebootcamp

def compute_probability(t, w, b):
    """优化边界条件处理和极值计算逻辑"""
    if t == 0:
        return (0, 1)
    
    # 计算最大公约数和最小公倍数
    gcd_val = gcd(w, b)
    lcm = (w * b) // gcd_val
    
    # 处理超大数值的溢出保护
    try:
        full_cycles = t // lcm
        remaining = t % lcm
    except:
        return (0, 1)
    
    min_step = min(w, b)
    count = (full_cycles + 1) * min_step - 1
    
    # 调整剩余部分
    if remaining < min_step - 1:
        count -= (min_step - 1 - remaining)
    
    # 结果规范化
    count = max(0, count)  # 确保非负
    total_gcd = gcd(count, t)
    
    return (count // total_gcd, t // total_gcd)

class Cthebigracebootcamp(Basebootcamp):
    def __init__(self, max_t=10**18, max_step=10**18):
        """
        参数范围支持题目要求的 1 ≤ t, w, b ≤ 5e18
        增加边界案例生成概率：
        - 50% 生成普通随机案例
        - 30% 生成互质步长案例
        - 20% 生成极值案例
        """
        self.max_t = max_t
        self.max_step = max_step
    
    def case_generator(self):
        """智能生成多类型测试案例"""
        case_type = random.choices(
            ['random', 'coprime', 'extreme'],
            weights=[0.5, 0.3, 0.2],
            k=1
        )[0]
        
        if case_type == 'coprime':
            # 生成互质步长
            w = random.randint(1, self.max_step)
            while True:
                b = random.randint(1, self.max_step)
                if gcd(w, b) == 1:
                    break
            t = random.randint(1, self.max_t)
        
        elif case_type == 'extreme':
            # 极值案例：最大参数或最小参数
            params = [
                (self.max_t, self.max_step, self.max_step),
                (1, 1, 1),
                (self.max_t, 1, self.max_step),
                (random.randint(1, 100), 1, 1)
            ]
            t, w, b = random.choice(params)
        
        else:  # random
            t = random.randint(1, self.max_t)
            w = random.randint(1, self.max_step)
            b = random.randint(1, self.max_step)
        
        return {'t': t, 'w': w, 'b': b}

    @staticmethod
    def prompt_func(question_case):
        case = question_case
        rule_desc = (
            "关键规则说明：\n"
            "1. Willman的最大行程：找到最大整数k使得k×w ≤ L\n"
            "2. Bolt的最大行程：找到最大整数m使得m×b ≤ L\n"
            "3. 平局条件：k×w = m×b\n"
            "4. 概率计算：满足条件的L数量 / 总可能性数t"
        )
        return f"""## 赛跑平局概率问题

比赛参数：
- 跑道最大长度 (t)：{case['t']}
- Willman步长 (w)：{case['w']}
- Bolt步长 (b)：{case['b']}

{rule_desc}

请计算平局概率，并以最简分数[answer]分子/分母[/answer]格式回答。例如：[answer]3/7[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强格式兼容性的正则表达式
        pattern = r'(?:\[answer\]|ANSWER:?)\s*(\d+)\s*/\s*(\d+)\s*(?:\[/answer\]|)'
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            last_p, last_q = matches[-1]
            try:
                p = int(last_p)
                q = int(last_q)
                if q > 0 and p >= 0:
                    return f"{p}/{q}"
            except ValueError:
                pass
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        try:
            p_user, q_user = map(int, solution.split('/'))
            if q_user <= 0 or p_user < 0:
                return False
        except:
            return False
        
        try:
            p_correct, q_correct = compute_probability(
                identity['t'],
                identity['w'],
                identity['b']
            )
        except:
            return False
        
        return p_user == p_correct and q_user == q_correct

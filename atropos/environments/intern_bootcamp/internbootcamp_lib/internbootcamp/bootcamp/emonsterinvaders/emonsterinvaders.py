"""# 

### 谜题描述
Ziota found a video game called \"Monster Invaders\".

Similar to every other shooting RPG game, \"Monster Invaders\" involves killing monsters and bosses with guns.

For the sake of simplicity, we only consider two different types of monsters and three different types of guns.

Namely, the two types of monsters are: 

  * a normal monster with 1 hp. 
  * a boss with 2 hp. 



And the three types of guns are: 

  * Pistol, deals 1 hp in damage to one monster, r_1 reloading time 
  * Laser gun, deals 1 hp in damage to all the monsters in the current level (including the boss), r_2 reloading time 
  * AWP, instantly kills any monster, r_3 reloading time 



The guns are initially not loaded, and the Ziota can only reload 1 gun at a time.

The levels of the game can be considered as an array a_1, a_2, …, a_n, in which the i-th stage has a_i normal monsters and 1 boss. Due to the nature of the game, Ziota cannot use the Pistol (the first type of gun) or AWP (the third type of gun) to shoot the boss before killing all of the a_i normal monsters.

If Ziota damages the boss but does not kill it immediately, he is forced to move out of the current level to an arbitrary adjacent level (adjacent levels of level i (1 < i < n) are levels i - 1 and i + 1, the only adjacent level of level 1 is level 2, the only adjacent level of level n is level n - 1). Ziota can also choose to move to an adjacent level at any time. Each move between adjacent levels are managed by portals with d teleportation time.

In order not to disrupt the space-time continuum within the game, it is strictly forbidden to reload or shoot monsters during teleportation. 

Ziota starts the game at level 1. The objective of the game is rather simple, to kill all the bosses in all the levels. He is curious about the minimum time to finish the game (assuming it takes no time to shoot the monsters with a loaded gun and Ziota has infinite ammo on all the three guns). Please help him find this value.

Input

The first line of the input contains five integers separated by single spaces: n (2 ≤ n ≤ 10^6) — the number of stages, r_1, r_2, r_3 (1 ≤ r_1 ≤ r_2 ≤ r_3 ≤ 10^9) — the reload time of the three guns respectively, d (1 ≤ d ≤ 10^9) — the time of moving between adjacent levels.

The second line of the input contains n integers separated by single spaces a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^6, 1 ≤ i ≤ n).

Output

Print one integer, the minimum time to finish the game.

Examples

Input


4 1 3 4 3
3 2 5 1


Output


34

Input


4 2 4 4 1
4 5 1 2


Output


31

Note

In the first test case, the optimal strategy is:

  * Use the pistol to kill three normal monsters and AWP to kill the boss (Total time 1⋅3+4=7) 
  * Move to stage two (Total time 7+3=10) 
  * Use the pistol twice and AWP to kill the boss (Total time 10+1⋅2+4=16) 
  * Move to stage three (Total time 16+3=19) 
  * Use the laser gun and forced to move to either stage four or two, here we move to stage four (Total time 19+3+3=25) 
  * Use the pistol once, use AWP to kill the boss (Total time 25+1⋅1+4=30) 
  * Move back to stage three (Total time 30+3=33) 
  * Kill the boss at stage three with the pistol (Total time 33+1=34) 



Note that here, we do not finish at level n, but when all the bosses are killed.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import math

if sys.subversion[0] == \"PyPy\":
    import io, atexit

    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))

    sys.stdin = io.BytesIO(sys.stdin.read())
    input = lambda: sys.stdin.readline().rstrip()

#t = int(raw_input())

# python codesforces210710.py < input.txt
#for zz in range(t):

str1 = raw_input().split(' ')
n, r1, r2, r3, d = map(int, str1)
str2 = raw_input().split(' ')
a = map(int, str2)
fs = 0
c1 = 0
c2 = 2 * d

for i in range(n):
    #print(aii)
    indcost = r3 + a[i] * r1
    aoecost = min(r2 + r1, r1 * (a[i] + 2))
    mincost = min(indcost, aoecost)
    mincost2l = min(indcost - d, aoecost)
    oto = indcost
    ott = mincost + 2 * d
    ttt = mincost + 2 * d
    tto = mincost
    #print(oto, ott, ttt, tto)
    if i < n - 1:
        c3 = min(c1 + oto, c2 + tto) + d
        c4 = min(c1 + ott, c2 + ttt) + d
    else:
        c3 = min(c1 + oto, c2 + mincost2l) + d
        c4 = min(c1 + ott, c2 + ttt) + d
    c1 = c3
    c2 = c4
    #print(c1, c2)
ans = min(c1, c2) - d
print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_min_time(n, r1, r2, r3, d, a_list):
    a = a_list
    c1 = 0
    c2 = 2 * d

    for i in range(n):
        ai = a[i]
        single_cost = r3 + ai * r1
        aoe_cost = min(r2 + r1, r1 * (ai + 2))
        min_cost = min(single_cost, aoe_cost)
        min_last = min(single_cost - d, aoe_cost)

        if i < n - 1:
            new_c1 = min(c1 + single_cost, c2 + min_cost) + d
            new_c2 = min(c1 + (min_cost + 2*d), c2 + (min_cost + 2*d)) + d
        else:
            new_c1 = min(c1 + single_cost, c2 + min_last) + d
            new_c2 = min(c1 + (min_cost + 2*d), c2 + (min_cost + 2*d)) + d
        
        c1, c2 = new_c1, new_c2

    return min(c1, c2) - d

class Emonsterinvadersbootcamp(Basebootcamp):
    def __init__(self, **params):
        default_params = {
            'min_n': 2,
            'max_n': 1000,        # 降低默认最大关卡数
            'min_a': 1,
            'max_a': 1000,        # 降低默认怪物数量
            'time_scale': 1e3,    # 时间参数规模控制
            'seed': None
        }
        default_params.update(params)
        self.params = default_params
        random.seed(self.params['seed'])

    def case_generator(self):
        # 增强边界条件生成能力
        n = self._generate_n()
        r1, r2, r3 = self._generate_weapon_times()
        d = random.randint(1, self.params['time_scale'])
        a = self._generate_monsters(n)
        
        expected = compute_min_time(n, r1, r2, r3, d, a)
        return {
            'n': n, 'r1': r1, 'r2': r2, 'r3': r3, 'd': d,
            'a': a, 'expected': expected
        }

    def _generate_n(self):
        # 10%概率生成边界值
        if random.random() < 0.1:
            return random.choice([2, self.params['max_n']])
        return random.randint(self.params['min_n'], self.params['max_n'])

    def _generate_weapon_times(self):
        # 生成武器时间参数，包含边界情况
        variants = [
            (1, 1, 1),        # 全相同
            (1, 2, 2),        # r2 = r3
            (1, 1, 2),        # r1 = r2
            (1, 3, 5)         # 正常情况
        ]
        if random.random() < 0.3:
            return random.choice(variants)
        
        r1 = random.randint(1, self.params['time_scale'])
        r2 = random.randint(r1, self.params['time_scale'])
        r3 = random.randint(r2, self.params['time_scale'])
        return r1, r2, r3

    def _generate_monsters(self, n):
        # 生成怪物配置，包含全1和最大值
        if random.random() < 0.2:
            return [1] * n
        if random.random() < 0.2:
            return [self.params['max_a']] * n
        return [random.randint(self.params['min_a'], self.params['max_a']) for _ in range(n)]

    @staticmethod
    def prompt_func(question_case):
        return f"""## Monster Invaders 最小时间计算
你需要为以下游戏配置计算最优通关时间：

### 关卡配置
- 总关卡数：{question_case['n']}
- 移动时间：{question_case['d']}
- 各关卡普通怪物数量：{' '.join(map(str, question_case['a']))}

### 武器参数
1. 手枪：装填时间 {question_case['r1']}
2. 激光枪：装填时间 {question_case['r2']}
3. AWP：装填时间 {question_case['r3']}

### 关键规则
1. 必须消灭所有普通怪物后才能用手枪/AWP攻击BOSS
2. 未击杀BOSS时强制传送且移动时不能操作
3. 每次仅可装填一把武器

请输出精确计算结果并包裹在[answer][/answer]标签内。示例：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强模式匹配鲁棒性
        patterns = [
            r'\[answer\][\n\s]*(\d+)[\n\s]*\[/answer\]',  # 标准格式
            r'answer[\s:]*(\d+)',                         # 简单声明格式
            r'最终结果\D*(\d+)',                           # 中文描述
            r'\b\d+\b(?=[^\.]*$)'                         # 最后一个独立数字
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    return int(matches[-1].strip())
                except (ValueError, TypeError):
                    continue
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['expected']
        except:
            return False

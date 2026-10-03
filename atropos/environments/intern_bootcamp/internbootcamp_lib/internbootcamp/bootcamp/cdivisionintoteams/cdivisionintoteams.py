"""# 

### 谜题描述
Petya loves football very much, especially when his parents aren't home. Each morning he comes to the yard, gathers his friends and they play all day. From time to time they have a break to have some food or do some chores (for example, water the flowers).

The key in football is to divide into teams fairly before the game begins. There are n boys playing football in the yard (including Petya), each boy's football playing skill is expressed with a non-negative characteristic ai (the larger it is, the better the boy plays). 

Let's denote the number of players in the first team as x, the number of players in the second team as y, the individual numbers of boys who play for the first team as pi and the individual numbers of boys who play for the second team as qi. Division n boys into two teams is considered fair if three conditions are fulfilled:

  * Each boy plays for exactly one team (x + y = n). 
  * The sizes of teams differ in no more than one (|x - y| ≤ 1). 
  * The total football playing skills for two teams differ in no more than by the value of skill the best player in the yard has. More formally: <image>



Your task is to help guys divide into two teams fairly. It is guaranteed that a fair division into two teams always exists.

Input

The first line contains the only integer n (2 ≤ n ≤ 105) which represents the number of guys in the yard. The next line contains n positive space-separated integers, ai (1 ≤ ai ≤ 104), the i-th number represents the i-th boy's playing skills. 

Output

On the first line print an integer x — the number of boys playing for the first team. On the second line print x integers — the individual numbers of boys playing for the first team. On the third line print an integer y — the number of boys playing for the second team, on the fourth line print y integers — the individual numbers of boys playing for the second team. Don't forget that you should fulfil all three conditions: x + y = n, |x - y| ≤ 1, and the condition that limits the total skills.

If there are multiple ways to solve the problem, print any of them.

The boys are numbered starting from one in the order in which their skills are given in the input data. You are allowed to print individual numbers of boys who belong to the same team in any order.

Examples

Input

3
1 2 1


Output

2
1 2 
1
3 


Input

5
2 3 3 1 1


Output

3
4 1 3 
2
5 2 

Note

Let's consider the first sample test. There we send the first and the second boy to the first team and the third boy to the second team. Let's check all three conditions of a fair division. The first limitation is fulfilled (all boys play), the second limitation on the sizes of groups (|2 - 1| = 1 ≤ 1) is fulfilled, the third limitation on the difference in skills ((2 + 1) - (1) = 2 ≤ 2) is fulfilled.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from operator import attrgetter, itemgetter, methodcaller, add
from collections import OrderedDict, Counter
from decimal import Decimal, getcontext
getcontext().prec = 7

def r(): return raw_input().strip()
def ri(): return int(r())
def riv(): return map(int, r().split())

def ans(f, s):
    print len(f)
    print \" \".join(map(lambda a: str(a[0]+1), f))
    print len(s)
    print \" \".join(map(lambda a: str(a[0]+1), s))

def main():
    n = ri()
    a = riv()
    
    m = max(a)
    delta = n/2
    a = list(enumerate(a))
    first, second = a[:delta], a[delta:]
    f, s = sum(map(itemgetter(1), first)), sum(map(itemgetter(1), second))
    if abs(f - s) <= m:
        ans(first, second)
        exit(0)

    first = list(sorted(first, key=itemgetter(1)))
    second = list(sorted(second, key=itemgetter(1)))
    if f > s: 
        f, s = s, f
        first, second = second, first

    while abs(s - f) >= m:
        switch_f = first[0]
        switch_s = second[-1]
        del first[0]
        first.append(switch_s)

        del second[-1]
        second.insert(0, switch_f)
        
        diff = switch_s[1] - switch_f[1]
        f += diff
        s -= diff
    
    ans(first, second)
    
if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cdivisionintoteamsbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, max_skill=10000):
        """
        初始化足球分队谜题训练场参数
        
        参数:
            min_n (int): 最小参与人数，默认2
            max_n (int): 最大参与人数，默认10
            max_skill (int): 最大技能值，默认10000
        """
        self.min_n = min_n
        self.max_n = max_n
        self.max_skill = max_skill
    
    def case_generator(self):
        """
        生成随机有效的谜题实例
        
        返回:
            dict: 包含n和a数组的可序列化字典
        """
        n = random.randint(self.min_n, self.max_n)
        a = [random.randint(1, self.max_skill) for _ in range(n)]
        return {
            'n': n,
            'a': a
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        将谜题实例转换为格式化的自然语言问题
        
        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 包含完整问题描述和格式要求的字符串
        """
        n = question_case['n']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        return f"""你是青少年足球比赛的负责人，需要公平地将孩子们分成两队。请仔细阅读以下规则并完成任务：

**分队规则**
1. 全员参与：所有{n}个孩子必须加入且只能加入一个队伍
2. 人数均衡：两队人数差不能超过1人（例如{n}人时可分{ (n+1)//2 }和{ n//2 }人）
3. 技能平衡：两队总技能差不能超过所有孩子中的最大技能值（当前最大技能值为{max(a)}）

**输入数据**
- 第一行：{n}（总人数）
- 第二行：{a_str}（各孩子技能值，按输入顺序编号为1~{n}）

**输出格式要求**
请严格按以下顺序输出：
1. 第一队人数
2. 第一队成员编号（空格分隔，任意顺序）
3. 第二队人数 
4. 第二队成员编号（空格分隔，任意顺序）

**示例格式**
[answer]
2
1 2
1
3
[/answer]

请将符合要求的最终答案放在[answer]标签之间："""

    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取规范化的答案结构
        
        参数:
            output: 模型完整输出文本
            
        返回:
            dict/None: 提取的答案结构，格式错误时返回None
        """
        # 定位最后一个answer标签内容
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        # 取最后一个answer块并清理空白
        answer_lines = [l.strip() for l in answer_blocks[-1].strip().split('\n') if l.strip()]
        
        try:
            # 解析四行结构
            if len(answer_lines) < 4:
                return None
                
            x = int(answer_lines[0])
            first_team = list(map(int, answer_lines[1].split()))
            y = int(answer_lines[2])
            second_team = list(map(int, answer_lines[3].split()))
            
            # 校验人数一致性
            if x != len(first_team) or y != len(second_team):
                return None
                
            return {
                'x': x,
                'first_team': first_team,
                'y': y,
                'second_team': second_team
            }
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案是否符合所有公平分队条件
        
        参数:
            solution: 从输出中提取的答案结构
            identity: 原始谜题实例
            
        返回:
            bool: 答案是否完全正确
        """
        n = identity['n']
        a = identity['a']
        max_skill = max(a)
        
        # 基础类型校验
        if not solution or not all(k in solution for k in ('x', 'y', 'first_team', 'second_team')):
            return False
            
        x = solution['x']
        y = solution['y']
        ft = solution['first_team']
        st = solution['second_team']

        # 条件1: 人数总和正确
        if x + y != n:
            return False
            
        # 条件2: 人数差合法
        if abs(x - y) > 1:
            return False
            
        # 条件3: 成员唯一性校验
        all_members = set(ft) | set(st)
        if len(all_members) != n or any(not (1 <= m <= n) for m in all_members):
            return False
            
        # 条件4: 技能差值校验
        sum_ft = sum(a[i-1] for i in ft)
        sum_st = sum(a[i-1] for i in st)
        if abs(sum_ft - sum_st) > max_skill:
            return False
            
        return True

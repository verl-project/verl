"""# 

### 谜题描述
You are preparing for an exam on scheduling theory. The exam will last for exactly T milliseconds and will consist of n problems. You can either solve problem i in exactly ti milliseconds or ignore it and spend no time. You don't need time to rest after solving a problem, either.

Unfortunately, your teacher considers some of the problems too easy for you. Thus, he assigned an integer ai to every problem i meaning that the problem i can bring you a point to the final score only in case you have solved no more than ai problems overall (including problem i).

Formally, suppose you solve problems p1, p2, ..., pk during the exam. Then, your final score s will be equal to the number of values of j between 1 and k such that k ≤ apj.

You have guessed that the real first problem of the exam is already in front of you. Therefore, you want to choose a set of problems to solve during the exam maximizing your final score in advance. Don't forget that the exam is limited in time, and you must have enough time to solve all chosen problems. If there exist different sets of problems leading to the maximum final score, any of them will do.

Input

The first line contains two integers n and T (1 ≤ n ≤ 2·105; 1 ≤ T ≤ 109) — the number of problems in the exam and the length of the exam in milliseconds, respectively.

Each of the next n lines contains two integers ai and ti (1 ≤ ai ≤ n; 1 ≤ ti ≤ 104). The problems are numbered from 1 to n.

Output

In the first line, output a single integer s — your maximum possible final score.

In the second line, output a single integer k (0 ≤ k ≤ n) — the number of problems you should solve.

In the third line, output k distinct integers p1, p2, ..., pk (1 ≤ pi ≤ n) — the indexes of problems you should solve, in any order.

If there are several optimal sets of problems, you may output any of them.

Examples

Input

5 300
3 100
4 150
4 80
2 90
2 300


Output

2
3
3 1 4


Input

2 100
1 787
2 788


Output

0
0



Input

2 100
2 42
2 58


Output

2
2
1 2

Note

In the first example, you should solve problems 3, 1, and 4. In this case you'll spend 80 + 100 + 90 = 270 milliseconds, falling within the length of the exam, 300 milliseconds (and even leaving yourself 30 milliseconds to have a rest). Problems 3 and 1 will bring you a point each, while problem 4 won't. You'll score two points.

In the second example, the length of the exam is catastrophically not enough to solve even a single problem.

In the third example, you have just enough time to solve both problems in 42 + 58 = 100 milliseconds and hand your solutions to the teacher with a smile.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO

sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))

input = lambda: sys.stdin.readline().rstrip('\r\n')

n,T = map(int,input().split(\" \"))
ai  = []
for i in range(n):
    a,b = map(int,input().split(\" \"))
    ai.append((b,-a,i+1))

ai.sort()
ans = []
low,high = 0,n+1
while low < high:

    mid = (low+high)/2
    cnt,t = 0,0
    for i in range(n):
        if -ai[i][1] >= mid and t+ai[i][0]<=T:
            cnt += 1
            t+= ai[i][0]
        elif t+ai[i][0]>T:
            break
        if cnt >= mid:
            break
    if cnt >= mid :
        ans.append(mid)
        low = mid+1
    else:
        high = mid


v= max(ans)
print  v
print v
cnt = 0
ret = []
for i in range(n):
    if cnt >= v:
        break
    if -ai[i][1] >= v:
        cnt += 1
        ret.append(ai[i][2])

print \" \".join(map(str,ret))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

class Dtooeasyproblemsbootcamp(Basebootcamp):
    def __init__(self, n=5, ti_min=1, ti_max=100):
        self.n = n
        self.ti_min = ti_min
        self.ti_max = ti_max
    
    def case_generator(self):
        """生成保证有解的谜题实例"""
        n = self.n
        if n == 0:  # 处理边界情况
            return {'n': 0, 'T': 0, 'problems': []}
        
        # 确定最大可能的k值
        k = random.randint(1, n)
        
        # 生成k个有效问题
        valid = []
        total_time = 0
        for _ in range(k):
            a = random.randint(k, n)  # 保证a_i >= k
            t = random.randint(self.ti_min, self.ti_max)
            valid.append((a, t))
            total_time += t
        
        # 生成无效问题（a_i <k 或时间过大）
        invalid = []
        for _ in range(n - k):
            a = random.randint(1, k-1) if k > 1 else 1  # 确保a_i <k
            t = random.randint(total_time+1, total_time*2)  # 时间无法被选中
            invalid.append((a, t))
        
        # 合并并打乱问题顺序
        all_problems = valid + invalid
        random.shuffle(all_problems)
        
        return {
            'n': n,
            'T': total_time,
            'problems': all_problems
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """生成符合格式的问题描述"""
        problem_lines = "\n".join(
            f"{i+1} {a} {t}" 
            for i, (a, t) in enumerate(question_case['problems'])
        )
        return f"""You are preparing for a scheduling theory exam lasting {question_case['T']}ms with {question_case['n']} problems. Solve problems to maximize your score. Each problem i gives a point only if solved ≤a_i total problems.

Input:
{question_case['n']} {question_case['T']}
{problem_lines}

Output your answer as:

[answer]
s
k
p1 p2 ... pk
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        """提取最后一个[answer]块内的答案"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) < 2:  # 至少需要s和k两行
            return None
        
        try:
            s = int(lines[0])
            k = int(lines[1])
            if k == 0:
                return {'s': s, 'k': k, 'p_list': []}
            p_list = list(map(int, lines[2].split())) if len(lines)>=3 else []
            if len(p_list) != k:
                return None
            return {'s': s, 'k': k, 'p_list': p_list}
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """核心验证逻辑"""
        # 获取标准答案
        try:
            correct = cls.solve_problem(
                identity['n'], 
                identity['T'], 
                identity['problems']
            )
        except:
            return False
        
        # 基本情况验证
        if solution['s'] != correct['s'] or solution['k'] != correct['k']:
            return False
        
        # 检查问题索引有效性
        p_list = solution['p_list']
        problem_dict = {i+1: (a, t) for i, (a, t) in enumerate(identity['problems'])}
        for p in p_list:
            if p not in problem_dict:
                return False
        
        # 验证总时间约束
        total_time = sum(problem_dict[p][1] for p in p_list)
        if total_time > identity['T']:
            return False
        
        # 验证每个问题的a_i >=k（k为正确的s值）
        k_max = correct['s']
        for p in p_list:
            if problem_dict[p][0] < k_max:
                return False
        
        return True

    @staticmethod
    def solve_problem(n, T, problems):
        """参考解法（带索引处理）"""
        indexed_problems = [(i+1, a, t) for i, (a, t) in enumerate(problems)]
        # 按ti升序，a降序排序（优先选择耗时少且a高的）
        sorted_problems = sorted(indexed_problems, key=lambda x: (x[2], -x[1]))
        
        # 二分查找最大k
        low, high = 0, n
        best_k = 0
        while low <= high:
            mid = (low + high) // 2
            cnt, total = 0, 0
            for p in sorted_problems:
                if p[1] >= mid and total + p[2] <= T:
                    cnt += 1
                    total += p[2]
                if cnt >= mid:
                    break
            if cnt >= mid:
                best_k = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # 收集答案索引
        result = []
        total = 0
        for p in sorted_problems:
            if p[1] >= best_k and total + p[2] <= T and len(result) < best_k:
                result.append(p[0])
                total += p[2]
        
        return {
            's': best_k,
            'k': best_k,
            'p_list': result
        }

"""# 

### 谜题描述
There are n children numbered from 1 to n in a kindergarten. Kindergarten teacher gave a_i (1 ≤ a_i ≤ n) candies to the i-th child. Children were seated in a row in order from 1 to n from left to right and started eating candies. 

While the i-th child was eating candies, he calculated two numbers l_i and r_i — the number of children seating to the left of him that got more candies than he and the number of children seating to the right of him that got more candies than he, respectively.

Formally, l_i is the number of indices j (1 ≤ j < i), such that a_i < a_j and r_i is the number of indices j (i < j ≤ n), such that a_i < a_j.

Each child told to the kindergarten teacher the numbers l_i and r_i that he calculated. Unfortunately, she forgot how many candies she has given to each child. So, she asks you for help: given the arrays l and r determine whether she could have given the candies to the children such that all children correctly calculated their values l_i and r_i, or some of them have definitely made a mistake. If it was possible, find any way how she could have done it.

Input

On the first line there is a single integer n (1 ≤ n ≤ 1000) — the number of children in the kindergarten.

On the next line there are n integers l_1, l_2, …, l_n (0 ≤ l_i ≤ n), separated by spaces.

On the next line, there are n integer numbers r_1, r_2, …, r_n (0 ≤ r_i ≤ n), separated by spaces.

Output

If there is no way to distribute the candies to the children so that all of them calculated their numbers correctly, print «NO» (without quotes).

Otherwise, print «YES» (without quotes) on the first line. On the next line, print n integers a_1, a_2, …, a_n, separated by spaces — the numbers of candies the children 1, 2, …, n received, respectively. Note that some of these numbers can be equal, but all numbers should satisfy the condition 1 ≤ a_i ≤ n. The number of children seating to the left of the i-th child that got more candies than he should be equal to l_i and the number of children seating to the right of the i-th child that got more candies than he should be equal to r_i. If there is more than one solution, find any of them.

Examples

Input

5
0 0 1 1 2
2 0 1 0 0


Output

YES
1 3 1 2 1


Input

4
0 0 2 0
1 1 1 1


Output

NO


Input

3
0 0 0
0 0 0


Output

YES
1 1 1

Note

In the first example, if the teacher distributed 1, 3, 1, 2, 1 candies to 1-st, 2-nd, 3-rd, 4-th, 5-th child, respectively, then all the values calculated by the children are correct. For example, the 5-th child was given 1 candy, to the left of him 2 children were given 1 candy, 1 child was given 2 candies and 1 child — 3 candies, so there are 2 children to the left of him that were given more candies than him.

In the second example it is impossible to distribute the candies, because the 4-th child made a mistake in calculating the value of r_4, because there are no children to the right of him, so r_4 should be equal to 0.

In the last example all children may have got the same number of candies, that's why all the numbers are 0. Note that each child should receive at least one candy.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
n = int(sys.stdin.readline())
l = map(int,sys.stdin.readline().split(' '))
r = map(int,sys.stdin.readline().split(' '))

result=[0]*n
num=n
flag = 0
flag2 = 1
for j in range(n):
    count = 0
    indices = []
    for i in range(n):
        if l[i] == 0 and r[i] == 0:
            count += 1
            l[i] = '.'
            r[i] = '.'
            indices.append(i)

    if count == 0:
        flag = 1
        break

    for i in indices:
        result[i] = num
        
    num -= 1

    for index in indices:
        for i in range(n):
            if i < index and r[i] != '.':
                r[i] -= 1
                
            if i > index and l[i] != '.':
                l[i] -= 1
            if(r[i] < 0 or l[i] < 0):
                flag = 1
                break

        if flag == 1:
            break
    flag2 = 0
    for i in range(n):
        if not (l[i] == '.' and r[i] == '.'):
            flag2 = 1
            break
    
    if flag2 == 0:
        break
    if flag == 1:
        break

if flag == 1:
    print \"NO\"        

if flag == 0 or flag2 == 0:
    print \"YES\"
    print ' '.join(map(str,result))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

class Ccandiesdistributionbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=15):
        self.min_n = min_n
        self.max_n = max_n
        
    def case_generator(self):
        case_type = random.choice([
            'valid_standard', 
            'valid_duplicates',
            'invalid_boundary',
            'invalid_overflow',
            'invalid_sum'
        ])
        
        if case_type.startswith('valid'):
            return self.generate_valid_case(case_type)
        return self.generate_invalid_case(case_type)

    def generate_valid_case(self, case_type):
        n = random.randint(self.min_n, self.max_n)
        a = []
        
        # 生成策略优化
        if case_type == 'valid_standard':
            # 使用拓扑排序生成合法案例
            graph = [[] for _ in range(n)]
            in_degree = [0]*n
            q = deque()
            
            # 构造约束关系
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.3:
                        graph[i].append(j)
                        in_degree[j] += 1
                    else:
                        graph[j].append(i) 
                        in_degree[i] += 1
                        
            # 拓扑排序生成合法值
            while q:
                u = q.popleft()
                a.append(random.randint(1, n))
                for v in graph[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        q.append(v)
            a += [random.randint(1, n) for _ in range(n - len(a))]
        else:  # valid_duplicates
            base = random.randint(1, n//2)
            a = [base + i % 3 for i in range(n)]
            random.shuffle(a)

        # 计算合法约束
        l = [sum(a[j] > a[i] for j in range(i)) for i in range(n)]
        r = [sum(a[j] > a[i] for j in range(i+1, n)) for i in range(n)]
        
        return {
            'n': n,
            'l': l,
            'r': r,
            'solvable': True,
            'type': case_type
        }

    def generate_invalid_case(self, case_type):
        n = random.randint(self.min_n, self.max_n)
        l = [0]*n
        r = [0]*n
        
        if case_type == 'invalid_boundary':
            # 边界条件无效：首位儿童左边有人，末位儿童右边有人
            targets = [0, n-1] if n > 1 else [0]
            for i in targets:
                if i == 0:
                    l[i] = random.randint(1, 3)
                else:
                    r[i] = random.randint(1, 3)
        
        elif case_type == 'invalid_overflow':
            # 数值超限：单个值超过理论最大值
            i = random.randint(0, n-1)
            max_possible = i if i < n-1 else 0
            l[i] = max_possible + random.randint(1, 2)
        
        elif case_type == 'invalid_sum':
            # 总和矛盾：l_i + r_i > 可能的最大值
            i = random.randint(0, n-1)
            max_total = (n - 1) - (i + (n - i - 1))
            if max_total < 0: max_total = 0
            current_sum = random.randint(max_total + 1, max_total + 3)
            l[i] = current_sum // 2
            r[i] = current_sum - l[i]
        
        return {
            'n': n,
            'l': l,
            'r': r,
            'solvable': False,
            'type': case_type
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        def format_array(arr):
            return ' '.join(f'\033[34m{a}\033[0m' for a in arr)  # 使用ANSI颜色增强显示
        
        return f"""## 幼儿园糖果分配验证问题（难度：★★★☆☆）

### 问题描述
幼儿园老师记录了{n}个孩子的糖果分配观察结果，但不确定孩子们是否计算正确。每个孩子i报告了两个数值：
- l_i（左侧糖果数比自己多的人数）：{format_array(question_case['l'])}
- r_i（右侧糖果数比自己多的人数）：{format_array(question_case['r'])}

### 验证要求
1. 判断是否存在满足以下条件的糖果分配方案：
   - 每个孩子获得1~{n}颗糖果
   - 每个孩子的l_i和r_i计算准确
2. 若存在，给出任意可行方案；否则说明无解

### 输出格式
将最终答案用[answer]标签包裹，示例如下：
[answer]
YES
2 3 1 2
[/answer]
或
[answer]
NO
[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        from itertools import dropwhile
        
        # 清理特殊字符
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)  # 移除ANSI转义码
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', clean_output, re.DOTALL)
        
        if not matches:
            return None
        
        content = matches[-1].strip()
        lines = list(filter(None, map(str.strip, content.splitlines())))
        
        if not lines:
            return None
        
        status = lines[0].upper()
        if status not in ('YES', 'NO'):
            return None
        
        result = {'status': status}
        if status == 'YES' and len(lines) >= 2:
            try:
                result['a'] = list(map(int, re.findall(r'\d+', lines[1])))
            except ValueError:
                return None
        
        return result

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        expected_solvable = identity['solvable']
        
        # 验证否定结论
        if solution['status'] == 'NO':
            return not expected_solvable
        
        # 验证肯定结论
        if solution.get('status') != 'YES' or 'a' not in solution:
            return False
        
        a = solution['a']
        n = identity['n']
        l = identity['l']
        r = identity['r']
        
        # 基础校验
        if len(a) != n or any(not 1 <= x <= n for x in a):
            return False
        
        # 精确验证
        try:
            for i in range(n):
                actual_l = sum(a[j] > a[i] for j in range(i))
                actual_r = sum(a[j] > a[i] for j in range(i+1, n))
                
                if actual_l != l[i] or actual_r != r[i]:
                    return False
        except IndexError:
            return False
        
        return True

"""# 

### 谜题描述
Now you can take online courses in the Berland State University! Polycarp needs to pass k main online courses of his specialty to get a diploma. In total n courses are availiable for the passage.

The situation is complicated by the dependence of online courses, for each course there is a list of those that must be passed before starting this online course (the list can be empty, it means that there is no limitation).

Help Polycarp to pass the least number of courses in total to get the specialty (it means to pass all main and necessary courses). Write a program which prints the order of courses. 

Polycarp passes courses consistently, he starts the next course when he finishes the previous one. Each course can't be passed more than once. 

Input

The first line contains n and k (1 ≤ k ≤ n ≤ 105) — the number of online-courses and the number of main courses of Polycarp's specialty. 

The second line contains k distinct integers from 1 to n — numbers of main online-courses of Polycarp's specialty. 

Then n lines follow, each of them describes the next course: the i-th of them corresponds to the course i. Each line starts from the integer ti (0 ≤ ti ≤ n - 1) — the number of courses on which the i-th depends. Then there follows the sequence of ti distinct integers from 1 to n — numbers of courses in random order, on which the i-th depends. It is guaranteed that no course can depend on itself. 

It is guaranteed that the sum of all values ti doesn't exceed 105. 

Output

Print -1, if there is no the way to get a specialty. 

Otherwise, in the first line print the integer m — the minimum number of online-courses which it is necessary to pass to get a specialty. In the second line print m distinct integers — numbers of courses which it is necessary to pass in the chronological order of their passage. If there are several answers it is allowed to print any of them.

Examples

Input

6 2
5 3
0
0
0
2 2 1
1 4
1 5


Output

5
1 2 3 4 5 


Input

9 3
3 9 5
0
0
3 9 4 5
0
0
1 8
1 6
1 2
2 1 2


Output

6
1 2 9 4 5 3 


Input

3 3
1 2 3
1 2
1 3
1 1


Output

-1

Note

In the first test firstly you can take courses number 1 and 2, after that you can take the course number 4, then you can take the course number 5, which is the main. After that you have to take only the course number 3, which is the last not passed main course. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

all_, main_ = [int(x) for x in raw_input().split()]
to_solve = [int(x) for x in raw_input().split()]
all_deps = [[int(x) for x in raw_input().split()]
            for _ in range(all_)
            ]
import time
started = time.time()
solved = set()
ordered_answer = []
to_solve_dict = {ts: all_deps[ts-1][0] for ts in to_solve}
while to_solve:
    ts = to_solve[-1]
    if time.time() - started > 1:
        print('{} {}'.format(len(to_solve), ts))
        sys.exit()
    if ts not in to_solve_dict:
        to_solve.pop()
        continue
    if to_solve_dict[ts] == 0:
        tmp = to_solve.pop()        
        if tmp in solved:
            continue
        solved.add(tmp)
        ordered_answer.append(tmp)
        del to_solve_dict[ts]
    elif to_solve_dict[ts] == -1:
        to_solve_dict[ts] = 0
        continue
    else:
        to_solve_dict[ts] = -1  # mark as grey
        for x in all_deps[ts-1][1:]:
            if x in to_solve_dict and to_solve_dict[x] == -1:
                print(-1)
                sys.exit()
            elif x not in solved:   
                to_solve.append(x)
                to_solve_dict[x] = all_deps[x-1][0]
                
print(len(ordered_answer))
print(' '.join(str(x) for x in ordered_answer))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Conlinecoursesinbsubootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=10, cycle_prob=0.3):
        self.min_n = min_n
        self.max_n = max_n
        self.cycle_prob = cycle_prob  # 生成循环依赖的概率
    
    def detect_cycle(self, adj):
        visited = {}
        stack = set()

        def dfs(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited[node] = True
            stack.add(node)
            for neighbor in adj.get(node, []):
                if dfs(neighbor):
                    return True
            stack.remove(node)
            return False

        for node in adj:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        k = random.randint(1, n)
        courses = list(range(1, n+1))
        random.shuffle(courses)
        main_courses = random.sample(courses, k)
        
        # 生成课程依赖关系
        dependencies = {}
        for course in courses:
            if random.random() < self.cycle_prob:
                # 允许生成任意依赖（可能产生循环）
                ti = random.randint(0, n-1)
                possible_deps = [c for c in courses if c != course]
                deps = random.sample(possible_deps, ti) if possible_deps else []
            else:
                # 生成无环依赖
                idx = courses.index(course)
                possible_deps = courses[:idx]
                ti = random.randint(0, len(possible_deps))
                deps = random.sample(possible_deps, ti) if possible_deps else []
            dependencies[course] = {'ti': len(deps), 'deps': deps}

        # 构建必须课程集合和邻接表
        required = set()
        q = deque(main_courses)
        while q:
            c = q.popleft()
            if c not in required:
                required.add(c)
                for dep in dependencies[c]['deps']:
                    q.append(dep)

        adj = {}
        for course in required:
            adj[course] = []
            for dep in dependencies[course]['deps']:
                if dep in required:
                    adj[course].append(dep)

        has_cycle = self.detect_cycle(adj)
        
        # 构建正确解（如果不存在循环）
        correct_order = []
        if not has_cycle:
            in_degree = {course: 0 for course in adj}
            for course in adj:
                for dep in adj[course]:
                    in_degree[dep] += 1
            q = deque([c for c in adj if in_degree[c] == 0])
            while q:
                node = q.popleft()
                correct_order.append(node)
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        q.append(neighbor)
            correct_order = list(reversed(correct_order))  # 调整拓扑顺序为执行顺序

        return {
            'n': n,
            'k': k,
            'main': main_courses,
            'dependencies': [{'course': c, 'ti': dependencies[c]['ti'], 'deps': dependencies[c]['deps']} for c in courses],
            'possible': not has_cycle,
            'correct_order': correct_order if not has_cycle else None
        }

    @staticmethod
    def prompt_func(question_case):
        main = ' '.join(map(str, question_case['main']))
        desc = f"""你是贝兰德国立大学的学生，需要完成专业中的{question_case['k']}门主修课程。总共有{question_case['n']}门课程，存在以下依赖关系：
        
课程编号 | 前置课程数量 | 前置课程列表
--------|------------|------------
"""
        for dep in question_case['dependencies']:
            desc += f"{dep['course']} | {dep['ti']} | {' '.join(map(str, dep['deps'])) if dep['deps'] else '无'}\n"
        
        desc += """
请确定完成所有主修课程需要的最少课程数量，并给出正确学习顺序。如果存在循环依赖无法完成，请输出-1。

答案格式要求：
- 如果无解：\n[answer]\n-1\n[/answer]
- 如果有解：\n[answer]\n<m>\n<course sequence>\n[/answer]
例如：
[answer]
3
1 2 3
[/answer]"""
        return desc

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if last_match == '-1':
            return -1
        try:
            lines = [l.strip() for l in last_match.split('\n') if l.strip()]
            m = int(lines[0])
            courses = list(map(int, lines[1].split()))
            if len(courses) != m:
                return None
            return courses
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == -1:
            return not identity['possible']
        if not identity['possible']:
            return False
        
        # 验证课程集合完整性
        required = set()
        q = deque(identity['main'])
        while q:
            c = q.popleft()
            if c not in required:
                required.add(c)
                deps = next(d['deps'] for d in identity['dependencies'] if d['course'] == c)
                for dep in deps:
                    q.append(dep)
        
        if set(solution) != required:
            return False
        
        # 验证顺序正确性
        pos = {c: i for i, c in enumerate(solution)}
        for c in solution:
            deps = next(d['deps'] for d in identity['dependencies'] if d['course'] == c)
            for dep in deps:
                if dep in pos and pos[dep] >= pos[c]:
                    return False
        return True

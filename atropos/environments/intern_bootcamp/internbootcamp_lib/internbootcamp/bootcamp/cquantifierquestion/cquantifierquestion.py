"""# 

### 谜题描述
Logical quantifiers are very useful tools for expressing claims about a set. For this problem, let's focus on the set of real numbers specifically. The set of real numbers includes zero and negatives. There are two kinds of quantifiers: universal (∀) and existential (∃). You can read more about them here.

The universal quantifier is used to make a claim that a statement holds for all real numbers. For example:

  * ∀ x,x<100 is read as: for all real numbers x, x is less than 100. This statement is false. 
  * ∀ x,x>x-1 is read as: for all real numbers x, x is greater than x-1. This statement is true. 



The existential quantifier is used to make a claim that there exists some real number for which the statement holds. For example:

  * ∃ x,x<100 is read as: there exists a real number x such that x is less than 100. This statement is true. 
  * ∃ x,x>x-1 is read as: there exists a real number x such that x is greater than x-1. This statement is true. 



Moreover, these quantifiers can be nested. For example:

  * ∀ x,∃ y,x<y is read as: for all real numbers x, there exists a real number y such that x is less than y. This statement is true since for every x, there exists y=x+1. 
  * ∃ y,∀ x,x<y is read as: there exists a real number y such that for all real numbers x, x is less than y. This statement is false because it claims that there is a maximum real number: a number y larger than every x. 



Note that the order of variables and quantifiers is important for the meaning and veracity of a statement.

There are n variables x_1,x_2,…,x_n, and you are given some formula of the form $$$ f(x_1,...,x_n):=(x_{j_1}<x_{k_1})∧ (x_{j_2}<x_{k_2})∧ ⋅⋅⋅∧ (x_{j_m}<x_{k_m}), $$$

where ∧ denotes logical AND. That is, f(x_1,…, x_n) is true if every inequality x_{j_i}<x_{k_i} holds. Otherwise, if at least one inequality does not hold, then f(x_1,…,x_n) is false.

Your task is to assign quantifiers Q_1,…,Q_n to either universal (∀) or existential (∃) so that the statement $$$ Q_1 x_1, Q_2 x_2, …, Q_n x_n, f(x_1,…, x_n) $$$

is true, and the number of universal quantifiers is maximized, or determine that the statement is false for every possible assignment of quantifiers.

Note that the order the variables appear in the statement is fixed. For example, if f(x_1,x_2):=(x_1<x_2) then you are not allowed to make x_2 appear first and use the statement ∀ x_2,∃ x_1, x_1<x_2. If you assign Q_1=∃ and Q_2=∀, it will only be interpreted as ∃ x_1,∀ x_2,x_1<x_2.

Input

The first line contains two integers n and m (2≤ n≤ 2⋅ 10^5; 1≤ m≤ 2⋅ 10^5) — the number of variables and the number of inequalities in the formula, respectively.

The next m lines describe the formula. The i-th of these lines contains two integers j_i,k_i (1≤ j_i,k_i≤ n, j_i≠ k_i).

Output

If there is no assignment of quantifiers for which the statement is true, output a single integer -1.

Otherwise, on the first line output an integer, the maximum possible number of universal quantifiers.

On the next line, output a string of length n, where the i-th character is \"A\" if Q_i should be a universal quantifier (∀), or \"E\" if Q_i should be an existential quantifier (∃). All letters should be upper-case. If there are multiple solutions where the number of universal quantifiers is maximum, print any.

Examples

Input


2 1
1 2


Output


1
AE


Input


4 3
1 2
2 3
3 1


Output


-1


Input


3 2
1 3
2 3


Output


2
AAE

Note

For the first test, the statement ∀ x_1, ∃ x_2, x_1<x_2 is true. Answers of \"EA\" and \"AA\" give false statements. The answer \"EE\" gives a true statement, but the number of universal quantifiers in this string is less than in our answer.

For the second test, we can show that no assignment of quantifiers, for which the statement is true exists.

For the third test, the statement ∀ x_1, ∀ x_2, ∃ x_3, (x_1<x_3)∧ (x_2<x_3) is true: We can set x_3=max\\{x_1,x_2\}+1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

def toposort(graph):
    n = len(graph)
    res = []
    found = [0] * n

    for i in range(n):
        if found[i]:
            continue
        stack = [i]
        while stack:
            node = stack.pop()
            if node < 0:
                res.append(~node)
            elif not found[node]:
                found[node] = 1
                stack.append(~node)
                for nei in graph[node]:
                    if not found[nei]:
                        stack.append(nei)
    
    # Check for cycle
    for node in res:
        if any(found[nei] for nei in graph[node]):
            return None
        found[node] = 0

    return res[::-1]


inp = [int(x) for x in sys.stdin.read().split()]; ii = 0

n = inp[ii]; ii += 1
m = inp[ii]; ii += 1

coupl1 = [[] for _ in range(n)]
coupl2 = [[] for _ in range(n)]
for _ in range(m):
    u = inp[ii] - 1; ii += 1
    v = inp[ii] - 1; ii += 1
    coupl1[u].append(v)
    coupl2[v].append(u)

order = toposort(coupl1)
if order is None:
    print -1
    sys.exit()

seen1 = list(range(n))
seen2 = list(range(n))
for node in order:
    for nei in coupl1[node]:
        seen1[nei] = min(seen1[node], seen1[nei])

for node in reversed(order):
    for nei in coupl2[node]:
        seen2[nei] = min(seen2[node], seen2[nei])

seen = [+(seen1[node] == seen2[node] == node) for node in range(n)]
print sum(seen)
print ''.join('A' if c else 'E' for c in seen)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from collections import deque

class Cquantifierquestionbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=5, m_min=1, m_max=5):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max

    def case_generator(self):
        # 生成DAG的边
        n = random.randint(self.n_min, self.n_max)
        # 生成拓扑序
        top_order = list(range(1, n+1))
        random.shuffle(top_order)
        
        # 生成所有可能的边（仅从前到后）
        possible_edges = []
        for i in range(len(top_order)):
            for j in range(i+1, len(top_order)):
                possible_edges.append((top_order[i], top_order[j]))
        
        # 确定边的数量
        max_valid_edges = len(possible_edges)
        m = random.randint(max(self.m_min, 1), min(self.m_max, max_valid_edges))
        edges = random.sample(possible_edges, m) if possible_edges else []
        
        return {
            'n': n,
            'm': len(edges),
            'edges': edges
        }

    @staticmethod
    def prompt_func(question_case):
        edges = question_case['edges']
        edges_str = '\n'.join(f"{j} {k}" for j, k in edges)
        return f"""请为以下逻辑问题分配量词（A/E）使表达式为真且全称最多：
变量数：{question_case['n']}，不等式数：{question_case['m']}
不等式列表：
{edges_str}
答案格式：[answer]...[/answer]，若无解输出-1"""

    @staticmethod
    def extract_output(output):
        # 提取最后一个答案块
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        content = answer_blocks[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
        # 处理-1的情况
        if lines[0] == '-1':
            return -1
        # 尝试解析两行格式
        if len(lines) >= 2:
            count_line = lines[0]
            quant_line = lines[1].upper()
            if count_line.isdigit() and len(quant_line) == int(count_line) and all(c in 'AE' for c in quant_line):
                return (int(count_line), quant_line)
        # 尝试单行格式，例如 "1 EA"
        if len(lines) == 1:
            parts = lines[0].split()
            if len(parts) == 2 and parts[0].isdigit() and len(parts[1]) == int(parts[0]):
                quant = parts[1].upper()
                if all(c in 'AE' for c in quant):
                    return (int(parts[0]), quant)
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 获取参考解
        ref_sol = cls.reference_solution(identity)
        # 处理无解情况
        if isinstance(ref_sol, int):
            return solution == -1
        # 处理有解情况
        return solution == ref_sol

    @classmethod
    def reference_solution(cls, identity):
        # 完全复制参考代码逻辑
        def toposort(graph):
            n = len(graph)
            res = []
            found = [0]*n

            for i in range(n):
                if found[i]:
                    continue
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node < 0:
                        res.append(~node)
                    elif not found[node]:
                        found[node] = 1
                        stack.append(~node)
                        for nei in graph[node]:
                            if not found[nei]:
                                stack.append(nei)
            
            # Check cycle
            found = [0]*n
            for node in res:
                if found[node]:
                    return None
                stack = [node]
                found[node] = 1
                while stack:
                    current = stack.pop()
                    for nei in graph[current]:
                        if found[nei]:
                            return None
                        if not found[nei]:
                            found[nei] = 1
                            stack.append(nei)
            return res[::-1]

        n = identity['n']
        edges = identity['edges']
        coupl1 = [[] for _ in range(n)]
        coupl2 = [[] for _ in range(n)]
        for j, k in edges:
            u = j - 1
            v = k - 1
            coupl1[u].append(v)
            coupl2[v].append(u)
        
        order = toposort(coupl1)
        if order is None:
            return -1
        
        seen1 = list(range(n))
        seen2 = list(range(n))
        
        for node in order:
            for nei in coupl1[node]:
                if seen1[nei] > seen1[node]:
                    seen1[nei] = seen1[node]
        
        for node in reversed(order):
            for nei in coupl2[node]:
                if seen2[nei] > seen2[node]:
                    seen2[nei] = seen2[node]
        
        seen = [(seen1[i] == i and seen2[i] == i) for i in range(n)]
        count = sum(seen)
        if count == 0:
            return -1
        quant = ''.join('A' if c else 'E' for c in seen)
        return (count, quant)

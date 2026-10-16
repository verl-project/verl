"""# 

### 谜题描述
On Children's Day, the child got a toy from Delayyy as a present. However, the child is so naughty that he can't wait to destroy the toy.

The toy consists of n parts and m ropes. Each rope links two parts, but every pair of parts is linked by at most one rope. To split the toy, the child must remove all its parts. The child can remove a single part at a time, and each remove consume an energy. Let's define an energy value of part i as vi. The child spend vf1 + vf2 + ... + vfk energy for removing part i where f1, f2, ..., fk are the parts that are directly connected to the i-th and haven't been removed.

Help the child to find out, what is the minimum total energy he should spend to remove all n parts.

Input

The first line contains two integers n and m (1 ≤ n ≤ 1000; 0 ≤ m ≤ 2000). The second line contains n integers: v1, v2, ..., vn (0 ≤ vi ≤ 105). Then followed m lines, each line contains two integers xi and yi, representing a rope from part xi to part yi (1 ≤ xi, yi ≤ n; xi ≠ yi).

Consider all the parts are numbered from 1 to n.

Output

Output the minimum total energy the child should spend to remove all n parts of the toy.

Examples

Input

4 3
10 20 30 40
1 4
1 2
2 3


Output

40


Input

4 4
100 100 100 100
1 2
2 3
2 4
3 4


Output

400


Input

7 10
40 10 20 10 20 80 40
1 5
4 7
4 5
5 2
5 7
6 4
1 6
1 3
4 3
1 4


Output

160

Note

One of the optimal sequence of actions in the first sample is:

  * First, remove part 3, cost of the action is 20. 
  * Then, remove part 2, cost of the action is 10. 
  * Next, remove part 4, cost of the action is 10. 
  * At last, remove part 1, cost of the action is 0. 



So the total energy the child paid is 20 + 10 + 10 + 0 = 40, which is the minimum.

In the second sample, the child will spend 400 no matter in what order he will remove the parts.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())
a = map(int, raw_input().split())
ans = 0
for i in xrange(m):
    x, y = map(int, raw_input().split())
    ans += min(a[x-1], a[y-1])
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Athechildandtoybootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, m_min=0, m_max=2000, v_min=0, v_max=10**5):
        """
        Initialize parameters for generating puzzle instances.
        Default parameters are set to generate small cases for testing.
        """
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.v_min = v_min
        self.v_max = v_max
    
    def case_generator(self):
        """
        Generate a puzzle instance with random n, m, values, and edges.
        Ensures edges are unique and nodes are within valid range.
        """
        # Determine n and possible m range
        n = random.randint(self.n_min, self.n_max)
        max_possible_edges = n * (n - 1) // 2
        m_upper = min(self.m_max, max_possible_edges)
        m = random.randint(self.m_min, m_upper)
        
        # Generate values for each node
        v = [random.randint(self.v_min, self.v_max) for _ in range(n)]
        
        # Generate unique undirected edges
        edges = set()
        while len(edges) < m:
            x = random.randint(1, n)
            y = random.randint(1, n)
            if x == y:
                continue
            u, w = (x, y) if x < y else (y, x)
            edges.add((u, w))
        edges = [list(pair) for pair in edges]
        
        return {
            'n': n,
            'm': m,
            'v': v,
            'edges': edges
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        Format the puzzle instance into a text prompt with problem description and answer format.
        """
        n = question_case['n']
        m = question_case['m']
        v = question_case['v']
        edges = question_case['edges']
        
        edges_str = '\n'.join(f"{x} {y}" for x, y in edges)
        prompt = f"""在儿童节，孩子收到了一个由多个部件和绳子组成的玩具。你的任务是帮助孩子找到拆除所有部件所需的最小总能量。规则如下：

玩具包含n个部件和m根绳子。每个部件有一个能量值v_i。每次只能移除一个部件，每次移除时消耗的能量等于该部件此时直接连接的、未被移除的部件的能量值之和。你的任务是确定一个移除顺序，使得总消耗的能量最小，并输出这个最小值。

输入数据格式：
- 第一行是两个整数n和m，分别表示部件数量和绳子数量。
- 第二行包含n个整数，表示每个部件的能量值v_1到v_n。
- 接下来m行，每行两个整数x和y，表示部件x和部件y之间有一根绳子连接。

输出格式：
- 输出一个整数，表示最小的总能量消耗。

示例：

输入：
4 3
10 20 30 40
1 4
1 2
2 3

输出：
40

现在需要解决以下问题实例：

输入：
{n} {m}
{' '.join(map(str, v))}
{edges_str}

请计算对应的最小总能量，并将最终答案放在[answer]和[/answer]标签之间。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        Extract the last correctly formatted answer from the model's output.
        """
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        Verify if the extracted solution matches the minimal total energy.
        """
        v = identity['v']
        edges = identity['edges']
        total = 0
        for x, y in edges:
            total += min(v[x-1], v[y-1])
        return solution == total

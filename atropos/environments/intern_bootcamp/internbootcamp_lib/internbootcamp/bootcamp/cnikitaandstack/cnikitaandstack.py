"""# 

### 谜题描述
Nikita has a stack. A stack in this problem is a data structure that supports two operations. Operation push(x) puts an integer x on the top of the stack, and operation pop() deletes the top integer from the stack, i. e. the last added. If the stack is empty, then the operation pop() does nothing.

Nikita made m operations with the stack but forgot them. Now Nikita wants to remember them. He remembers them one by one, on the i-th step he remembers an operation he made pi-th. In other words, he remembers the operations in order of some permutation p1, p2, ..., pm. After each step Nikita wants to know what is the integer on the top of the stack after performing the operations he have already remembered, in the corresponding order. Help him!

Input

The first line contains the integer m (1 ≤ m ≤ 105) — the number of operations Nikita made.

The next m lines contain the operations Nikita remembers. The i-th line starts with two integers pi and ti (1 ≤ pi ≤ m, ti = 0 or ti = 1) — the index of operation he remembers on the step i, and the type of the operation. ti equals 0, if the operation is pop(), and 1, is the operation is push(x). If the operation is push(x), the line also contains the integer xi (1 ≤ xi ≤ 106) — the integer added to the stack.

It is guaranteed that each integer from 1 to m is present exactly once among integers pi.

Output

Print m integers. The integer i should equal the number on the top of the stack after performing all the operations Nikita remembered on the steps from 1 to i. If the stack is empty after performing all these operations, print -1.

Examples

Input

2
2 1 2
1 0


Output

2
2


Input

3
1 1 2
2 1 3
3 0


Output

2
3
2


Input

5
5 0
4 0
3 1 1
2 1 1
1 1 2


Output

-1
-1
-1
-1
2

Note

In the first example, after Nikita remembers the operation on the first step, the operation push(2) is the only operation, so the answer is 2. After he remembers the operation pop() which was done before push(2), answer stays the same.

In the second example, the operations are push(2), push(3) and pop(). Nikita remembers them in the order they were performed.

In the third example Nikita remembers the operations in the reversed order.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def _main():
    from sys import stdin, stdout
    
    def modify(x, left, right, L, R, V):
        if L <= left and right <= R:
            add[x] += V
            return
        if add[x]:
            add[2*x] += add[x]
            add[2*x+1] += add[x]
            add[x] = 0
        mid = (left + right) / 2
        if L <= mid:
            modify(2 * x, left, mid, L, R, V)
        if mid < R:
            modify(2 * x + 1, mid + 1, right, L, R, V)
        mx[x] = max(mx[2*x] + add[2*x],
                    mx[2*x+1] + add[2*x + 1])

    def find_last(x, left, right):
        if left == right:
            return left
        if add[x]:
            add[2*x] += add[x]
            add[2*x+1] += add[x]
            add[x] = 0
        mid = (left + right) / 2
        ans = -1
        if mx[2 * x + 1] + add[2 * x + 1] > 0:
            ans = find_last(2 * x + 1, mid + 1, right)
        else:
            ans = find_last(2 * x, left, mid)
        mx[x] = max(mx[2*x] + add[2*x],
                    mx[2*x+1] + add[2*x + 1])
        return ans

    N = int(stdin.readline().strip())
    AA = [map(int, stdin.readline().strip().split())
          for _ in xrange(N)]
    
    MAXN = 3*N+3
    add = [0] * MAXN
    mx = [0] * MAXN
    val = [0] * MAXN
    for inp in AA:
        p, t = inp[:2]
        if t: val[p] = inp[2]
        modify(1, 1, N, 1, p, 1 if t else -1)
        if mx[1] + add[1] <= 0:
            stdout.write('-1\n')
        else:
            stdout.write(str(val[find_last(1, 1, N)])+'\n')

_main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cnikitaandstackbootcamp(Basebootcamp):
    def __init__(self, m_min=3, m_max=5, **kwargs):
        super().__init__(**kwargs)
        self.m_min = m_min
        self.m_max = m_max

    def case_generator(self):
        m = random.randint(self.m_min, self.m_max)
        permutation = random.sample(range(1, m+1), m)  # 严格全排列
        
        # 生成原始操作序列（保证push足够多）
        operations = []
        push_count = 0
        for _ in range(m):
            # 动态调整push概率：剩余操作中至少需要与pop数量匹配
            remaining_ops = m - len(operations)
            min_push = max(0, (remaining_ops + 1) // 2)
            current_push_prob = 0.6 if push_count > (len(operations) - push_count) else 0.8
            
            ti = 1 if random.random() < current_push_prob else 0
            if ti == 1:
                xi = random.randint(1, 10**6)
                operations.append({'type': ti, 'x': xi})
                push_count += 1
            else:
                operations.append({'type': ti})
        
        # 构建输入数据
        input_lines = [[m]]
        for pi in permutation:
            op = operations[pi-1]
            line = [pi, op['type']]
            if op['type'] == 1:
                line.append(op['x'])
            input_lines.append(line)
        
        # 严格计算预期输出
        expected_output = []
        for i in range(1, m+1):
            # 获取当前已回忆的所有操作索引（按原始顺序排序）
            current_ops = sorted(permutation[:i])
            
            # 模拟栈操作
            stack = []
            for p in current_ops:
                op = operations[p-1]
                if op['type'] == 1:
                    stack.append(op['x'])
                else:
                    if stack:
                        stack.pop()
            
            # 记录栈顶状态
            expected_output.append(stack[-1] if stack else -1)
        
        return {
            'm': m,
            'permutation': permutation,
            'operations': operations,
            'input_lines': input_lines,
            'expected_output': expected_output
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        problem_lines = []
        problem_lines.append(str(question_case['m']))
        for line in question_case['input_lines'][1:]:  # 跳过首行的m
            problem_lines.append(' '.join(map(str, line)))
        
        return (
            "Nikita 按某顺序回忆栈操作，每一步需输出当前栈顶。输入格式：\n"
            f"{problem_lines[0]}\n" + "\n".join(problem_lines[1:]) + "\n"
            "请将m个结果放在[answer]和[/answer]之间，每行一个整数。"
        )

    @staticmethod
    def extract_output(output):
        last_answer = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not last_answer:
            return None
        
        nums = []
        for line in last_answer[-1].split('\n'):
            line = line.strip()
            if re.fullmatch(r'-?\d+', line):
                nums.append(int(line))
        return nums if len(nums) == len(last_answer[-1].strip().split('\n')) else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_output']

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
#include <bits/stdc++.h>
using namespace std;
int getint() {
  int f = 1, x = 0;
  char ch = getchar();
  while (ch < '0' || ch > '9') {
    if (ch == '-') f = -1;
    ch = getchar();
  }
  while (ch >= '0' && ch <= '9') {
    x = x * 10 + ch - '0';
    ch = getchar();
  }
  return f * x;
}
struct node {
  long long x, y;
  int l, r;
} tr[212345 << 2];
void pushup(int i) {
  int lx = tr[(i << 1)].x, ly = tr[(i << 1)].y, rx = tr[(i << 1 | 1)].x,
      ry = tr[(i << 1 | 1)].y;
  int t = min(ly, rx);
  int x = lx + rx - t;
  int y = ry + ly - t;
  tr[i].x = x;
  tr[i].y = y;
  return;
}
void build(int i, int l, int r) {
  tr[i].l = l;
  tr[i].r = r;
  tr[i].x = tr[i].y = 0;
  if (l == r) {
    return;
  }
  int mid = (l + r) >> 1;
  build((i << 1), l, (mid));
  build((i << 1 | 1), (mid + 1), r);
  return;
}
int op[212345];
int w[212345];
void modify(int i, int id) {
  int l = tr[i].l, r = tr[i].r;
  if (l == r) {
    if (op[id])
      tr[i].y = 1;
    else
      tr[i].x = 1;
    return;
  }
  int mid = (l + r) >> 1;
  if (id <= (mid))
    modify((i << 1), id);
  else
    modify((i << 1 | 1), id);
  pushup(i);
  return;
}
int query(int i, int x) {
  int l = tr[i].l, r = tr[i].r;
  if (tr[i].y <= x) return 0;
  if (l == r) return l;
  if (tr[(i << 1 | 1)].y && tr[(i << 1 | 1)].y > x)
    return query((i << 1 | 1), x);
  else
    return query((i << 1), x - tr[(i << 1 | 1)].y + tr[(i << 1 | 1)].x);
}
int main() {
  int n = getint();
  build(1, 1, n);
  for (int i = 1; i <= n; i++) {
    int id = getint();
    op[id] = getint();
    if (op[id]) w[id] = getint();
    modify(1, id);
    id = query(1, 0);
    if (op[id] == 0)
      printf(\"-1\n\");
    else
      printf(\"%d\n\", w[id]);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Enikitaandstackbootcamp(Basebootcamp):
    def __init__(self, m=5):
        super().__init__()
        self.params = {'m': m}
    
    def case_generator(self):
        m = self.params['m']
        ops = []
        for _ in range(m):
            ti = random.choice([0, 1])
            xi = random.randint(1, 10**6) if ti == 1 else None
            ops.append({'ti': ti, 'xi': xi})
        
        pi_permutation = list(range(1, m+1))
        random.shuffle(pi_permutation)
        
        input_lines = []
        for pi in pi_permutation:
            op = ops[pi - 1]
            line = [pi, op['ti']]
            if op['ti'] == 1:
                line.append(op['xi'])
            input_lines.append(line)
        
        # 高效生成正确输出（参考线段树方法）
        op_data = {}
        for idx, line in enumerate(input_lines):
            pi, ti = line[0], line[1]
            xi = line[2] if ti == 1 else None
            op_data[pi] = (ti, xi)
        
        sorted_pis = sorted(op_data.keys())
        ordered_ops = [op_data[pi] for pi in sorted_pis]
        n = m
        
        class SegmentTreeNode:
            __slots__ = ['l', 'r', 'push_cnt', 'pop_cnt']
            def __init__(self, l, r):
                self.l = l
                self.r = r
                self.push_cnt = 0
                self.pop_cnt = 0
        
        tree = [None] * (4 * n)
        
        def build(i, l, r):
            tree[i] = SegmentTreeNode(l, r)
            if l == r:
                return
            mid = (l + r) // 2
            build(2*i, l, mid)
            build(2*i+1, mid+1, r)
        
        build(1, 1, n)
        
        def update(i, pos, ti):
            node = tree[i]
            if node.l == node.r:
                if ti == 1:
                    node.push_cnt = 1
                    node.pop_cnt = 0
                else:
                    node.pop_cnt = 1
                    node.push_cnt = 0
                return
            mid = (node.l + node.r) // 2
            if pos <= mid:
                update(2*i, pos, ti)
            else:
                update(2*i+1, pos, ti)
            lc = tree[2*i]
            rc = tree[2*i+1]
            matched = min(lc.push_cnt, rc.pop_cnt)
            node.push_cnt = rc.push_cnt + (lc.push_cnt - matched)
            node.pop_cnt = lc.pop_cnt + (rc.pop_cnt - matched)
        
        pos_map = {pi: i+1 for i, pi in enumerate(sorted_pis)}
        
        correct_output = []
        active_set = set()
        
        for step, pi in enumerate(pi_permutation, 1):
            active_set.add(pi)
            ti, xi = op_data[pi]
            update(1, pos_map[pi], ti)
            
            def find_top():
                node = tree[1]
                if node.push_cnt == 0:
                    return -1
                current = 1
                excess = 0
                while True:
                    cn = tree[current]
                    if cn.l == cn.r:
                        if cn.push_cnt > 0 and (excess + cn.push_cnt) > 0:
                            return cn.l
                        return -1
                    rc = tree[2*current+1]
                    if rc.push_cnt > excess:
                        current = 2*current+1
                    else:
                        excess -= rc.push_cnt
                        excess += rc.pop_cnt
                        current = 2*current
                return -1
            
            top_pos = find_top()
            if top_pos == -1:
                correct_output.append(-1)
            else:
                original_pi = sorted_pis[top_pos-1]
                ti, xi = op_data[original_pi]
                correct_output.append(xi)
        
        case = {
            'm': m,
            'input_lines': input_lines,
            'correct_output': correct_output,
            'ops_order': sorted_pis,
            'op_data': op_data
        }
        return case
    
    @staticmethod
    def prompt_func(question_case) -> str:
        m = question_case['m']
        input_lines = question_case['input_lines']
        input_str = f"{m}\n" + "\n".join(" ".join(map(str, line)) for line in input_lines)
        
        prompt = f"""你是编程竞赛的选手，请解决以下问题。仔细阅读问题描述，并按照输入输出格式要求进行处理。

问题描述：
Nikita有一个栈。栈支持两种操作：push(x)将整数x压入栈顶，pop()删除栈顶元素（如果栈为空则不执行任何操作）。Nikita执行了{m}次操作，现在他按步骤回忆这些操作。在第i步他回忆的是原操作中的第pi个操作。每次回忆后，他需要执行所有已回忆的操作（按原操作的时间顺序）后的栈顶元素，若栈空则输出-1。

输入格式：
第一行是整数m（1 ≤ m ≤ 1e5）。
接下来的m行，每行包含pi（1-based原操作序号）、ti（0或1），若ti为1则后跟xi（1 ≤ xi ≤ 1e6）。
保证pi是1到m的一个排列。

输出格式：
输出{m}行，每行是执行前i个回忆操作后的栈顶元素。

示例输入：
2
2 1 2
1 0

示例输出：
2
2

请根据以下输入解决问题，并将答案按输出格式要求放在[answer]标签内。

输入：
{input_str}

请将你的答案放在[answer]标签内，例如：
[answer]
输出行1
输出行2
...
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_str = matches[-1].strip()
        lines = [line.strip() for line in answer_str.split('\n') if line.strip()]
        try:
            solution = list(map(int, lines))
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = identity['correct_output']
        return solution == correct

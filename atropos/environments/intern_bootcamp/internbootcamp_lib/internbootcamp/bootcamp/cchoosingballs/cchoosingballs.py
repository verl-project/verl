"""# 

### 谜题描述
There are n balls. They are arranged in a row. Each ball has a color (for convenience an integer) and an integer value. The color of the i-th ball is ci and the value of the i-th ball is vi.

Squirrel Liss chooses some balls and makes a new sequence without changing the relative order of the balls. She wants to maximize the value of this sequence.

The value of the sequence is defined as the sum of following values for each ball (where a and b are given constants):

  * If the ball is not in the beginning of the sequence and the color of the ball is same as previous ball's color, add (the value of the ball)  ×  a. 
  * Otherwise, add (the value of the ball)  ×  b. 



You are given q queries. Each query contains two integers ai and bi. For each query find the maximal value of the sequence she can make when a = ai and b = bi.

Note that the new sequence can be empty, and the value of an empty sequence is defined as zero.

Input

The first line contains two integers n and q (1 ≤ n ≤ 105; 1 ≤ q ≤ 500). The second line contains n integers: v1, v2, ..., vn (|vi| ≤ 105). The third line contains n integers: c1, c2, ..., cn (1 ≤ ci ≤ n).

The following q lines contain the values of the constants a and b for queries. The i-th of these lines contains two integers ai and bi (|ai|, |bi| ≤ 105).

In each line integers are separated by single spaces.

Output

For each query, output a line containing an integer — the answer to the query. The i-th line contains the answer to the i-th query in the input order.

Please, do not write the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

6 3
1 -2 3 4 0 -1
1 2 1 2 1 1
5 1
-2 1
1 0


Output

20
9
4


Input

4 1
-3 6 -1 2
1 2 3 1
1 -1


Output

5

Note

In the first example, to achieve the maximal value:

  * In the first query, you should select 1st, 3rd, and 4th ball. 
  * In the second query, you should select 3rd, 4th, 5th and 6th ball. 
  * In the third query, you should select 2nd and 4th ball. 



Note that there may be other ways to achieve the maximal value.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

inp = [int(x) for x in sys.stdin.read().split()]; ii = 0

n = inp[ii]; ii += 1
q = inp[ii]; ii += 1

V = [float(x) for x in inp[ii:ii + n]]; ii += n
C = [c - 1 for c in inp[ii:ii + n]]; ii += n

A = inp[ii::2]
B = inp[ii+1::2]

inf = 1e30

out = []

for _ in range(q):
    a = float(A[_])
    b = float(B[_])

    bestcolor = [-inf]*n
    
    best1 = 0.0
    color1 = -1
    best2 = 0.0

    for i in range(n):
        c = C[i]
        v = V[i]

        x = max(bestcolor[c] + v * a, (best1 if color1 != c else best2) + v * b)
        
        bestcolor[c] = max(bestcolor[c], x)
        if x > best1:
            if color1 != c:
                best2 = best1
            best1 = x
            color1 = c
        elif x > best2 and color1 != c:
            best2 = x
    out.append(best1)
print '\n'.join(str(int(x)) for x in out)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cchoosingballsbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=10, min_q=1, max_q=5, max_v=10, max_color=None, max_ab=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_q = min_q
        self.max_q = max_q
        self.max_v = max_v
        self.max_color = max_color or max_n  # Ensure colors don't exceed problem constraints
        self.max_ab = max_ab

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        q = random.randint(self.min_q, self.max_q)
        v = [random.randint(-self.max_v, self.max_v) for _ in range(n)]
        max_color_val = min(n, self.max_color)
        c = [random.randint(1, max_color_val) for _ in range(n)]
        queries = [(
            random.randint(-self.max_ab, self.max_ab),
            random.randint(-self.max_ab, self.max_ab)
        ) for _ in range(q)]
        answers = self._compute_answers(n, q, v, c, queries)
        return {
            'n': n,
            'q': q,
            'v': v,
            'c': c,
            'queries': queries,
            'answers': answers
        }

    def _compute_answers(self, n, q, v, c, queries):
        answers = []
        for a, b in queries:
            bestcolor = [-float('inf')] * (max(c) if c else 0)  # Handle empty case
            best1, color1, best2 = 0.0, -1, 0.0
            for i in range(n):
                current_c = c[i] - 1  # Convert to 0-based
                current_v = v[i]

                # Calculate option possibilities
                same_color_value = bestcolor[current_c] + current_v * a
                diff_color_value = (best2 if current_c == color1 else best1) + current_v * b
                x = max(same_color_value, diff_color_value)

                # Update color tracking
                if x > bestcolor[current_c]:
                    bestcolor[current_c] = x

                # Maintain top2 values
                if x > best1:
                    if current_c != color1:
                        best2 = best1
                    best1, color1 = x, current_c
                elif current_c != color1 and x > best2:
                    best2 = x
            answers.append(int(best1))  # Correct conversion without rounding
        return answers

    @staticmethod
    def prompt_func(question_case):
        prompt = [
            "作为松鼠Liss的助手，你需要计算不同参数下的球序列最大价值。",
            "规则如下：",
            "1. 保持球的原始顺序，选择任意子序列（包括空序列）",
            "2. 每个球的贡献值计算方式：",
            "   - 如果当前球不是序列的第一个且颜色与前一个相同：贡献值 = 球值 × a",
            "   - 否则：贡献值 = 球值 × b",
            "--- 题目数据 ---",
            f"球数量（n）: {question_case['n']}",
            f"查询次数（q）: {question_case['q']}",
            "球价值列表：" + ' '.join(map(str, question_case['v'])),
            "球颜色列表：" + ' '.join(map(str, question_case['c'])),
            "查询参数（a b）："
        ]
        for i, (a, b) in enumerate(question_case['queries'], 1):
            prompt.append(f"查询{i}: {a} {b}")
        prompt.append(
            "请输出每个查询对应的最大价值，严格按照出现顺序每行一个整数，并用[ANSWER]标签包裹。\n"
            "示例：\n[ANSWER]\n12\n5\n7\n[/ANSWER]"
        )
        return '\n'.join(prompt)

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[ANSWER\](.*?)\[/ANSWER\]', output, re.DOTALL | re.IGNORECASE)
        if not answer_blocks:
            return None
        try:
            # 提取最后一个ANSWER块并转换数值
            last_answer = answer_blocks[-1].strip().split()
            return list(map(int, last_answer))
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 严格匹配答案顺序和数值
        return solution == identity.get('answers', [])

# 示例测试用例验证（开发时测试用）
if __name__ == "__main__":
    # 初始化训练场
    bootcamp = Cchoosingballsbootcamp(min_n=6, max_n=6, min_q=3, max_q=3)
    
    # 生成示例测试用例
    test_case = {
        'n': 6,
        'q': 3,
        'v': [1, -2, 3, 4, 0, -1],
        'c': [1, 2, 1, 2, 1, 1],
        'queries': [(5, 1), (-2, 1), (1, 0)],
        'answers': [20, 9, 4]
    }
    
    # 验证算法实现
    computed = bootcamp._compute_answers(
        test_case['n'], test_case['q'],
        test_case['v'], test_case['c'],
        test_case['queries']
    )
    print("算法验证:", computed == test_case['answers'])  # 应输出True
    
    # 测试完整流程
    case = bootcamp.case_generator()
    prompt = Cchoosingballsbootcamp.prompt_func(case)
    print("\n生成的问题示例:\n" + prompt[:500] + "...")

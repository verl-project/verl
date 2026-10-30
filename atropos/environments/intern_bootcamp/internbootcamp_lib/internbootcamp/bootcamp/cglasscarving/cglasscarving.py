"""# 

### 谜题描述
Leonid wants to become a glass carver (the person who creates beautiful artworks by cutting the glass). He already has a rectangular w mm  ×  h mm sheet of glass, a diamond glass cutter and lots of enthusiasm. What he lacks is understanding of what to carve and how.

In order not to waste time, he decided to practice the technique of carving. To do this, he makes vertical and horizontal cuts through the entire sheet. This process results in making smaller rectangular fragments of glass. Leonid does not move the newly made glass fragments. In particular, a cut divides each fragment of glass that it goes through into smaller fragments.

After each cut Leonid tries to determine what area the largest of the currently available glass fragments has. Since there appear more and more fragments, this question takes him more and more time and distracts him from the fascinating process.

Leonid offers to divide the labor — he will cut glass, and you will calculate the area of the maximum fragment after each cut. Do you agree?

Input

The first line contains three integers w, h, n (2 ≤ w, h ≤ 200 000, 1 ≤ n ≤ 200 000).

Next n lines contain the descriptions of the cuts. Each description has the form H y or V x. In the first case Leonid makes the horizontal cut at the distance y millimeters (1 ≤ y ≤ h - 1) from the lower edge of the original sheet of glass. In the second case Leonid makes a vertical cut at distance x (1 ≤ x ≤ w - 1) millimeters from the left edge of the original sheet of glass. It is guaranteed that Leonid won't make two identical cuts.

Output

After each cut print on a single line the area of the maximum available glass fragment in mm2.

Examples

Input

4 3 4
H 2
V 2
V 3
V 1


Output

8
4
4
2


Input

7 6 5
H 4
V 3
V 5
H 2
V 1


Output

28
16
12
6
4

Note

Picture for the first sample test: 

<image> Picture for the second sample test:  <image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def gao(c):
    n = len(c)
    a = [(c[i], i) for i in range(n)]
    a.sort()
    l = [0] * n
    r = [0] * n
    for i in range(1, n-1):
        l[i] = i - 1
        r[i] = i + 1
    index = [0] * n
    for i in range(n):
        index[a[i][1]] = i
    mx = 0
    for i in range(1, n):
        mx = max(mx, a[i][0] - a[i-1][0])
    ans = [mx]
    for i in range(n-1, 2, -1):
        p = index[i]
        mx = max(mx, a[r[p]][0] - a[l[p]][0])
        ans.append(mx)
        l[r[p]] = l[p]
        r[l[p]] = r[p]
    ans.reverse()
    return ans

[w, h, n] = list(map(int, raw_input().split()))
H = [0, h] + [0] * n
V = [0, w] + [0] * n
for i in range(n):
    [op, v] = raw_input().split()
    v = int(v)
    if op == 'H':
        H[i+2] = v
        V[i+2] = V[i+1]
    else:
        V[i+2] = v
        H[i+2] = H[i+1]
ansH = []
ansV = []
ansH = gao(H)
ansV = gao(V)
for i in range(n):
    print ansH[i] * ansV[i]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cglasscarvingbootcamp(Basebootcamp):
    def __init__(self, default_w=4, default_h=3, max_cuts=4):
        """
        初始化玻璃切割训练场环境。
        :param default_w: 默认玻璃板宽度
        :param default_h: 默认玻璃板高度
        :param max_cuts: 默认切割次数
        """
        self.default_w = default_w
        self.default_h = default_h
        self.max_cuts = max_cuts

    def case_generator(self):
        """
        生成一个玻璃切割谜题实例，包含初始尺寸、切割序列及预期结果。
        """
        w = self.default_w
        h = self.default_h
        n = self.max_cuts

        cuts = []
        h_cuts = set()  # 记录所有H切割的y坐标
        v_cuts = set()  # 记录所有V切割的x坐标

        for _ in range(n):
            # 动态选择可用的切割方向
            can_h = len(h_cuts) < h - 1
            can_v = len(v_cuts) < w - 1
            if not can_h and not can_v:
                raise ValueError("无法生成更多切割步骤，请调整参数")

            choices = []
            if can_h:
                choices.append('H')
            if can_v:
                choices.append('V')
            
            op = random.choice(choices)
            
            if op == 'H':
                available = list(set(range(1, h)) - h_cuts)
                y = random.choice(available)
                cuts.append({'type': 'H', 'value': y})
                h_cuts.add(y)
            else:
                available = list(set(range(1, w)) - v_cuts)
                x = random.choice(available)
                cuts.append({'type': 'V', 'value': x})
                v_cuts.add(x)

        # 构建切割序列数组
        H = [0, h] + [0] * n
        V = [0, w] + [0] * n
        for i in range(n):
            cut = cuts[i]
            if cut['type'] == 'H':
                H[i+2] = cut['value']
                V[i+2] = V[i+1]
            else:
                V[i+2] = cut['value']
                H[i+2] = H[i+1]

        # 计算预期结果
        ansH = self.gao(H)
        ansV = self.gao(V)
        expected_areas = [ansH[i] * ansV[i] for i in range(n)]

        return {
            'w': w,
            'h': h,
            'cuts': cuts,
            'expected_areas': expected_areas
        }

    @staticmethod
    def gao(c):
        """参考代码中的gao函数实现"""
        n = len(c)
        a = sorted((val, idx) for idx, val in enumerate(c))
        l = list(range(-1, n))
        r = list(range(1, n+2))
        index = [0] * n
        for i, (val, idx) in enumerate(a):
            index[idx] = i

        mx = 0
        for i in range(1, n):
            mx = max(mx, a[i][0] - a[i-1][0])
        
        ans = [mx]
        for i in range(n-1, 2, -1):
            pos = index[i]
            left = l[pos]
            right = r[pos]
            if left >= 0:
                r[left] = right
            if right < n:
                l[right] = left
            current_gap = a[right][0] - a[left][0] if right < n else 0
            mx = max(mx, current_gap)
            ans.append(mx)
        
        ans.reverse()
        return ans[:n]

    @staticmethod
    def prompt_func(question_case) -> str:
        """将问题案例转换为自然语言描述"""
        w = question_case['w']
        h = question_case['h']
        cuts = question_case['cuts']
        problem = (
            "Leonid有一块宽{w}毫米、高{h}毫米的玻璃板。他按顺序进行了以下{num_cuts}次切割：\n"
        ).format(w=w, h=h, num_cuts=len(cuts))
        
        for i, cut in enumerate(cuts, 1):
            problem += f"{i}. {cut['type']} {cut['value']}\n"
        
        problem += (
            "\n每次切割后，请计算当前最大玻璃碎片的面积（单位：平方毫米），并将所有结果按顺序排列，用[answer]标签包裹。\n"
            "例如：\n[answer]\n8\n4\n4\n2\n[/answer]"
        )
        return problem

    @staticmethod
    def extract_output(output):
        """从模型输出中提取最后一个[answer]块内的数字序列"""
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        last_answer = answer_blocks[-1].strip()
        solution = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if line:
                try:
                    solution.append(int(line))
                except ValueError:
                    continue
        return solution if solution else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证答案是否与预期结果完全匹配"""
        expected = identity['expected_areas']
        return isinstance(solution, list) and solution == expected

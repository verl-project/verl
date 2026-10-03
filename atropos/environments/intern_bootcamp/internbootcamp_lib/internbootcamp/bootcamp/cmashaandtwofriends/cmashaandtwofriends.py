"""# 

### 谜题描述
Recently, Masha was presented with a chessboard with a height of n and a width of m.

The rows on the chessboard are numbered from 1 to n from bottom to top. The columns are numbered from 1 to m from left to right. Therefore, each cell can be specified with the coordinates (x,y), where x is the column number, and y is the row number (do not mix up).

Let us call a rectangle with coordinates (a,b,c,d) a rectangle lower left point of which has coordinates (a,b), and the upper right one — (c,d).

The chessboard is painted black and white as follows:

<image> An example of a chessboard.

Masha was very happy with the gift and, therefore, invited her friends Maxim and Denis to show off. The guys decided to make her a treat — they bought her a can of white and a can of black paint, so that if the old board deteriorates, it can be repainted. When they came to Masha, something unpleasant happened: first, Maxim went over the threshold and spilled white paint on the rectangle (x_1,y_1,x_2,y_2). Then after him Denis spilled black paint on the rectangle (x_3,y_3,x_4,y_4).

To spill paint of color color onto a certain rectangle means that all the cells that belong to the given rectangle become color. The cell dyeing is superimposed on each other (if at first some cell is spilled with white paint and then with black one, then its color will be black).

Masha was shocked! She drove away from the guests and decided to find out how spoiled the gift was. For this, she needs to know the number of cells of white and black color. Help her find these numbers!

Input

The first line contains a single integer t (1 ≤ t ≤ 10^3) — the number of test cases.

Each of them is described in the following format:

The first line contains two integers n and m (1 ≤ n,m ≤ 10^9) — the size of the board.

The second line contains four integers x_1, y_1, x_2, y_2 (1 ≤ x_1 ≤ x_2 ≤ m, 1 ≤ y_1 ≤ y_2 ≤ n) — the coordinates of the rectangle, the white paint was spilled on.

The third line contains four integers x_3, y_3, x_4, y_4 (1 ≤ x_3 ≤ x_4 ≤ m, 1 ≤ y_3 ≤ y_4 ≤ n) — the coordinates of the rectangle, the black paint was spilled on.

Output

Output t lines, each of which contains two numbers — the number of white and black cells after spilling paint, respectively.

Example

Input


5
2 2
1 1 2 2
1 1 2 2
3 4
2 2 3 2
3 1 4 3
1 5
1 1 5 1
3 1 5 1
4 4
1 1 4 2
1 3 4 4
3 4
1 2 4 2
2 1 3 3


Output


0 4
3 9
2 3
8 8
4 8

Note

Explanation for examples:

The first picture of each illustration shows how the field looked before the dyes were spilled. The second picture of each illustration shows how the field looked after Maxim spoiled white dye (the rectangle on which the dye was spilled is highlighted with red). The third picture in each illustration shows how the field looked after Denis spoiled black dye (the rectangle on which the dye was spilled is highlighted with red).

In the first test, the paint on the field changed as follows:

<image>

In the second test, the paint on the field changed as follows:

<image>

In the third test, the paint on the field changed as follows:

<image>

In the fourth test, the paint on the field changed as follows:

<image>

In the fifth test, the paint on the field changed as follows:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
input = raw_input

def calc_in_rect(r1, c1, r2, c2):
    n = r2 - r1 + 1
    m = c2 - c1 + 1
    if n <= 0 or m <= 0:
        return [0] * 3

    ret = [0, 0, max(0, n * m)]

    ret[0] = ret[1] = n * m // 2
    if (n * m) & 1:
        ret[(r1 + c1) & 1] += 1

    return ret


def solve():
    n, m = map(int, input().split())
    c1, r1, c2, r2 = map(int, input().split())

    c3, r3, c4, r4 = map(int, input().split())

    inter = [
            max(r1, r3),
            max(c1, c3),
            min(r2, r4),
            min(c2, c4)
            ]

    calc_s = calc_in_rect(r1, c1, r2, c2)
    calc_d = calc_in_rect(r3, c3, r4, c4)
    calc_inter = calc_in_rect(*inter)

    num_wh = calc_in_rect(1, 1, n, m)[0]
#    print('*' * 20, num_wh)
#    print(inter)

    num_wh += calc_s[1]
#    print('*' * 20, num_wh)
    num_wh -= calc_inter[2]
#    print('*' * 20, num_wh)

    num_wh -= calc_d[0] - calc_inter[0]

#    print('*' * 20, num_wh)
    print num_wh, (n * m) - num_wh



for i in range(int(input())):
    solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cmashaandtwofriendsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_n = params.get('max_n', 10**9)
        self.max_m = params.get('max_m', 10**9)
        super().__init__(**params)  # 显式传递剩余参数
    
    def case_generator(self):
        n = random.randint(1, min(self.max_n, 10**5))  # 限制测试生成范围
        m = random.randint(1, min(self.max_m, 10**5))
        
        # 生成合法矩形区域
        def gen_rect(max_w, max_h):
            x1 = random.randint(1, max_w)
            x2 = random.randint(x1, max_w)
            y1 = random.randint(1, max_h)
            y2 = random.randint(y1, max_h)
            return [x1, y1, x2, y2]
        
        return {
            'n': n,
            'm': m,
            'white_rect': gen_rect(m, n),
            'black_rect': gen_rect(m, n)
        }
    
    @staticmethod
    def prompt_func(case) -> str:
        params = {
            'n': case['n'],
            'm': case['m'],
            'wx1': case['white_rect'][0],
            'wy1': case['white_rect'][1],
            'wx2': case['white_rect'][2],
            'wy2': case['white_rect'][3],
            'bx1': case['black_rect'][0],
            'by1': case['black_rect'][1],
            'bx2': case['black_rect'][2],
            'by2': case['black_rect'][3]
        }
        return f"""棋盘尺寸：{params['n']}行{params['m']}列。初始颜色规则：(x+y)为偶数是白格，否则黑格。

Maxim在矩形区域(x1={params['wx1']}, y1={params['wy1']})到(x2={params['wx2']}, y2={params['wy2']})泼白漆。
Denis在矩形区域(x1={params['bx1']}, y1={params['by1']})到(x2={params['bx2']}, y2={params['by2']})泼黑漆。

求最终白格和黑格数量，答案格式：[answer]白格数 黑格数[/answer]。"""

    @staticmethod
    def extract_output(output):
        # 增强格式匹配鲁棒性
        patterns = [
            r'\[answer\]([\d,;|]+\s+[\d,;|]+)\[\/?answer\]',  # 宽松匹配标签
            r'答案\s*:\s*(\d+)\s+(\d+)'                      # 中文格式
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                last = matches[-1]
                if isinstance(last, tuple):
                    return f"{last[0]} {last[1]}"
                else:
                    nums = re.findall(r'\d+', last)
                    if len(nums) >= 2:
                        return f"{nums[-2]} {nums[-1]}"
        return None

    @classmethod
    def _verify_correction(cls, solution, case):
        try:
            sol_white, sol_black = map(int, solution.split())
        except:
            return False

        # 参考算法实现
        def calc(r1, c1, r2, c2):
            if r1 > r2 or c1 > c2:
                return (0, 0)
            total = (r2 - r1 + 1) * (c2 - c1 + 1)
            white = total // 2
            if total % 2:
                white += (r1 + c1) % 2 == 0
            return (white, total - white)

        # 棋盘参数
        n, m = case['n'], case['m']
        w_rect = case['white_rect']
        b_rect = case['black_rect']
        
        # 初始白格数
        total = n * m
        initial_white = (total + 1) // 2  # 标准棋盘计算方式

        # 第一次染色（白漆）
        w_area = (w_rect[2] - w_rect[0] + 1) * (w_rect[3] - w_rect[1] + 1)
        w_initial_white, w_initial_black = calc(w_rect[0], w_rect[1], w_rect[2], w_rect[3])
        after_white = initial_white + w_initial_black

        # 第二次染色（黑漆） 
        b_area = (b_rect[2] - b_rect[0] + 1) * (b_rect[3] - b_rect[1] + 1)
        b_initial_white, _ = calc(b_rect[0], b_rect[1], b_rect[2], b_rect[3])

        # 计算重叠区域
        x_overlap = (
            max(w_rect[0], b_rect[0]),
            min(w_rect[2], b_rect[2])
        )
        y_overlap = (
            max(w_rect[1], b_rect[1]),
            min(w_rect[3], b_rect[3])
        )
        overlap_white, overlap_black = calc(x_overlap[0], y_overlap[0], x_overlap[1], y_overlap[1])
        overlap_area = (x_overlap[1] - x_overlap[0] + 1) * (y_overlap[1] - y_overlap[0] + 1) if x_overlap[0] <= x_overlap[1] and y_overlap[0] <= y_overlap[1] else 0

        # 最终计算
        final_white = after_white - overlap_area - (b_initial_white - overlap_white)
        final_black = total - final_white

        return (sol_white == final_white) and (sol_black == final_black)

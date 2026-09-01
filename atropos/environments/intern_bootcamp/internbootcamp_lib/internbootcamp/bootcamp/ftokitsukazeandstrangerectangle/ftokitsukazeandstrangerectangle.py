"""# 

### 谜题描述
There are n points on the plane, the i-th of which is at (x_i, y_i). Tokitsukaze wants to draw a strange rectangular area and pick all the points in the area.

The strange area is enclosed by three lines, x = l, y = a and x = r, as its left side, its bottom side and its right side respectively, where l, r and a can be any real numbers satisfying that l < r. The upper side of the area is boundless, which you can regard as a line parallel to the x-axis at infinity. The following figure shows a strange rectangular area.

<image>

A point (x_i, y_i) is in the strange rectangular area if and only if l < x_i < r and y_i > a. For example, in the above figure, p_1 is in the area while p_2 is not.

Tokitsukaze wants to know how many different non-empty sets she can obtain by picking all the points in a strange rectangular area, where we think two sets are different if there exists at least one point in one set of them but not in the other.

Input

The first line contains a single integer n (1 ≤ n ≤ 2 × 10^5) — the number of points on the plane.

The i-th of the next n lines contains two integers x_i, y_i (1 ≤ x_i, y_i ≤ 10^9) — the coordinates of the i-th point.

All points are distinct.

Output

Print a single integer — the number of different non-empty sets of points she can obtain.

Examples

Input

3
1 1
1 2
1 3


Output

3


Input

3
1 1
2 1
3 1


Output

6


Input

4
2 1
2 2
3 1
3 2


Output

6

Note

For the first example, there is exactly one set having k points for k = 1, 2, 3, so the total number is 3.

For the second example, the numbers of sets having k points for k = 1, 2, 3 are 3, 2, 1 respectively, and their sum is 6.

For the third example, as the following figure shows, there are

  * 2 sets having one point; 
  * 3 sets having two points; 
  * 1 set having four points. 



Therefore, the number of different non-empty sets in this example is 2 + 3 + 0 + 1 = 6.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
class seg:
    def __init__(self,n):
        self.m = 1
        while self.m < n: self.m *= 2
        self.data = [0]*(2*self.m)
    def add(self, ind, val):
        ind += self.m
        while ind:
            self.data[ind] += val
            ind >>= 1

    def summa(self, l, r):
        l += self.m
        r += self.m
        s = 0
        while l<r:
            if l&1:
                s += self.data[l]
                l += 1
            if r&1:
                r -= 1
                s += self.data[r]
            l >>= 1
            r >>= 1
        return s

def main():
    inp = readnumbers()
    ii = 0

    n = inp[ii]
    ii += 1

    X = inp[ii :ii + 2*n:2]
    Y = inp[ii + 1:ii + 2*n:2]
    ii += 2*n
    
    X2 = sorted(X)
    Y2 = sorted(Y)
    xmap = {}
    for x in X2:
        if x not in xmap:
            xmap[x] = len(xmap)
    ymap = {}
    for y in Y2:
        if y not in ymap:
            ymap[y] = len(ymap)
    X = [xmap[x] for x in X]
    Y = [ymap[y] for y in Y]

    Ybuckets = [[] for _ in range(len(ymap))]
    for i in range(n):
        Ybuckets[Y[i]].append(X[i])

    for yvec in Ybuckets:
        yvec.sort()

    ways = 0
    sumseg = seg(len(xmap))
    for yvec in reversed(Ybuckets):
        for x in yvec:
            if sumseg.summa(x, x+1)==0:
                sumseg.add(x,1)
        old_x = -1
        for x in yvec:
            ways += sumseg.summa(old_x + 1, x + 1) * (sumseg.summa(x + 1, len(xmap)) + 1)
            old_x = x
    print ways



######## Python 2 and 3 footer by Pajenegod and c1729

# Note because cf runs old PyPy3 version which doesn't have the sped up
# unicode strings, PyPy3 strings will many times be slower than pypy2.
# There is a way to get around this by using binary strings in PyPy3
# but its syntax is different which makes it kind of a mess to use.

# So on cf, use PyPy2 for best string performance.

py2 = round(0.5)
if py2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    range = xrange

import os, sys
from io import IOBase, BytesIO

BUFSIZE = 8192
class FastIO(BytesIO):
    newlines = 0

    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.writable = \"x\" in file.mode or \"w\" in file.mode
        self.write = super(FastIO, self).write if self.writable else None

    def _fill(self):
        s = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
        self.seek((self.tell(), self.seek(0,2), super(FastIO, self).write(s))[0])
        return s

    def read(self):
        while self._fill(): pass
        return super(FastIO,self).read()

    def readline(self):
        while self.newlines == 0:
            s = self._fill(); self.newlines = s.count(b\"\n\") + (not s)
        self.newlines -= 1
        return super(FastIO, self).readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.getvalue())
            self.truncate(0), self.seek(0)

class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        if py2:
            self.write = self.buffer.write
            self.read = self.buffer.read
            self.readline = self.buffer.readline
        else:
            self.write = lambda s:self.buffer.write(s.encode('ascii'))
            self.read = lambda:self.buffer.read().decode('ascii')
            self.readline = lambda:self.buffer.readline().decode('ascii')


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip('\r\n')

# Cout implemented in Python
import sys
class ostream:
    def __lshift__(self,a):
        sys.stdout.write(str(a))
        return self
cout = ostream()
endl = '\n'

# Read all remaining integers in stdin, type is given by optional argument, this is fast
def readnumbers(zero = 0):
    conv = ord if py2 else lambda x:x
    A = []; numb = zero; sign = 1; i = 0; s = sys.stdin.buffer.read()
    try:
        while True:
            if s[i] >= b'0' [0]:
                numb = 10 * numb + conv(s[i]) - 48
            elif s[i] == b'-' [0]: sign = -1
            elif s[i] != b'\r' [0]:
                A.append(sign*numb)
                numb = zero; sign = 1
            i += 1
    except:pass
    if s and s[-1] >= b'0' [0]:
        A.append(sign*numb)
    return A

if __name__== \"__main__\":
  main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class SegmentTree:
    def __init__(self, size):
        self.m = 1
        while self.m < size:
            self.m <<= 1
        self.data = [0] * (2 * self.m)
    
    def update(self, index, value):
        index += self.m
        while index > 0:
            self.data[index] += value
            index >>= 1
    
    def query_range(self, l, r):
        res = 0
        l += self.m
        r += self.m
        while l < r:
            if l % 2 == 1:
                res += self.data[l]
                l += 1
            if r % 2 == 1:
                r -= 1
                res += self.data[r]
            l >>= 1
            r >>= 1
        return res

def calculate_answer(points):
    if not points:
        return 0
    
    # 离散化坐标
    x_coords = sorted({x for x, y in points})
    y_coords = sorted({y for x, y in points})
    
    x_map = {x: i for i, x in enumerate(x_coords)}
    y_map = {y: i for i, y in enumerate(y_coords)}
    
    # 按y分层存储x坐标
    y_buckets = [[] for _ in range(len(y_coords))]
    for x, y in points:
        y_idx = y_map[y]
        y_buckets[y_idx].append(x_map[x])
    
    for bucket in y_buckets:
        bucket.sort()
    
    total = 0
    st = SegmentTree(len(x_coords))
    
    # 按y降序处理
    for bucket in reversed(y_buckets):
        # 添加当前层的点
        for x in bucket:
            if st.query_range(x, x+1) == 0:
                st.update(x, 1)
        
        prev_x = -1
        for x in bucket:
            # 计算左区域贡献
            left = st.query_range(prev_x + 1, x + 1)
            # 计算右区域贡献（包括无穷大情况）
            right = st.query_range(x + 1, len(x_coords)) + 1
            total += left * right
            prev_x = x
    
    return total

class Ftokitsukazeandstrangerectanglebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, x_max=100, y_max=100):
        self.min_n = min_n
        self.max_n = max_n
        self.x_max = x_max
        self.y_max = y_max
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        points = set()
        while len(points) < n:
            x = random.randint(1, self.x_max)
            y = random.randint(1, self.y_max)
            points.add((x, y))
        points = list(points)
        answer = calculate_answer(points)
        return {
            'n': n,
            'points': points,
            'answer': answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        points = question_case['points']
        n = question_case['n']
        input_lines = [f"{x} {y}" for x, y in points]
        input_str = '\n'.join([str(n)] + input_lines)
        return f"""请解决以下几何谜题，输出最终答案的数值。

题目描述：
平面上有{n}个互不相同的点。定义一个特殊矩形区域：由三条直线x=l（左边界）、x=r（右边界，满足l<r）和y=a（底边界）围成，顶部无限延伸。点(x_i, y_i)位于区域内当且仅当l < x_i < r且y_i > a。求可以形成的不同非空点集的数量。

输入格式：
第一行：n
接下来n行：每行两个整数x_i y_i

输入数据：
{input_str}

请将答案数值置于[answer]标签内，例如：[answer]42[/answer]。确保结果为整数。"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']

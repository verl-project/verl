"""# 

### 谜题描述
On a plane are n points (xi, yi) with integer coordinates between 0 and 106. The distance between the two points with numbers a and b is said to be the following value: <image> (the distance calculated by such formula is called Manhattan distance).

We call a hamiltonian path to be some permutation pi of numbers from 1 to n. We say that the length of this path is value <image>.

Find some hamiltonian path with a length of no more than 25 × 108. Note that you do not have to minimize the path length.

Input

The first line contains integer n (1 ≤ n ≤ 106).

The i + 1-th line contains the coordinates of the i-th point: xi and yi (0 ≤ xi, yi ≤ 106).

It is guaranteed that no two points coincide.

Output

Print the permutation of numbers pi from 1 to n — the sought Hamiltonian path. The permutation must meet the inequality <image>.

If there are multiple possible answers, print any of them.

It is guaranteed that the answer exists.

Examples

Input

5
0 7
8 10
3 4
5 0
9 12


Output

4 3 1 2 5 

Note

In the sample test the total distance is:

<image>

(|5 - 3| + |0 - 4|) + (|3 - 0| + |4 - 7|) + (|0 - 8| + |7 - 10|) + (|8 - 9| + |10 - 12|) = 2 + 4 + 3 + 3 + 8 + 3 + 1 + 2 = 26

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long MAXN = 1e6 + 1, SQ = 1200;
int n;
int x[MAXN], y[MAXN], a[MAXN];
bool cmp(int i, int j) {
  short int tmp = x[i] / SQ, tmpp = x[j] / SQ;
  if (tmp ^ tmpp)
    return x[i] < x[j];
  else {
    if (tmp & 1)
      return y[i] > y[j];
    else
      return y[i] < y[j];
  }
}
int main() {
  ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0);
  cin >> n;
  for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
  for (int i = 0; i < n; i++) a[i] = i;
  sort(a, a + n, cmp);
  for (int i = 0; i < n; i++) cout << a[i] + 1 << ' ';
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Epointsonplanebootcamp(Basebootcamp):
    def __init__(self, n=5, SQ=1200):
        self.n = n
        self.SQ = SQ
    
    def case_generator(self):
        n = self.n
        SQ = self.SQ
        points_set = set()
        while len(points_set) < n:
            x = random.randint(0, 10**6)
            y = random.randint(0, 10**6)
            points_set.add((x, y))
        points = [list(p) for p in points_set]
        return {
            'n': n,
            'points': points,
            'SQ': SQ
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])]
        for point in question_case['points']:
            input_lines.append(f"{point[0]} {point[1]}")
        input_example = '\n'.join(input_lines)
        prompt = f"""你是一个算法竞赛专家，需要解决一个哈密顿路径问题。题目要求找到一条路径，总曼哈顿距离不超过25×10^8。

输入格式：
- 首行为点数n（1 ≤ n ≤ 1e6)
- 接下来n行每行两个整数，表示点的x和y坐标

输出格式：
- 一个1到n的排列，用空格分隔

当前问题输入：
{input_example}

请将答案置于[answer]和[/answer]之间。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        parts = last_answer.split()
        try:
            solution = list(map(int, parts))
            return solution
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        points = identity['points']
        
        # 验证排列有效性
        if len(solution) != n or sorted(solution) != list(range(1, n+1)):
            return False
        
        # 实际计算总距离
        total = 0
        for i in range(n-1):
            idx1 = solution[i] - 1
            idx2 = solution[i+1] - 1
            x1, y1 = points[idx1]
            x2, y2 = points[idx2]
            total += abs(x1 - x2) + abs(y1 - y2)
            if total > 25e8:
                return False
        return total <= 25e8

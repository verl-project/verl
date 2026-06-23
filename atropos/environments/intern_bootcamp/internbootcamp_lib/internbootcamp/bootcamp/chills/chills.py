"""# 

### 谜题描述
Welcome to Innopolis city. Throughout the whole year, Innopolis citizens suffer from everlasting city construction. 

From the window in your room, you see the sequence of n hills, where i-th of them has height ai. The Innopolis administration wants to build some houses on the hills. However, for the sake of city appearance, a house can be only built on the hill, which is strictly higher than neighbouring hills (if they are present). For example, if the sequence of heights is 5, 4, 6, 2, then houses could be built on hills with heights 5 and 6 only.

The Innopolis administration has an excavator, that can decrease the height of an arbitrary hill by one in one hour. The excavator can only work on one hill at a time. It is allowed to decrease hills up to zero height, or even to negative values. Increasing height of any hill is impossible. The city administration wants to build k houses, so there must be at least k hills that satisfy the condition above. What is the minimum time required to adjust the hills to achieve the administration's plan?

However, the exact value of k is not yet determined, so could you please calculate answers for all k in range <image>? Here <image> denotes n divided by two, rounded up.

Input

The first line of input contains the only integer n (1 ≤ n ≤ 5000)—the number of the hills in the sequence.

Second line contains n integers ai (1 ≤ ai ≤ 100 000)—the heights of the hills in the sequence.

Output

Print exactly <image> numbers separated by spaces. The i-th printed number should be equal to the minimum number of hours required to level hills so it becomes possible to build i houses.

Examples

Input

5
1 1 1 1 1


Output

1 2 2 


Input

3
1 2 3


Output

0 2 


Input

5
1 2 3 2 2


Output

0 1 3 

Note

In the first example, to get at least one hill suitable for construction, one can decrease the second hill by one in one hour, then the sequence of heights becomes 1, 0, 1, 1, 1 and the first hill becomes suitable for construction.

In the first example, to get at least two or at least three suitable hills, one can decrease the second and the fourth hills, then the sequence of heights becomes 1, 0, 1, 0, 1, and hills 1, 3, 5 become suitable for construction.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 5003;
const int inf = 1000000007;
int n, a[N], f[N][N], s[N][N];
int main() {
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; i++) {
    scanf(\"%d\", &a[i]);
  }
  for (int i = 0; i <= n + 1; i++)
    for (int j = 0; j <= n + 1; j++) f[i][j] = inf;
  for (int i = 0; i <= n + 1; i++)
    for (int j = 0; j <= n + 1; j++) s[i][j] = inf;
  a[0] = 0;
  a[n + 1] = 0;
  s[0][0] = 0;
  for (int i = 0; i <= n; i++)
    for (int j = 0; j <= n; j++) {
      if (f[i][j] < inf) {
        if (i + 2 <= n) {
          f[i + 2][j + 1] = min(f[i + 2][j + 1],
                                f[i][j] + min(a[i + 1], a[i] - 1) -
                                    min(min(a[i + 1], a[i] - 1), a[i + 2] - 1) +
                                    a[i + 3] - min(a[i + 3], a[i + 2] - 1));
        }
        s[i + 1][j] = min(s[i + 1][j], f[i][j]);
      }
      if (s[i][j] < inf) {
        f[i + 1][j + 1] = min(f[i + 1][j + 1],
                              s[i][j] + a[i + 2] - min(a[i + 2], a[i + 1] - 1) +
                                  a[i] - min(a[i], a[i + 1] - 1));
        s[i + 1][j] = min(s[i + 1][j], s[i][j]);
      }
    }
  for (int i = 1; i <= (n + 1) / 2; i++) printf(\"%d \", s[n + 1][i]);
  printf(\"\n\");
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import ceil

class Chillsbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=8, max_height=20):
        """
        初始化参数：
        min_n: 最小山丘数量（设为3避免边界情况）
        max_n: 最大山丘数量
        max_height: 山丘最大初始高度
        """
        self.min_n = min_n
        self.max_n = max_n
        self.max_height = max_height
    
    def case_generator(self):
        """生成有效案例，确保n≥3"""
        n = random.randint(self.min_n, self.max_n)
        a = [random.randint(1, self.max_height) for _ in range(n)]
        expected_output = self.compute_min_times(n, a)
        return {
            'n': n,
            'a': a,
            'expected_output': expected_output  # 存储预计算的标准答案
        }
    
    @staticmethod
    def compute_min_times(n, a_list):
        """安全版本的动态规划实现"""
        # 扩展数组边界
        a = [0] * (n + 4)
        a[1:n+1] = a_list
        a[0] = a[n+1] = 0
        
        INF = 10**18
        size = n + 4  # 扩展数组尺寸
        f = [[INF]*(n+4) for _ in range(n+4)]
        s = [[INF]*(n+4) for _ in range(n+4)]
        s[0][0] = 0
        
        for i in range(n+1):
            for j in range(n+1):
                if f[i][j] < INF:
                    # 处理i+2的情况
                    if i + 2 <= n and i+3 < len(a):
                        prev = min(a[i+1], a[i]-1)
                        curr = min(prev, a[i+2]-1)
                        delta1 = prev - curr
                        delta2 = a[i+3] - min(a[i+3], a[i+2]-1)
                        if i+2 < len(f) and j+1 < len(f[0]):
                            f[i+2][j+1] = min(f[i+2][j+1], f[i][j] + delta1 + delta2)
                    
                    # 更新s数组
                    if i+1 < len(s):
                        s[i+1][j] = min(s[i+1][j], f[i][j])
                
                if s[i][j] < INF:
                    # 确保i+2不越界
                    if i+1 < len(a)-1 and i < len(a)-2:
                        delta = (a[i+2] - min(a[i+2], a[i+1]-1)) + (a[i] - min(a[i], a[i+1]-1))
                        if i+1 < len(f) and j+1 < len(f[0]):
                            f[i+1][j+1] = min(f[i+1][j+1], s[i][j] + delta)
                    
                    # 更新s数组
                    if i+1 < len(s):
                        s[i+1][j] = min(s[i+1][j], s[i][j])
        
        # 生成最终结果
        max_k = (n + 1) // 2
        return [s[n+1][k] if k <= max_k else 0 for k in range(1, max_k+1)]
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """生成包含明确格式说明的问题描述"""
        n = question_case['n']
        a = question_case['a']
        return f"""您是Innopolis的城市规划师。现有{n}座山丘，高度为：{' '.join(map(str, a))}。
        
规则说明：
1. 每座房屋必须建在严格高于相邻山丘的位置
2. 每次操作可降低任一山丘1单位高度
3. 需要计算k=1到k={ceil(n/2)}的最小操作时间

输出要求：
用空格分隔的{ceil(n/2)}个整数，放在[answer]标签内。例如：[answer]0 1 3[/answer]"""

    @staticmethod
    def extract_output(output):
        """安全抽取最后一个答案"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return list(map(int, matches[-1].strip().split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格验证答案正确性"""
        expected = identity['expected_output']
        return solution == expected

"""# 

### 谜题描述
<image>

Recently, a wild Krakozyabra appeared at Jelly Castle. It is, truth to be said, always eager to have something for dinner.

Its favorite meal is natural numbers (typically served with honey sauce), or, to be more precise, the zeros in their corresponding decimal representations. As for other digits, Krakozyabra dislikes them; moreover, they often cause it indigestion! So, as a necessary precaution, Krakozyabra prefers to sort the digits of a number in non-descending order before proceeding to feast. Then, the leading zeros of the resulting number are eaten and the remaining part is discarded as an inedible tail.

For example, if Krakozyabra is to have the number 57040 for dinner, its inedible tail would be the number 457.

Slastyona is not really fond of the idea of Krakozyabra living in her castle. Hovewer, her natural hospitality prevents her from leaving her guest without food. Slastyona has a range of natural numbers from L to R, which she is going to feed the guest with. Help her determine how many distinct inedible tails are going to be discarded by Krakozyabra by the end of the dinner.

Input

In the first and only string, the numbers L and R are given – the boundaries of the range (1 ≤ L ≤ R ≤ 1018).

Output

Output the sole number – the answer for the problem.

Examples

Input

1 10


Output

9


Input

40 57


Output

17


Input

157 165


Output

9

Note

In the first sample case, the inedible tails are the numbers from 1 to 9. Note that 10 and 1 have the same inedible tail – the number 1.

In the second sample case, each number has a unique inedible tail, except for the pair 45, 54. The answer to this sample case is going to be (57 - 40 + 1) - 1 = 17.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long INF = 1e17;
const long long maxn = 2e5 + 700;
const int mod = 1e9 + 7;
template <typename T>
inline void read(T &a) {
  char c = getchar();
  T x = 0, f = 1;
  while (!isdigit(c)) {
    if (c == '-') f = -1;
    c = getchar();
  }
  while (isdigit(c)) {
    x = (x << 1) + (x << 3) + c - '0';
    c = getchar();
  }
  a = f * x;
}
long long n, m, p;
char s[maxn];
int a[maxn], b[maxn];
int digit(long long x, int *d) {
  int ans = 0;
  while (x) {
    d[++ans] = x % 10;
    x /= 10;
  }
  return ans;
}
int path[maxn];
int check(int pos, int lbound, int rbound) {
  if (!pos) return 1;
  if (!lbound && !rbound) return 1;
  int l = lbound ? a[pos] : 0;
  int r = rbound ? b[pos] : 9;
  for (int i = l; i <= r; i++) {
    if (path[i] > 0) {
      path[i]--;
      if (check(pos - 1, lbound && i == l, rbound && i == r)) {
        path[i]++;
        return 1;
      }
      path[i]++;
    }
  }
  return 0;
}
int pos, res = 0;
void dfs(int u, int w) {
  if (u == 9) {
    path[u] = w;
    if (check(pos, 1, 1)) res++;
    return;
  }
  for (int i = 0; i <= w; i++) {
    path[u] = i;
    dfs(u + 1, w - i);
  }
}
int main() {
  read(n);
  read(m);
  pos = digit(n, a);
  pos = digit(m, b);
  dfs(0, pos);
  printf(\"%d\n\", res);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Eeverhungrykrakozyabrabootcamp(Basebootcamp):
    def __init__(self, max_case_size=100):
        super().__init__()
        self.max_case_size = max(max_case_size, 1)  # 保证至少1的间隔
        
    def case_generator(self):
        max_L = 10**9
        L = random.randint(1, max_L - 1)  # 确保L+max_case_size不超过max_L
        
        # 动态计算安全范围
        max_possible_R = min(L + self.max_case_size, max_L)
        R = random.randint(L, max_possible_R)
        
        # 优化暴力计算：使用集合生成式
        tails = {
            ''.join(sorted(str(x))).lstrip('0') or '0'
            for x in range(L, R + 1)
        }
        
        return {
            'L': L,
            'R': R,
            'answer': len(tails)
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        L = question_case['L']
        R = question_case['R']
        prompt = f"""你正在解决Eeverhungrykrakozyabra的数学谜题：

规则描述：
1. 对于数字N，将其各位数字按非降序排列（如57040 → 00457）
2. 去除所有前导零（00457 → 457）
3. 若结果为空字符串则视为0（如0000 → 0）

任务要求：
计算区间[{L}, {R}]内所有整数经上述处理后的不同结果数量。

答案格式要求：
将最终答案放在[answer]标签内，如：[answer]42[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']

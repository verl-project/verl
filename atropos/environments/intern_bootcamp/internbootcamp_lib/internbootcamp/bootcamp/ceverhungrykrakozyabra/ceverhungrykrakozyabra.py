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
const int MAXN = 19;
char L[MAXN + 5], R[MAXN + 5];
int x[MAXN + 5], y[MAXN + 5];
int c[10], cc[10];
bool check_less(int i, int nz) {
  if (i == MAXN) {
    if (nz > 0)
      return false;
    else
      return true;
  }
  for (int j = 0; j < y[i]; j++) {
    if (cc[j] > 0) {
      if (MAXN - i >= nz)
        return true;
      else
        return false;
    }
  }
  if (cc[y[i]] > 0) {
    cc[y[i]]--;
    nz--;
    bool ok = check_less(i + 1, nz);
    cc[y[i]]++;
    nz++;
    return ok;
  } else
    return false;
}
bool check_more(int i, int nz) {
  if (i == MAXN) {
    if (nz > 0)
      return false;
    else
      return true;
  }
  for (int j = x[i] + 1; j < 10; j++) {
    if (cc[j] > 0) {
      if (MAXN - i >= nz)
        return true;
      else
        return false;
    }
  }
  if (cc[x[i]] > 0) {
    cc[x[i]]--;
    nz--;
    bool ok = check_more(i + 1, nz);
    cc[x[i]]++;
    nz++;
    return ok;
  } else
    return false;
}
bool check() {
  memcpy(cc, c, sizeof c);
  int nz = 0;
  for (int i = 0; i < 10; i++) nz += cc[i];
  for (int i = 0; i < MAXN; i++) {
    if (x[i] == y[i]) {
      if (cc[x[i]] == 0) return false;
      cc[x[i]]--;
      nz--;
    } else {
      for (int j = x[i] + 1; j < y[i]; j++) {
        if (cc[j] > 0) {
          if (MAXN - i >= nz)
            return true;
          else
            return false;
        }
      }
      if (cc[x[i]] > 0) {
        cc[x[i]]--;
        nz--;
        if (check_more(i + 1, nz)) return true;
        cc[x[i]]++;
        nz++;
      }
      if (cc[y[i]] > 0) {
        cc[y[i]]--;
        nz--;
        if (check_less(i + 1, nz)) return true;
        cc[y[i]]++;
        nz++;
      }
      return false;
    }
  }
  for (int i = 0; i < 10; i++)
    if (cc[i] > 0) return false;
  return true;
}
int solve(int i, int d) {
  if (i == MAXN) {
    if (check())
      return 1;
    else
      return 0;
  } else {
    int r = 0;
    c[d]++;
    r += solve(i + 1, d);
    c[d]--;
    if (d < 9) r += solve(i, d + 1);
    return r;
  }
}
int main() {
  scanf(\"%s %s\", L, R);
  int n = strlen(L);
  int m = strlen(R);
  reverse(L, L + n);
  reverse(R, R + m);
  for (int i = 0; i < MAXN; i++) {
    if (i < n)
      x[i] = L[i] - '0';
    else
      x[i] = 0;
    if (i < m)
      y[i] = R[i] - '0';
    else
      y[i] = 0;
  }
  reverse(x, x + MAXN);
  reverse(y, y + MAXN);
  printf(\"%d\n\", solve(0, 0));
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ceverhungrykrakozyabrabootcamp(Basebootcamp):
    def __init__(self, max_L=1000, max_range=100):
        """
        初始化训练场环境，定义生成范围参数。
        
        参数:
            max_L (int): 生成左边界L的最大值，默认为1000
            max_range (int): 生成范围的最大跨度(L到R)，默认为100
        """
        self.max_L = max_L
        self.max_range = max_range
    
    def case_generator(self):
        """
        生成符合要求的谜题实例，确保L ≤ R且范围合理
        """
        L = random.randint(1, self.max_L)
        R_max = min(L + self.max_range, 10**18)
        R = random.randint(L, R_max)
        return {'L': L, 'R': R}
    
    @staticmethod
    def prompt_func(question_case):
        """
        将问题实例转换为自然语言描述，包含格式要求
        """
        L = question_case['L']
        R = question_case['R']
        return f"""Slastyona需要喂养Ceverhungrykrakozyabra，这个生物会按以下规则处理数字：
1. 将数字各位排序成非降序（例如57040 → 00457）
2. 去除所有前导零（00457 → 457）
3. 余下部分称为"不可食用尾巴"

请计算范围[{L}, {R}]内所有数字处理后产生的不同尾巴数量。

输出要求：
1. 答案必须是整数
2. 将最终答案放在[answer]和[/answer]之间
示例：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取最后一个[answer]标签内容
        """
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案的正确性：遍历所有数字计算实际结果
        """
        L = identity['L']
        R = identity['R']
        unique_tails = set()
        
        for num in range(L, R + 1):
            # 处理数字并标准化表示
            sorted_str = ''.join(sorted(str(num))).lstrip('0')
            # 处理全零情况（根据题目输入限制不会出现）
            unique_tails.add(sorted_str if sorted_str else '0')
        
        return solution == len(unique_tails)

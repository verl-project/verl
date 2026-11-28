"""# 

### 谜题描述
Yaroslav, Andrey and Roman love playing cubes. Sometimes they get together and play cubes for hours and hours! 

Today they got together again and they are playing cubes. Yaroslav took unit cubes and composed them into an a × a × a cube, Andrey made a b × b × b cube and Roman made a c × c × c cube. After that the game was finished and the guys left. But later, Vitaly entered the room. He saw the cubes and wanted to make a cube as well. But what size should the cube be? Of course it should be a large cube with the side of length a + b + c. Besides, Vitaly decided to decompose the cubes built by Yaroslav, Andrey and Roman and compose his own large cube out of them. However, it turned out that the unit cubes he got from destroying the three cubes just weren't enough to make a large cube. We know that Vitaly was short of exactly n cubes. Vitaly got upset, demolished everything and left. As he was leaving, he met Petya and told him that there had been three cubes in the room and that he needed another n unit cubes to make his own large cube.

Petya entered the room and saw the messily scattered cubes. He wanted to make it neat and orderly again. But he only knows that there had been three cubes, made of small unit cubes and that Vitaly needed n more unit cubes to make a large one! Help Petya understand, how many ways of sizes a, b, c are there to restore Yaroslav's, Andrey's and Roman's cubes.

Input

The single line of the input contains integer n (1 ≤ n ≤ 1014). We know that all numbers a, b, c are positive integers.

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

In the single line print the required number of ways. If it turns out that there isn't a single way of suitable sizes of a, b, c, print 0. 

Examples

Input

24


Output

1


Input

648


Output

7


Input

5


Output

0


Input

93163582512000


Output

39090

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long p[10000005];
int idxp;
int main() {
  long long n;
  int i, j;
  long long countt = 0;
  scanf(\"%I64d\", &n);
  if (n % 3 != 0)
    printf(\"0\n\");
  else {
    n = n / 3;
    long long r = (long long)sqrt(n);
    for (i = 1; i <= r; i++) {
      if (n % i == 0) p[idxp++] = i;
    }
    for (i = 0; i < idxp; i++) {
      for (j = i; j < idxp; j++) {
        long long C = n / (p[i] * p[j]);
        if (p[i] * p[j] * C == n) {
          long long A = p[i], B = p[j];
          if ((A + B + C) & 1) continue;
          if (A + B <= C) continue;
          if (C < B) continue;
          if (A == B && B == C)
            countt++;
          else if (A == B || B == C)
            countt += 3;
          else
            countt += 6;
        }
      }
    }
    printf(\"%I64d\n\", countt);
  }
  scanf(\" \");
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import isqrt
from bootcamp import Basebootcamp

def solve(n):
    """
    正确计算满足条件的三元组数量，修正因数枚举逻辑
    """
    if n % 3 != 0:
        return 0
    
    m = n // 3
    if m == 0:
        return 0
    
    # 获取m的所有因数
    factors = set()
    for i in range(1, isqrt(m) + 1):
        if m % i == 0:
            factors.add(i)
            factors.add(m // i)
    factors = sorted(factors)
    
    count = 0
    # 遍历所有有序因数对 (A <= B)
    for i in range(len(factors)):
        A = factors[i]
        for j in range(i, len(factors)):
            B = factors[j]
            if A * B == 0 or m % (A * B) != 0:
                continue
            
            C = m // (A * B)
            # 检查几何条件
            if (A + B + C) % 2 != 0:
                continue
            if A + B <= C:
                continue
            if B > C:  # 保证 B <= C
                continue
            
            # 计数不同排列组合
            if A == B == C:
                count += 1
            elif A == B or B == C:
                count += 3
            else:
                count += 6
    
    return count

class Ecubeproblembootcamp(Basebootcamp):
    def __init__(self, max_n=10**14, seed=None):
        super().__init__()
        self.max_n = max_n  # 支持题目要求的最大n值
        self.rng = random.Random(seed)
    
    def case_generator(self):
        """
        直接生成符合题目范围的n值，覆盖所有情况
        """
        n = self.rng.randint(1, self.max_n)
        return {"n": n, "expected": solve(n)}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case["n"]
        return f"""
三位朋友分别用单位立方体搭建了边长为a、b、c的立方体。Vitaly试图将这些立方体拆解后重组为边长为(a+b+c)的大立方体时，发现缺少{n}个单位立方体。

请计算满足以下条件的正整数解(a,b,c)的数量：
(a + b + c)^3 - a^3 - b^3 - c^3 = {n}

规则说明：
1. a, b, c均为正整数
2. 不同顺序视为不同解（如(1,2,3)和(2,1,3)视为两个解）
3. 只需要给出解的个数，不需要具体数值

请将最终答案用[answer]标签包裹，例如：[answer]42[/answer]
        """.strip()
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity["expected"]

# 测试验证
if __name__ == "__main__":
    # 验证已知测试用例
    assert solve(24) == 1    # 原题样例1
    assert solve(648) ==7    # 原题样例2
    assert solve(5) ==0      # 原题样例3
    assert solve(93163582512000) ==39090  # 原题大数样例

    # 测试训练场功能
    bootcamp = Ecubeproblembootcamp(max_n=10**5, seed=42)
    test_case = bootcamp.case_generator()
    prompt = Ecubeproblembootcamp.prompt_func(test_case)
    print(f"生成的测试案例n={test_case['n']}, 预期答案={test_case['expected']}")

    # 测试答案提取
    sample_output = "可能的解有3种[answer]3[/answer] 最终答案应该是[answer]42[/answer]"
    assert Ecubeproblembootcamp.extract_output(sample_output) == 42

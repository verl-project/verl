"""# 

### 谜题描述
Alexandra has a paper strip with n numbers on it. Let's call them ai from left to right.

Now Alexandra wants to split it into some pieces (possibly 1). For each piece of strip, it must satisfy:

  * Each piece should contain at least l numbers.
  * The difference between the maximal and the minimal number on the piece should be at most s.



Please help Alexandra to find the minimal number of pieces meeting the condition above.

Input

The first line contains three space-separated integers n, s, l (1 ≤ n ≤ 105, 0 ≤ s ≤ 109, 1 ≤ l ≤ 105).

The second line contains n integers ai separated by spaces ( - 109 ≤ ai ≤ 109).

Output

Output the minimal number of strip pieces.

If there are no ways to split the strip, output -1.

Examples

Input

7 2 2
1 3 1 2 4 1 2


Output

3


Input

7 2 2
1 100 1 100 1 100 1


Output

-1

Note

For the first sample, we can split the strip into 3 pieces: [1, 3, 1], [2, 4], [1, 2].

For the second sample, we can't let 1 and 100 be on the same piece, so no solution exists.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/python
# coding: utf-8

def make(a, n, s, l):
  pieces = []
  i = 1
  tmpMin = a[0]
  tmpMax = a[0]
  tmpPieces = [a[0]]
  while i < n:
    if abs(a[i] - tmpMin) <= s and abs(a[i] - tmpMax) <= s:
      tmpPieces.append(a[i])
      if a[i] < tmpMin:
        tmpMin = a[i]
      elif a[i] > tmpMax:
        tmpMax = a[i]
    else:
      pieces.append(tmpPieces)
      tmpPieces = [a[i]]
      tmpMin = a[i]
      tmpMax = a[i]
    i += 1
  pieces.append(tmpPieces)
  fail = False
  for j in xrange(len(pieces)):
    if len(pieces[j]) < l:
      if j > 0:
        preVpc = pieces[j - 1]
        minJ = min(pieces[j])
        maxJ = max(pieces[j])
        while len(pieces[j]) < l:
          tmp = preVpc.pop()
          if abs(tmp - minJ) <= s and abs(tmp - maxJ) <= s:
            pieces[j].insert(0, tmp)
            if tmp < minJ:
              minJ = tmp
            elif tmp > maxJ:
              maxJ = tmp
          else:
            return -1
          if len(preVpc) < l:
            return -1
      else:
        return -1
  return len(pieces)
n, s, l = map(int, raw_input().split())
a = map(int, raw_input().split())
ans = make(a, n, s, l)
if ans < 0:
  a.reverse()
  ans = make(a, n, s, l)
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from math import inf

class Dstripbootcamp(Basebootcamp):
    def __init__(self, min_n=5, max_n=15, min_s=1, max_s=20, min_l=2, max_l=5, ensure_solvable=False):
        # 参数有效性验证
        assert min_l <= max_l <= max_n, "Invalid length constraints"
        self.min_n = min_n
        self.max_n = max_n
        self.min_s = min_s
        self.max_s = max_s
        self.min_l = min_l
        self.max_l = max_l
        self.ensure_solvable = ensure_solvable

    def case_generator(self):
        for _ in range(1000):  # 防止无限循环
            n = random.randint(self.min_n, self.max_n)
            l = random.randint(self.min_l, min(self.max_l, n))
            s = random.randint(self.min_s, self.max_s)
            
            # 生成有效数组的三种模式
            if random.random() < 0.3 and self.ensure_solvable:
                # 模式1：保证有解的序列（分段生成）
                a = []
                remaining = n
                while remaining > 0:
                    seg_len = random.randint(l, min(3*l, remaining))
                    base = random.randint(0, 50)
                    a.extend([base + random.randint(0, s) for _ in range(seg_len)])
                    remaining -= seg_len
                random.shuffle(a)
            elif random.random() < 0.5:
                # 模式2：随机数组带验证
                a = [random.randint(0, 100) for _ in range(n)]
            else:
                # 模式3：刻意构造无解情况
                a = [0]*(l-1) + [100]*(n-l+1)
                s = 10
                l += 1  # 确保无法满足长度要求
                
            # 计算正确答案
            dp = [inf] * (n+1)
            dp[0] = 0
            for i in range(1, n+1):
                for j in range(max(0, i-3*l), i-l+1):
                    if j < 0: continue
                    seg = a[j:i]
                    if max(seg)-min(seg) <= s:
                        dp[i] = min(dp[i], dp[j]+1)
            ans = dp[n] if dp[n] != inf else -1
            
            # 二次验证
            if self.ensure_solvable and ans == -1:
                continue
            if not self.ensure_solvable and ans != -1:
                # 反向验证无解情况
                if all(max(a[i:i+l]) - min(a[i:i+l]) > s for i in range(n-l+1)):
                    ans = -1
            return {
                'n': n, 's': s, 'l': l,
                'a': a, 'correct_answer': ans
            }
        raise RuntimeError("Failed to generate valid case")

    @staticmethod
    def prompt_func(case):
        return (
            f"Split the sequence into minimal pieces where each:\n"
            f"- Contains ≥{case['l']} numbers\n- Max-min ≤{case['s']}\n\n"
            f"Input:\n{case['n']} {case['s']} {case['l']}\n"
            f"{' '.join(map(str, case['a']))}\n\n"
            "Output the minimal number of pieces or -1 if impossible.\n"
            "Format: [answer]result[/answer]"
        )

    @staticmethod
    def extract_output(output):
        # 严格匹配最终答案
        matches = re.findall(
            r'\[answer\][\s]*(-?\d+)[\s]*\[/answer\]', 
            output, 
            flags=re.IGNORECASE
        )
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 动态验证算法
        n, s, l, a = identity['n'], identity['s'], identity['l'], identity['a']
        
        # 边界条件检查
        if solution == -1:
            # 验证确实无解
            dp = [inf]*(n+1)
            dp[0] = 0
            for i in range(1, n+1):
                for j in range(max(0, i-3*l), i-l+1):
                    if j < 0: continue
                    seg = a[j:i]
                    if max(seg)-min(seg) <= s:
                        dp[i] = min(dp[i], dp[j]+1)
            return dp[n] == inf
        
        # 正向验证
        current_pos = 0
        pieces = []
        while current_pos < n:
            found = False
            for end in range(min(n, current_pos+l), n+1):
                seg = a[current_pos:end]
                if len(seg) >= l and max(seg)-min(seg) <= s:
                    pieces.append(seg)
                    current_pos = end
                    found = True
                    break
            if not found:
                return False
        return len(pieces) == solution

# 测试用例验证函数
def _test():
    bootcamp = Dstripbootcamp(ensure_solvable=True)
    case = bootcamp.case_generator()
    print("Generated case:", case)
    print("Prompt:\n", bootcamp.prompt_func(case))
    
    # 测试解法
    assert bootcamp._verify_correction(case['correct_answer'], case), "Validation failed"
    
if __name__ == "__main__":
    _test()

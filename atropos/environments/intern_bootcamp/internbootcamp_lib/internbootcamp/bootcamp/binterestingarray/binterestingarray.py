"""# 

### 谜题描述
We'll call an array of n non-negative integers a[1], a[2], ..., a[n] interesting, if it meets m constraints. The i-th of the m constraints consists of three integers li, ri, qi (1 ≤ li ≤ ri ≤ n) meaning that value <image> should be equal to qi. 

Your task is to find any interesting array of n elements or state that such array doesn't exist.

Expression x&y means the bitwise AND of numbers x and y. In programming languages C++, Java and Python this operation is represented as \"&\", in Pascal — as \"and\".

Input

The first line contains two integers n, m (1 ≤ n ≤ 105, 1 ≤ m ≤ 105) — the number of elements in the array and the number of limits.

Each of the next m lines contains three integers li, ri, qi (1 ≤ li ≤ ri ≤ n, 0 ≤ qi < 230) describing the i-th limit.

Output

If the interesting array exists, in the first line print \"YES\" (without the quotes) and in the second line print n integers a[1], a[2], ..., a[n] (0 ≤ a[i] < 230) decribing the interesting array. If there are multiple answers, print any of them.

If the interesting array doesn't exist, print \"NO\" (without the quotes) in the single line.

Examples

Input

3 1
1 3 3


Output

YES
3 3 3


Input

3 2
1 3 3
1 3 2


Output

NO

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long base = 1331;
const long long N = 1e5 + 1;
template <typename T>
inline void Cin(T& x) {
  char c = getchar();
  x = 0;
  while (c < '0' || c > '9') c = getchar();
  while (c >= '0' && c <= '9') x = x * 10 + c - '0', c = getchar();
}
template <typename T, typename... Args>
inline void Cin(T& a, Args&... args) {
  Cin(a);
  Cin(args...);
}
long long s[N][31], n, m, a[N], l[N], r[N], q[N], res, ST[4 * N];
void build(long long id, long long l, long long r) {
  if (l == r) {
    ST[id] = a[l];
    return;
  }
  long long mid = (l + r) / 2;
  build(id * 2, l, mid);
  build(id * 2 + 1, mid + 1, r);
  ST[id] = ST[id * 2] & ST[id * 2 + 1];
}
void get(long long id, long long l, long long r, long long L, long long R) {
  if (L > r || R < l) {
    return;
  }
  if (L <= l && r <= R) {
    res = res & ST[id];
    return;
  }
  long long mid = (l + r) / 2;
  get(id * 2, l, mid, L, R);
  get(id * 2 + 1, mid + 1, r, L, R);
}
void read(void) {
  cin >> n >> m;
  for (long long i = 1; i <= (long long)(m); ++i) {
    cin >> l[i] >> r[i] >> q[i];
    for (long long j = 0; j < (long long)(31); ++j) {
      if (q[i] >> j & 1 == 1) {
        s[l[i]][j]++;
        s[r[i] + 1][j]--;
      }
    }
  }
  for (long long i = 1; i <= (long long)(n); ++i) {
    for (long long j = 0; j < (long long)(31); ++j) {
      s[i][j] += s[i - 1][j];
      if (s[i][j] > 0) a[i] += (1 << j);
    }
  }
  build(1, 1, n);
  for (long long i = 1; i <= (long long)(m); ++i) {
    res = a[l[i]];
    get(1, 1, n, l[i], r[i]);
    if (res != q[i]) {
      cout << \"NO\";
      return;
    }
  }
  cout << \"YES\" << '\n';
  for (long long i = 1; i <= (long long)(n); ++i) cout << a[i] << ' ';
}
signed main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  read();
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from typing import Dict, Any

class Binterestingarraybootcamp(Basebootcamp):
    def __init__(self, n_min=5, n_max=10, m_min=3, m_max=5, p_unsolvable=0.3):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.p_unsolvable = p_unsolvable
    
    def case_generator(self) -> Dict[str, Any]:
        # 增加更多不可解的情况类型
        if random.random() > self.p_unsolvable:
            # 生成可解案例时确保覆盖不同区间类型
            n = random.randint(self.n_min, self.n_max)
            m = random.randint(self.m_min, self.m_max)
            a = [random.randint(0, (1<<30)-1) for _ in range(n)]
            
            # 生成不同区间类型：全范围/左半段/右半段/随机区间
            intervals = [
                (1, n),  # 全数组
                (1, n//2),  # 左半段
                (n//2+1, n),  # 右半段
                *[tuple(sorted((random.randint(1, n), random.randint(1, n)))) 
                 for _ in range(m-3)]  # 随机区间
            ]
            random.shuffle(intervals)
            
            constraints = []
            for l, r in intervals[:m]:
                q = a[l-1]
                for num in a[l:r]:  # 计算实际的区间AND
                    q &= num
                constraints.append((l, r, q))
            
            return {
                "n": n,
                "m": m,
                "constraints": constraints,
                "solution_exists": True
            }
        else:
            # 生成更丰富的不可解案例
            n = random.randint(3, self.n_max)
            conflict_type = random.choice(['bit', 'composite', 'overlap'])
            
            if conflict_type == 'bit':  # 单一位冲突
                k = random.randint(0, 29)
                constraints = [
                    (1, n, 1 << k),
                    (random.randint(1, n), random.randint(1, n), 
                     random.randint(0, (1 << 30)-1) & ~(1 << k))
                ]
            elif conflict_type == 'composite':  # 复合位冲突
                constraints = [
                    (1, 2, 3),  # binary 11
                    (1, 1, 1),
                    (2, 2, 1),
                    (1, 2, 1)  # 实际AND应为1，但要求3
                ]
                n = 2
            else:  # 区间重叠冲突
                constraints = [
                    (1, 3, 4),   # binary 100
                    (1, 2, 5),   # binary 101
                    (2, 3, 6)    # binary 110
                ]
                n = 3
            
            return {
                "n": n,
                "m": len(constraints),
                "constraints": constraints,
                "solution_exists": False
            }
    
    @staticmethod
    def prompt_func(question_case: Dict) -> str:
        constraints_str = "\n".join(
            f"{l} {r} {q}" for l, r, q in question_case["constraints"]
        )
        return f"""给定数组长度n={question_case['n']}和m={question_case['m']}个约束条件，每个约束形如l r q，要求区间[l,r]的按位与等于q。请判断是否存在满足所有约束的数组，若存在输出YES及数组，否则输出NO。答案放在[answer]和[/answer]之间。

输入数据示例：
{question_case['n']} {question_case['m']}
{constraints_str}

要求：
1. 数组元素必须满足所有区间约束
2. 元素取值范围：[0, 2^30)
3. 按位与操作定义：对应二进制位都为1时结果位才为1

答案格式：
[answer]
YES
a1 a2 ... an
[/answer]
或：
[answer]
NO
[/answer]"""
    
    @staticmethod
    def extract_output(output: str) -> str:
        matches = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            output.replace('\n', ' '), 
            re.DOTALL
        )
        if matches:
            last_answer = matches[-1].strip()
            # 清理多余的空格和换行
            return ' '.join(last_answer.split())
        return None
    
    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict) -> bool:
        if not solution:
            return False
        
        solution = solution.upper().split()
        expected = identity["solution_exists"]
        
        if expected:
            if solution[0] != "YES" or len(solution) != identity["n"] + 1:
                return False
            
            try:
                arr = list(map(int, solution[1:]))
                if any(not (0 <= x < (1<<30)) for x in arr):
                    return False
            except:
                return False
            
            for l, r, q in identity["constraints"]:
                current_and = arr[l-1]
                for num in arr[l:r]:
                    current_and &= num
                    if current_and < q:  # 提前终止优化的AND计算
                        break
                if current_and != q:
                    return False
            return True
        else:
            return solution[0] == "NO"

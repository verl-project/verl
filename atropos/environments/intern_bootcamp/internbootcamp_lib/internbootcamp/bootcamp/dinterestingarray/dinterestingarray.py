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
const int MAXN = 1e5 + 10, STAN = (1 << 30) - 1;
struct QJ {
  int l, r, q;
} q[MAXN];
int n, m;
struct Seg {
  int water[MAXN * 4], sh[MAXN * 4];
  bool fir[MAXN * 4];
  Seg() {
    memset(water, 0, sizeof(water));
    memset(fir, 0, sizeof(fir));
  }
  int _st, _ed, _x, _t;
  void _insert(int num, int l, int r) {
    if (_st <= l && r <= _ed) {
      water[num] |= _x;
      return;
    }
    int mid = (l + r) >> 1;
    if (_st <= mid) _insert(num << 1, l, mid);
    if (_ed >= mid + 1) _insert(num << 1 | 1, mid + 1, r);
  }
  int _swim(int num, int l, int r, int now) {
    int x;
    now |= water[num];
    if (l == r) {
      if (!fir[num]) {
        sh[num] = now;
        fir[num] = true;
      } else
        sh[num] &= now;
      return now;
    }
    int mid = (l + r) >> 1;
    if (_t <= mid)
      now = _swim(num << 1, l, mid, now);
    else
      now = _swim(num << 1 | 1, mid + 1, r, now);
    if (!fir[num]) {
      sh[num] = now;
      fir[num] = true;
    } else
      sh[num] &= now;
    return now;
  }
  int _check(int num, int l, int r) {
    if (l == r) return sh[num];
    if (_st <= l && r <= _ed) return sh[num];
    int mid = (l + r) >> 1;
    int ans = STAN;
    if (_st <= mid) ans &= _check(num << 1, l, mid);
    if (_ed >= mid + 1) ans &= _check(num << 1 | 1, mid + 1, r);
    return ans;
  }
  inline void Insert(int l, int r, int q) {
    _st = l, _ed = r, _x = q;
    _insert(1, 1, n);
  }
  inline int Swim(int t) {
    _t = t;
    return _swim(1, 1, n, 0);
  }
  inline bool Check(int l, int r, int q) {
    _st = l;
    _ed = r;
    return _check(1, 1, n) == q;
  }
} T;
int a[MAXN];
int main() {
  scanf(\"%d%d\", &n, &m);
  for (int i = 1; i <= m; i++) {
    scanf(\"%d%d%d\", &q[i].l, &q[i].r, &q[i].q);
    T.Insert(q[i].l, q[i].r, q[i].q);
  }
  for (int i = 1; i <= n; i++) a[i] = T.Swim(i);
  for (int i = 1; i <= m; i++)
    if (!T.Check(q[i].l, q[i].r, q[i].q)) {
      printf(\"NO\n\");
      return 0;
    }
  printf(\"YES\n\");
  for (int i = 1; i <= n; i++) printf(\"%d \", a[i]);
  printf(\"\n\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dinterestingarraybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_range = params.get('n_range', (3, 10))
        self.m_range = params.get('m_range', (2, 6)) 
        self.bit_width = params.get('bit_width', 8)
        self.qi_max = (1 << self.bit_width) - 1
        self.solvable_prob = params.get('solvable_prob', 0.5)

    def case_generator(self):
        """重构的案例生成逻辑，保证有效性"""
        n = random.randint(*self.n_range)
        m = random.randint(*self.m_range)
        
        # 生成初始有效约束集
        base_case = self._generate_solvable_case(n, m)
        if random.random() < self.solvable_prob:
            return base_case
        
        # 构造矛盾案例：添加不兼容的约束
        conflict_case = self._add_conflict_constraint(base_case)
        solution_exists, possible_a = self._validate_case(conflict_case)
        return {
            **conflict_case,
            'solution_exists': solution_exists,
            'possible_a': possible_a
        }

    def _generate_solvable_case(self, n, m):
        """生成必定有解的案例"""
        a = [random.randint(0, self.qi_max) for _ in range(n)]
        constraints = []
        for _ in range(m-1):
            l = random.randint(1, n)
            r = random.randint(l, n)
            current_and = a[l-1]
            for i in range(l, r):
                current_and &= a[i]
            constraints.append((l, r, current_and))
        
        # 添加全局约束保证解存在
        constraints.append((1, n, current_and))
        return {
            'n': n,
            'm': m,
            'constraints': constraints,
            'solution_exists': True,
            'possible_a': a
        }

    def _add_conflict_constraint(self, case):
        """添加矛盾约束"""
        # 复制原有约束
        new_constraints = case['constraints'][:]
        l, r = self._find_overlap_interval(new_constraints)
        
        # 生成矛盾的约束值
        original_q = new_constraints[0][2]
        conflict_q = original_q ^ (1 << random.randint(0, self.bit_width-1))
        
        # 添加新约束
        new_constraints.append((l, r, conflict_q))
        return {
            'n': case['n'],
            'm': case['m'] + 1,
            'constraints': new_constraints
        }

    def _find_overlap_interval(self, constraints):
        """找到多个约束的重叠区间"""
        intervals = [(l, r) for l, r, _ in constraints]
        max_l = max(l for l, _ in intervals)
        min_r = min(r for _, r in intervals)
        if max_l <= min_r:
            return (max_l, min_r)
        return (1, constraints[0][0])  # 默认返回第一个约束的区间

    def _validate_case(self, case):
        """科学校验案例有效性"""
        n = case['n']
        constraints = case['constraints']
        
        # 初始化各bit位的允许范围
        bit_masks = [0xFFFFFFFF for _ in range(n)]
        
        # 应用所有约束
        for l, r, q in constraints:
            for i in range(l-1, r):
                bit_masks[i] &= q
        
        # 检查所有位置是否可能
        for i in range(n):
            if bit_masks[i] == 0 and not any(
                (l-1 <= i <= r-1 and q == 0) 
                for l, r, q in constraints
            ):
                return False, None
        
        # 验证约束一致性
        for l, r, q in constraints:
            required_bits = q
            possible_and = 0xFFFFFFFF
            for i in range(l-1, r):
                possible_and &= bit_masks[i]
            if (possible_and & required_bits) != required_bits:
                return False, None
        
        # 构造可行解
        solution = [random.randint(0, mask) & mask for mask in bit_masks]
        return True, solution

    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['m']}"]
        for l, r, q in question_case['constraints']:
            input_lines.append(f"{l} {r} {q}")
        input_section = "\n".join(input_lines)
        
        prompt = f"""Solve the array puzzle with bitwise AND constraints. 

Problem Statement:
- Array length: {question_case['n']}
- Number of constraints: {question_case['m']}
- Constraints (l, r, q format):
{input_section}

Requirements:
1. Determine if there exists an array of {question_case['n']} non-negative integers satisfying ALL constraints
2. Each constraint requires: a[l] AND a[l+1] AND ... AND a[r] = q
3. If exists, output "YES" followed by the array elements
4. If not exists, output "NO"

Format your final answer within [answer] tags like:
[answer]
YES
5 3 7 2
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        # 增强容错性的正则表达式
        answer_blocks = re.findall(
            r'\[ *answer *\](.*?)\[ */ *answer *\]', 
            output, 
            flags=re.IGNORECASE|re.DOTALL
        )
        if not answer_blocks:
            return None
        
        # 取最后一个答案块并标准化处理
        raw_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in raw_answer.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        status = lines[0].upper()
        result = {'status': status}
        
        if status == 'YES' and len(lines) >= 2:
            try:
                arr = list(map(int, lines[1].split()))
                if all(0 <= x < (1<<30) for x in arr):
                    result['array'] = arr
                else:
                    return None
            except:
                return None
        return result if status in ('YES', 'NO') else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 基础校验
        if not solution or 'status' not in solution:
            return False
        if solution['status'] not in ('YES', 'NO'):
            return False
        
        # 状态一致性检查
        expected_status = 'YES' if identity['solution_exists'] else 'NO'
        if solution['status'] != expected_status:
            return False
        
        # 无解案例快速返回
        if not identity['solution_exists']:
            return solution['status'] == 'NO'
        
        # 有解案例详细验证
        arr = solution.get('array', [])
        if len(arr) != identity['n']:
            return False
        if any(not isinstance(x, int) or x < 0 or x >= (1<<30) for x in arr):
            return False
        
        # 逐约束验证
        for l, r, q in identity['constraints']:
            current_and = arr[l-1]
            for i in range(l, r):
                current_and &= arr[i]
                if current_and < q:  # 提前终止优化
                    break
            if current_and != q:
                return False
        return True

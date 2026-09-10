"""# 

### 谜题描述
Johnny has just found the new, great tutorial: \"How to become a grandmaster?\". The tutorial tells many strange and unexpected for Johnny things, such as you have to be patient or that very important is solving many harder and harder problems. 

The boy has found an online judge with tasks divided by topics they cover. He has picked p^{k_i} problems from i-th category (p is his favorite number). He wants to solve them in two weeks (the patience condition is too hard for Johnny, so for simplicity, he looks only at easy tasks, which can be solved in such a period). Now our future grandmaster has to decide which topics to cover first and which the second week. Help him assign topics in such a way, that workload is balanced.

Formally, given n numbers p^{k_i}, the boy wants to divide them into two disjoint sets, minimizing the absolute difference between sums of numbers in each set. Find the minimal absolute difference. Output the result modulo 10^{9}+7.

Input

Input consists of multiple test cases. The first line contains one integer t (1 ≤ t ≤ 10^5) — the number of test cases. Each test case is described as follows:

The first line contains two integers n and p (1 ≤ n, p ≤ 10^6). The second line contains n integers k_i (0 ≤ k_i ≤ 10^6).

The sum of n over all test cases doesn't exceed 10^6.

Output

Output one integer — the reminder of division the answer by 1 000 000 007.

Example

Input


4
5 2
2 3 4 4 3
3 1
2 10 1000
4 5
0 1 1 100
1 8
89


Output


4
1
146981438
747093407

Note

You have to minimize the difference, not it's remainder. For example, if the minimum difference is equal to 2, but there is also a distribution where the difference is 10^9 + 8, then the answer is 2, not 1.

In the first test case of the example, there're the following numbers: 4, 8, 16, 16, and 8. We can divide them into such two sets: {4, 8, 16} and {8, 16}. Then the difference between the sums of numbers in sets would be 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1000005;
const long long INF = 1000000007;
template <class T>
void byebye(T _rpl) {
  cout << _rpl << endl;
  exit(0);
}
int nextint() {
  int x;
  scanf(\"%d\", &x);
  return x;
}
long long nextll() {
  long long x;
  scanf(\"%lld\", &x);
  return x;
}
template <class T>
T emax(T& t1, T t2) {
  if (t1 < t2) t1 = t2;
  return t1;
}
template <class T>
T emin(T& t1, T t2) {
  if (t1 > t2) t1 = t2;
  return t1;
}
int read() {
  int f = 1, ret = 0;
  char c = getchar();
  while (c < '0' || c > '9') {
    if (c == '-') f = -1;
    c = getchar();
  }
  while (c >= '0' && c <= '9') {
    ret = ret * 10 + (c - '0');
    c = getchar();
  }
  return ret * f;
}
void write(int x) {
  if (x < 0) {
    putchar('-');
    write(-x);
    return;
  }
  if (x < 10) {
    putchar(x + 48);
    return;
  }
  write(x / 10);
  putchar(x % 10 + 48);
}
int n;
int sum[maxn];
int val[maxn];
int a[maxn];
int query(int x) {
  if (x == 0) return sum[x];
  int ret = 0;
  for (; x; x -= x & (-x)) ret += sum[x];
  return ret;
}
void modify(int x, int val) {
  if (x == 0) {
    sum[x] += val;
    return;
  }
  for (; x <= n; x += x & (-x)) sum[x] += val;
}
long long fastmp(long long b, int x) {
  if (x == 0) return 1ll;
  long long t = fastmp(b, x / 2);
  if (x & 1)
    return 1ll * t * t % INF * b % INF;
  else
    return 1ll * t * t % INF;
}
long long fastp(long long b, int x) {
  if (x == 0) return 1ll;
  long long t = fastp(b, x >> 1);
  if (x & 1)
    return 1ll * t * t * b;
  else
    return 1ll * t * t;
}
int main() {
  int t = read();
  while (t--) {
    int n = read(), p = read();
    long long x = 1;
    int lg = 0;
    if (p != 1)
      while (x < 1e6) lg++, x *= p;
    else
      lg = 1e6;
    vector<pair<int, int> > F, S;
    vector<int> v;
    for (int i = 0; i < n; ++i) {
      a[i] = read();
      val[a[i]]++;
      v.push_back(a[i]);
    }
    sort(v.begin(), v.end());
    v.resize(unique(v.begin(), v.end()) - v.begin());
    int L = v.size();
    int lp = L - 1;
    for (int rr = L - 1; rr >= 0;) {
      int r = v[rr];
      if (val[r] <= 0) {
        --rr;
        continue;
      }
      if (val[r] % 2 == 0) {
        val[r] = 0;
        --rr;
        continue;
      }
      val[r] = 0;
      for (; lp >= 0; --lp)
        if (val[v[lp]] != 0) break;
      if (lp < 0) {
        F.push_back(pair<int, int>(r, 1));
        break;
      }
      long long need = r - v[lp];
      bool flag = true;
      if (need > lg) {
        F.push_back(pair<int, int>(r, 1));
        break;
      }
      need = fastp(p, need);
      int s = lp;
      for (; lp >= 0 && flag; --lp) {
        if (need > 1e6) {
          flag = false;
          break;
        }
        if (need <= val[v[lp]]) {
          val[v[lp]] -= need;
          break;
        }
        need -= val[v[lp]];
        if (lp != 0) {
          if (v[lp] - v[lp - 1] > lg) {
            flag = false;
          } else
            need *= fastp(p, v[lp] - v[lp - 1]);
        }
      }
      if (lp < 0 || !flag) {
        F.push_back(pair<int, int>(r, 1));
        break;
      }
      for (int j = lp + 1; j <= s; ++j) val[v[j]] = 0;
      r = lp;
    }
    for (int i = 0; i < L; ++i)
      if (val[v[i]]) S.push_back(pair<int, int>(v[i], val[v[i]]));
    long long s1 = 0, s2 = 0;
    for (pair<int, int> v1 : F) {
      s1 += fastmp(p, v1.first) * v1.second % INF;
      s1 %= INF;
    }
    for (pair<int, int> v2 : S) {
      s2 += fastmp(p, v2.first) * v2.second % INF;
      s2 %= INF;
    }
    s1 -= s2;
    if (s1 < 0) s1 += INF;
    printf(\"%d\n\", s1);
    for (int i = 0; i < n; ++i) {
      val[a[i]] = 0;
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

MOD = 10**9 + 7

def compute_min_difference(n, p, k_list):
    if p == 1:
        return (n % 2) % MOD
    
    val = defaultdict(int)
    for k in k_list:
        val[k] += 1

    v = sorted(val.keys())
    F = []
    S = []

    # 计算最大有效指数差
    lg = 0
    x = 1
    while x < 1e6 and p > 1:
        x *= p
        lg += 1

    rr = len(v) - 1
    while rr >= 0:
        current_k = v[rr]
        if val[current_k] <= 0:
            rr -= 1
            continue
        
        # 处理偶数情况
        if val[current_k] % 2 == 0:
            val[current_k] = 0
            rr -= 1
            continue
        
        # 处理奇数情况
        val[current_k] = 0
        lp = rr - 1
        while lp >= 0 and val[v[lp]] <= 0:
            lp -= 1
        
        # 没有可配对元素
        if lp < 0:
            F.append((current_k, 1))
            break
        
        # 判断指数差是否可合并
        need_steps = current_k - v[lp]
        if need_steps > lg:
            F.append((current_k, 1))
            break
        
        # 计算需要合并的数量
        need = p ** need_steps
        flag = True
        original_lp = lp
        
        # 合并操作
        while lp >= 0 and flag:
            current_lp_k = v[lp]
            
            if need > 1e6:
                flag = False
                break
            
            if val[current_lp_k] >= need:
                val[current_lp_k] -= need
                need = 0
                break
            else:
                need -= val[current_lp_k]
                val[current_lp_k] = 0
                
                if lp == 0:
                    flag = False
                    break
                
                # 计算下一级指数差
                step = current_lp_k - v[lp-1]
                if step > lg:
                    flag = False
                    break
                
                need *= p ** step
                lp -= 1
        
        if not flag or lp < 0:
            F.append((current_k, 1))
            break
        
        # 清理中间元素
        for j in range(lp + 1, original_lp + 1):
            val[v[j]] = 0
    
    # 收集剩余元素
    for k in v:
        if val[k] > 0:
            S.append((k, val[k]))
    
    # 计算最终结果
    sum_F = sum(pow(p, k, MOD) * cnt % MOD for k, cnt in F) % MOD
    sum_S = sum(pow(p, k, MOD) * cnt % MOD for k, cnt in S) % MOD
    return abs(sum_F - sum_S) % MOD

class Ejohnnyandgrandmasterbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_p=10**6, max_k=1000):
        """
        参数优化：增加有效取值范围
        - 支持生成n=1的边界情况
        - 允许k=0的特殊指数
        - 覆盖大p和小p的组合
        """
        self.max_n = max_n
        self.max_p = max_p
        self.max_k = max_k

    def case_generator(self):
        # 生成参数时增加特例概率
        p = random.choice([
            random.randint(1, 10),
            random.randint(10**5, 10**6),
            1  # 特殊case概率提升
        ])
        
        # 控制n的取值范围
        n = random.choice([
            random.randint(1, 10),
            random.randint(1, self.max_n),
            1  # 单元素case
        ])
        
        # k生成策略优化
        k_list = random.choices(
            population=[0, 1, random.randint(2, 10), random.randint(10, self.max_k)],
            weights=[0.2, 0.2, 0.3, 0.3],
            k=n
        )
        
        # 计算预期答案
        expected = compute_min_difference(n, p, k_list)
        return {
            'n': n,
            'p': p,
            'k_list': k_list,
            'expected_answer': expected
        }

    @staticmethod
    def prompt_func(question_case):
        # 增强提示信息的格式要求
        return f"""你需要将{pow(question_case['p'], question_case['k_list'][0]) if question_case['k_list'] else 0}等数值分成两个集合，使得两集合和的绝对差最小。参数：
n={question_case['n']}, p={question_case['p']}, k列表={question_case['k_list']}

请将答案数值（取模后的结果）放在[answer]标签内，例如：[answer]12345[/answer]。"""

    @staticmethod
    def extract_output(output):
        # 增强提取逻辑的鲁棒性
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 添加类型转换容错
        try:
            return int(solution) == identity['expected_answer']
        except (ValueError, TypeError):
            return False

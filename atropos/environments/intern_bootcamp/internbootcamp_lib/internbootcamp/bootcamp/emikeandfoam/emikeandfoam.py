"""# 

### 谜题描述
Mike is a bartender at Rico's bar. At Rico's, they put beer glasses in a special shelf. There are n kinds of beer at Rico's numbered from 1 to n. i-th kind of beer has ai milliliters of foam on it.

<image>

Maxim is Mike's boss. Today he told Mike to perform q queries. Initially the shelf is empty. In each request, Maxim gives him a number x. If beer number x is already in the shelf, then Mike should remove it from the shelf, otherwise he should put it in the shelf.

After each query, Mike should tell him the score of the shelf. Bears are geeks. So they think that the score of a shelf is the number of pairs (i, j) of glasses in the shelf such that i < j and <image> where <image> is the greatest common divisor of numbers a and b.

Mike is tired. So he asked you to help him in performing these requests.

Input

The first line of input contains numbers n and q (1 ≤ n, q ≤ 2 × 105), the number of different kinds of beer and number of queries.

The next line contains n space separated integers, a1, a2, ... , an (1 ≤ ai ≤ 5 × 105), the height of foam in top of each kind of beer.

The next q lines contain the queries. Each query consists of a single integer integer x (1 ≤ x ≤ n), the index of a beer that should be added or removed from the shelf.

Output

For each query, print the answer for that query in one line.

Examples

Input

5 6
1 2 3 4 6
1
2
3
4
5
1


Output

0
1
3
5
6
2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
static const int maxn = 5e5 + 6;
int sp[maxn];
vector<int> divisor[maxn];
void seive() {
  for (int i = 2; i < maxn; i++) sp[i] = i;
  for (int i = 2; i < maxn; i++) {
    for (int j = 1; 1LL * i * j < maxn; j++) {
      sp[i * j] = min(sp[i * j], sp[i]);
      divisor[i * j].push_back(i);
    }
  }
}
signed main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  seive();
  int n, q;
  cin >> n >> q;
  vector<int> arr(n + 1);
  for (int i = 1; i <= n; i++) cin >> arr[i];
  vector<int> in_self(n + 1);
  int in_self_cnt = 0;
  vector<int> divi(maxn);
  auto get = [&](vector<int> &vec) {
    int n = vec.size();
    long long coprime = 0;
    for (int mask = 0; mask < (1 << n); mask++) {
      long long d = 1;
      int bitCnt = 0;
      for (int i = 0; i < n; i++) {
        if ((mask >> i) & 1) {
          d *= vec[i];
          bitCnt++;
        }
      }
      if (bitCnt & 1)
        coprime += divi[d];
      else
        coprime -= divi[d];
    }
    return coprime;
  };
  long long ans = 0;
  while (q--) {
    int pos;
    cin >> pos;
    vector<int> prime_factors;
    int num = arr[pos];
    while (num > 1) {
      int x = sp[num];
      prime_factors.push_back(x);
      while (num % x == 0) num /= x;
    }
    sort(prime_factors.begin(), prime_factors.end());
    num = arr[pos];
    if (in_self[pos]) {
      for (int d : divisor[num]) divi[d]--;
      in_self[pos] = 0;
      in_self_cnt--;
      ans -= in_self_cnt - get(prime_factors);
    } else {
      ans += in_self_cnt - get(prime_factors);
      for (int d : divisor[num]) divi[d]++;
      in_self[pos] = 1;
      in_self_cnt++;
    }
    cout << ans << endl;
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Emikeandfoambootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_q=6, max_ai=20):
        self.max_n = max_n
        self.max_q = max_q
        self.max_ai = max_ai

    def case_generator(self):
        n = random.randint(1, self.max_n)
        q = random.randint(1, self.max_q)
        a = [random.randint(1, self.max_ai) for _ in range(n)]
        queries = [random.randint(1, n) for _ in range(q)]
        return {'n': n, 'q': q, 'a': a, 'queries': queries}

    @staticmethod
    def prompt_func(case) -> str:
        return f"""Mike需要管理啤酒货架。现有{case['n']}种啤酒（编号1-{case['n']}），泡沫量分别为：{' '.join(map(str, case['a']))}。
处理{case['q']}次查询（数字表示切换对应啤酒的状态），每次查询后计算货架中满足i<j且gcd(a_i,a_j)=1的啤酒对数。

输入格式：
1行：{case['n']} {case['q']}
2行：{' '.join(map(str, case['a']))}
后续{case['q']}行：{' '.join(map(str, case['queries']))}

将每次查询后的结果依次写在[answer]和[/answer]之间，每个结果占一行。示例：
[answer]
0
2
5
[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answers = []
        for line in matches[-1].strip().split('\n'):
            if line.strip().isdigit():
                answers.append(int(line.strip()))
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = cls._compute_expected(
            identity['n'], identity['q'], 
            identity['a'], identity['queries']
        )
        return solution == expected

    @classmethod
    def _compute_expected(cls, n, q, a, queries):
        # 预计算每个数的质因数分解
        prime_factors_list = []
        max_ai = max(a) if a else 1
        sieve = cls._build_sieve(max_ai)

        for num in a:
            factors = set()
            temp = num
            while temp > 1:
                p = sieve[temp]
                factors.add(p)
                while temp % p == 0:
                    temp //= p
            prime_factors_list.append(sorted(factors))

        # 初始化状态
        in_self = defaultdict(bool)
        divi_counts = defaultdict(int)
        current_total = 0
        answer = 0
        output = []

        for x in queries:
            idx = x-1  # queries是1-based
            num = a[idx]
            factors = prime_factors_list[idx]

            if in_self[idx]:
                # 移除操作
                sign = -1
                in_self[idx] = False
            else:
                # 添加操作
                sign = +1
                in_self[idx] = True

            # 计算当前贡献
            coprime_count = 0
            k = len(factors)
            for mask in range(1, 1 << k):
                d = 1
                bits = 0
                for i in range(k):
                    if mask & (1 << i):
                        d *= factors[i]
                        bits += 1
                cnt = divi_counts[d]
                coprime_count += cnt if bits % 2 else -cnt

            delta = sign * (current_total - coprime_count)
            answer += delta
            output.append(answer)

            # 更新除数计数
            for mask in cls._generate_divisors(num):
                divi_counts[mask] += sign

            current_total += sign

        return output

    @staticmethod
    def _build_sieve(max_num):
        sieve = list(range(max_num+1))
        for i in range(2, int(math.sqrt(max_num))+1):
            if sieve[i] == i:
                for j in range(i*i, max_num+1, i):
                    if sieve[j] == j:
                        sieve[j] = i
        return sieve

    @staticmethod
    def _generate_divisors(num):
        if num == 1:
            return []
        divisors = set()
        for i in range(2, int(math.isqrt(num)) + 1):
            if num % i == 0:
                divisors.update({i, num//i})
        divisors.add(num)
        return divisors

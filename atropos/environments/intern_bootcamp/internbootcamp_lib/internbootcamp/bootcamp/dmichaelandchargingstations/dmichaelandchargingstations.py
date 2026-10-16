"""# 

### 谜题描述
Michael has just bought a new electric car for moving across city. Michael does not like to overwork, so each day he drives to only one of two his jobs.

Michael's day starts from charging his electric car for getting to the work and back. He spends 1000 burles on charge if he goes to the first job, and 2000 burles if he goes to the second job.

On a charging station he uses there is a loyalty program that involves bonus cards. Bonus card may have some non-negative amount of bonus burles. Each time customer is going to buy something for the price of x burles, he is allowed to pay an amount of y (0 ≤ y ≤ x) burles that does not exceed the bonus card balance with bonus burles. In this case he pays x - y burles with cash, and the balance on the bonus card is decreased by y bonus burles. 

If customer pays whole price with cash (i.e., y = 0) then 10% of price is returned back to the bonus card. This means that bonus card balance increases by <image> bonus burles. Initially the bonus card balance is equal to 0 bonus burles.

Michael has planned next n days and he knows how much does the charge cost on each of those days. Help Michael determine the minimum amount of burles in cash he has to spend with optimal use of bonus card. Assume that Michael is able to cover any part of the price with cash in any day. It is not necessary to spend all bonus burles at the end of the given period.

Input

The first line of input contains a single integer n (1 ≤ n ≤ 300 000), the number of days Michael has planned.

Next line contains n integers a1, a2, ..., an (ai = 1000 or ai = 2000) with ai denoting the charging cost at the day i.

Output

Output the minimum amount of burles Michael has to spend.

Examples

Input

3
1000 2000 1000


Output

3700


Input

6
2000 2000 2000 2000 2000 1000


Output

10000

Note

In the first sample case the most optimal way for Michael is to pay for the first two days spending 3000 burles and get 300 bonus burles as return. After that he is able to pay only 700 burles for the third days, covering the rest of the price with bonus burles.

In the second sample case the most optimal way for Michael is to pay the whole price for the first five days, getting 1000 bonus burles as return and being able to use them on the last day without paying anything in cash.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MOD = (int)1e9 + 7;
const int MOD2 = 1007681537;
const int INF = (int)1e9;
const long long LINF = (long long)1e18;
const long double PI = acos((long double)-1);
const long double EPS = 1e-9;
inline long long gcd(long long a, long long b) {
  long long r;
  while (b) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}
inline long long lcm(long long a, long long b) { return a / gcd(a, b) * b; }
inline long long fpow(long long n, long long k, int p = MOD) {
  long long r = 1;
  for (; k; k >>= 1) {
    if (k & 1) r = r * n % p;
    n = n * n % p;
  }
  return r;
}
template <class T>
inline int chkmin(T& a, const T& val) {
  return val < a ? a = val, 1 : 0;
}
template <class T>
inline int chkmax(T& a, const T& val) {
  return a < val ? a = val, 1 : 0;
}
inline long long isqrt(long long k) {
  long long r = sqrt(k) + 1;
  while (r * r > k) r--;
  return r;
}
inline long long icbrt(long long k) {
  long long r = cbrt(k) + 1;
  while (r * r * r > k) r--;
  return r;
}
inline void addmod(int& a, int val, int p = MOD) {
  if ((a = (a + val)) >= p) a -= p;
}
inline void submod(int& a, int val, int p = MOD) {
  if ((a = (a - val)) < 0) a += p;
}
inline int mult(int a, int b, int p = MOD) { return (long long)a * b % p; }
inline int inv(int a, int p = MOD) { return fpow(a, p - 2, p); }
inline int sign(long double x) { return x < -EPS ? -1 : x > +EPS; }
inline int sign(long double x, long double y) { return sign(x - y); }
const int maxn = 1e6 + 5;
int n;
int a[maxn];
vector<int> pos1;
vector<int> pos2;
int mn[maxn << 1];
int sm[maxn << 1];
void upd(int p, int val) {
  p += 1 << 19;
  mn[p] = sm[p] = val;
  while (p > 1) {
    p >>= 1;
    sm[p] = sm[p << 1] + sm[p << 1 | 1];
    mn[p] = min(mn[p << 1], sm[p << 1] + mn[p << 1 | 1]);
  }
}
int check(int mi) {
  for (int i = (0); i < (n); i++) {
    upd(i, -a[i]);
  }
  for (int i = (0); i < (int((pos2).size())); i++) {
    int ix = pos2[i];
    upd(ix, 200);
  }
  int ptr = int((pos2).size()) - 1;
  for (int x = (0); x < (int((pos1).size()) + 1); x++)
    if (mi >= 1000 * x) {
      int y = min(int((pos2).size()), (mi - 1000 * x) / 2000);
      while (ptr >= y) {
        int ix = pos2[ptr];
        upd(ix, -2000);
        ptr--;
      }
      if (x) {
        int ix = pos1[x - 1];
        upd(ix, 100);
      }
      int rm = mi - 1000 * x - 2000 * y;
      if ((!x && a[0] == 1000) || (!y && a[0] == 2000)) {
        upd(0, rm - a[0]);
      } else {
        upd(0, rm + a[0] / 10);
      }
      if (mn[1] >= 0) {
        return 1;
      }
    }
  return 0;
}
void solve() {
  cin >> n;
  for (int i = (0); i < (n); i++) {
    cin >> a[i];
    if (a[i] == 1000) {
      pos1.push_back(i);
    } else {
      pos2.push_back(i);
    }
  }
  int lo = 0, hi = 600000000;
  while (lo < hi) {
    int mi = lo + hi >> 1;
    if (!check(mi)) {
      lo = mi + 1;
    } else {
      hi = mi;
    }
  }
  cout << (lo + hi >> 1) << \"\n\";
}
int main() {
  int JUDGE_ONLINE = 1;
  if (fopen(\"in.txt\", \"r\")) {
    JUDGE_ONLINE = 0;
    assert(freopen(\"in.txt\", \"r\", stdin));
  } else {
    ios_base::sync_with_stdio(0), cin.tie(0);
  }
  solve();
  if (!JUDGE_ONLINE) {
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Dmichaelandchargingstationsbootcamp(Basebootcamp):
    def __init__(self, max_days=10, seed=None):
        super().__init__()
        self.max_days = min(max_days, 300000)  # Ensure constraint compliance
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def case_generator(self):
        random.seed(self.seed)
        n = random.randint(1, self.max_days)
        a = [random.choice([1000, 2000]) for _ in range(n)]
        return {
            'n': n,
            'a': a,
            'correct_answer': self.calculate_min_burles(a)
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        a_str = ' '.join(map(str, question_case['a']))
        return f"""Michael需要规划未来{question_case['n']}天的充电费用，每天费用为1000或2000布勒。请帮他计算使用奖金卡后的最小现金支出。

规则说明：
1. 奖金卡初始余额为0
2. 每天可选择两种支付方式：
   a) 全额现金支付：支付全额费用并获得10%的奖金（向下取整）
   b) 组合支付：使用y布勒奖金（0 ≤ y ≤ 费用与余额最小值）和现金支付剩余部分
3. 奖金累计规则：
   - 只在使用全额现金支付时获得奖励
   - 奖励金额为支付金额的10%（向下取整）

输入格式：
{question_case['n']}
{a_str}

请将最终答案放入[answer]标签内，例如：[answer]3700[/answer]。"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

    @staticmethod
    def calculate_min_burles(a):
        pos2 = [i for i, price in enumerate(a) if price == 2000]
        low, high = 0, sum(a)
        answer = high
        
        def is_feasible(mid):
            balance = 0
            cash_used = 0
            ptr = len(pos2) - 1
            
            # Process days in reverse order
            for i in reversed(range(len(a))):
                if a[i] == 2000 and ptr >= 0 and i == pos2[ptr]:
                    if cash_used + 2000 <= mid:
                        cash_used += 2000
                        balance += 200
                        ptr -= 1
                    else:
                        use = min(balance, 2000)
                        cash_used += (2000 - use)
                        balance -= use
                        if cash_used > mid:
                            return False
                else:
                    price = a[i]
                    use = min(balance, price)
                    cash_used += (price - use)
                    balance -= use
                    if cash_used > mid:
                        return False
                    if use == 0:
                        balance += price // 10
            
            # Process remaining 2000 days
            while ptr >= 0:
                use = min(balance, 2000)
                cash_used += (2000 - use)
                balance -= use
                if cash_used > mid:
                    return False
                if use == 0:
                    balance += 200
                ptr -= 1
            
            return cash_used <= mid
        
        # Binary search
        while low <= high:
            mid = (low + high) // 2
            if is_feasible(mid):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        return answer

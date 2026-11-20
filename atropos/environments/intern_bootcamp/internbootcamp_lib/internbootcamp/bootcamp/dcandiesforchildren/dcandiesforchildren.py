"""# 

### 谜题描述
At the children's festival, children were dancing in a circle. When music stopped playing, the children were still standing in a circle. Then Lena remembered, that her parents gave her a candy box with exactly k candies \"Wilky May\". Lena is not a greedy person, so she decided to present all her candies to her friends in the circle. Lena knows, that some of her friends have a sweet tooth and others do not. Sweet tooth takes out of the box two candies, if the box has at least two candies, and otherwise takes one. The rest of Lena's friends always take exactly one candy from the box.

Before starting to give candies, Lena step out of the circle, after that there were exactly n people remaining there. Lena numbered her friends in a clockwise order with positive integers starting with 1 in such a way that index 1 was assigned to her best friend Roma.

Initially, Lena gave the box to the friend with number l, after that each friend (starting from friend number l) took candies from the box and passed the box to the next friend in clockwise order. The process ended with the friend number r taking the last candy (or two, who knows) and the empty box. Please note that it is possible that some of Lena's friends took candy from the box several times, that is, the box could have gone several full circles before becoming empty.

Lena does not know which of her friends have a sweet tooth, but she is interested in the maximum possible number of friends that can have a sweet tooth. If the situation could not happen, and Lena have been proved wrong in her observations, please tell her about this.

Input

The only line contains four integers n, l, r and k (1 ≤ n, k ≤ 10^{11}, 1 ≤ l, r ≤ n) — the number of children in the circle, the number of friend, who was given a box with candies, the number of friend, who has taken last candy and the initial number of candies in the box respectively.

Output

Print exactly one integer — the maximum possible number of sweet tooth among the friends of Lena or \"-1\" (quotes for clarity), if Lena is wrong.

Examples

Input

4 1 4 12


Output

2


Input

5 3 4 10


Output

3


Input

10 5 5 1


Output

10


Input

5 4 5 6


Output

-1

Note

In the first example, any two friends can be sweet tooths, this way each person will receive the box with candies twice and the last person to take sweets will be the fourth friend.

In the second example, sweet tooths can be any three friends, except for the friend on the third position.

In the third example, only one friend will take candy, but he can still be a sweet tooth, but just not being able to take two candies. All other friends in the circle can be sweet tooths as well, they just will not be able to take a candy even once.

In the fourth example, Lena is wrong and this situation couldn't happen.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
#pragma GCC optimize(\"O3\")
using namespace std;
const long long N = 2e3 + 10;
const long long inf = 1e18 + 10ll;
const long long mod = 1e9 + 7;
const long long eps = 1e-9;
mt19937 rnd(chrono::high_resolution_clock::now().time_since_epoch().count());
using namespace std;
long long n, l, r, k;
long long len1;
long long len2;
long long gcd(long long a, long long b, long long &first, long long &second) {
  if (b == 0) {
    first = 1;
    second = 0;
    return a;
  }
  long long g = gcd(b, a % b, first, second);
  long long nx = second;
  long long ny = first - second * (a / b);
  first = nx;
  second = ny;
  return g;
}
bool findsol(long long a, long long b, long long c, long long &x0,
             long long &y0, long long &g) {
  g = gcd(a, b, x0, y0);
  if (c % g != 0) return false;
  x0 *= c / g;
  y0 *= c / g;
  return true;
}
void shift(long long &first, long long &second, long long a, long long b,
           long long cnt) {
  first += b * cnt;
  second -= a * cnt;
}
bool findmxsol(long long a, long long b, long long c, long long &first,
               long long &second) {
  long long g = -1;
  if (!findsol(a, b, c, first, second, g)) return false;
  a /= g;
  b /= g;
  if (first < len1) shift(first, second, a, b, (len1 - first) / b);
  if (first > len1) shift(first, second, a, b, -((first - len1 + b - 1) / b));
  pair<long long, long long> p1 = {first, second};
  if (first < 0 || first > len1 || second < 0 || second > len2) p1 = {-1, -1};
  if (first < 0) shift(first, second, a, b, (0 - first + b - 1) / b);
  if (first > 0) shift(first, second, a, b, -(first - 0) / b);
  pair<long long, long long> p2 = {first, second};
  if (first < 0 || first > len1 || second < 0 || second > len2) p2 = {-1, -1};
  if (second < len2) shift(first, second, a, b, -(len2 - second) / a);
  if (second > len2) shift(first, second, a, b, (second - len2 + a - 1) / a);
  pair<long long, long long> p3 = {first, second};
  if (first < 0 || first > len1 || second < 0 || second > len2) p3 = {-1, -1};
  if (second < 0) shift(first, second, a, b, -(0 - second + a - 1) / a);
  if (second > 0) shift(first, second, a, b, (second - 0) / a);
  pair<long long, long long> p4 = {first, second};
  if (first < 0 || first > len1 || second < 0 || second > len2) p4 = {-1, -1};
  pair<long long, long long> mxp = {-1, -1};
  if (p1.first <= len1 && p1.second <= len2 && p1.first >= 0 &&
      p1.second >= 0 && (mxp.first + mxp.second < p1.first + p1.second))
    mxp = p1;
  if (p2.first <= len1 && p2.second <= len2 && p2.first >= 0 &&
      p2.second >= 0 && (mxp.first + mxp.second < p2.first + p2.second))
    mxp = p2;
  if (p3.first <= len1 && p3.second <= len2 && p3.first >= 0 &&
      p3.second >= 0 && (mxp.first + mxp.second < p3.first + p3.second))
    mxp = p3;
  if (p4.first <= len1 && p4.second <= len2 && p4.first >= 0 &&
      p4.second >= 0 && (mxp.first + mxp.second < p4.first + p4.second))
    mxp = p4;
  first = mxp.first;
  second = mxp.second;
  return true;
}
pair<long long, long long> calc(long long t, bool is) {
  long long second = k - n * t - len1 + is;
  pair<long long, long long> p = {-1, -1};
  findmxsol((t + 1), t, second, p.first, p.second);
  return p;
}
long long mul(long long a, long long b) {
  if (a <= 1e9 && b <= 1e9) return a * b;
  return inf;
}
int32_t main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  cout.precision(10);
  cin >> n >> l >> r >> k;
  if (r < l)
    len1 = (n - l) + r + 1;
  else
    len1 = r - l + 1;
  len2 = n - len1;
  long long ans = -1;
  if (n <= 7e7) {
    for (long long i = 0; i <= n; ++i) {
      long long t = (k - len1) / (i + n);
      long long second = k - len1 - t * (i + n);
      if (second >= max(0ll, (i - len2)) && second <= len1) ans = max(ans, i);
      t = (k + 1 - len1) / (i + n);
      second = k + 1 - len1 - t * (i + n);
      if (second >= max(0ll, (i - len2)) && second <= len1 && second)
        ans = max(ans, i);
    }
  } else {
    long long kneed = (k + 1) / 2;
    if (kneed <= len1 && kneed + k / 2 >= len1)
      ans = max(ans, k / 2 - (len1 - kneed) + len2);
    ++k;
    kneed = (k + 1) / 2;
    if (kneed <= len1 && kneed + k / 2 >= len1)
      ans = max(ans, k / 2 - (len1 - kneed) + len2);
    --k;
    for (long long i = 1; i <= k / n; ++i) {
      long long y1 = k - n * i - len1;
      long long y2 = k - n * i - len1 + 1;
      pair<long long, long long> p1 = calc(i, false);
      pair<long long, long long> p2 = calc(i, true);
      if (p1.first <= len1 && p1.second <= len2 &&
          ((i + 1) * p1.first + i * p1.second == y1) && y1 >= 0 &&
          p1.first >= 0 && p1.second >= 0)
        ans = max(ans, p1.first + p1.second);
      if (p2.first <= len1 && p2.second <= len2 &&
          ((i + 1) * p2.first + i * p2.second == y2) && y2 >= 0 &&
          p2.first >= 1 && p2.second >= 0)
        ans = max(ans, p2.first + p2.second);
    }
  }
  cout << ans << \"\n\";
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import math
from bootcamp import Basebootcamp

class Dcandiesforchildrenbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10**11, k_min=1, k_max=10**11):
        self.n_min = n_min
        self.n_max = n_max
        self.k_min = k_min
        self.k_max = k_max
    
    def case_generator(self):
        def rand_bigint():
            return random.randint(1, 10**5) if random.random() < 0.5 else random.randint(10**10, 10**11)
        
        n = rand_bigint()
        l = random.randint(1, n)
        r = random.randint(1, n)
        k = rand_bigint()
        return {'n': n, 'l': l, 'r': r, 'k': k}
    
    @staticmethod
    def prompt_func(question_case):
        params = question_case
        return f"""## 谜题描述

音乐停止后，{params['n']}个孩子围成圆圈。Lena将装有{params['k']}颗糖果的盒子交给朋友{params['l']}，最后由朋友{params['r']}拿到最后糖果。

**规则：**
1. 甜牙齿朋友：当盒中≥2糖时取2颗，否则取1颗
2. 其他朋友：每次固定取1颗
3. 盒子按顺时针传递，可能循环多圈

**任务：**
求可能的最大甜牙齿朋友数（若情况不可能返回-1）

**答案格式：**
将最终答案放在[answer]标签内，如：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == cls._correct_solve(identity)
        except:
            return False

    @staticmethod
    def _correct_solve(params):
        n = params['n']
        l = params['l']
        r = params['r']
        k = params['k']

        # 修正关键计算错误：r < l时的len1计算
        def compute_len1():
            if r < l:
                return (n - l) + r + 1  # 修正此处，增加+1
            else:
                return r - l + 1
        
        len1 = compute_len1()
        len2 = n - len1

        if len1 == 0 or k < len1:
            return -1

        if n <= 10**6:  # 小规模优化
            for sweet_count in range(n, -1, -1):
                # 方程：t*(sweet + n) + rem = k - len1
                t = (k - len1) // (sweet_count + n)
                rem = k - len1 - t*(sweet_count + n)
                
                if rem >= max(0, sweet_count - len2) and rem <= len1:
                    return sweet_count
                
                # 处理最后一轮多拿一个的情况
                t = (k + 1 - len1) // (sweet_count + n)
                rem = k + 1 - len1 - t*(sweet_count + n)
                if rem > 0 and rem >= max(0, sweet_count - len2) and rem <= len1:
                    return sweet_count
            return -1
        else:  # 数学推导法
            def extended_gcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x, y = extended_gcd(b, a % b)
                return (g, y, x - (a//b)*y)

            max_sweet = -1
            for delta in [0, 1]:  # 处理k和k+1两种情况
                C = k + delta - len1
                if C < 0:
                    continue
                
                a, b = 1, 1
                g, x, y = extended_gcd(a, b)
                if C % g != 0:
                    continue
                
                x0 = x * (C//g)
                y0 = y * (C//g)
                
                # 通解形式：x = x0 + k*b/g，y = y0 -k*a/g
                min_k = max(math.ceil((-y0)/a), math.ceil((x0 - len1)/b))
                max_k = min(math.floor((n - y0)/a), math.floor(x0/b))
                
                if min_k > max_k:
                    continue
                
                # 取最大的x + y
                best_x = x0 - b*min_k
                best_y = y0 + a*min_k
                candidate = best_x + best_y
                max_sweet = max(max_sweet, candidate)
            
            return max_sweet if 0 <= max_sweet <= n else -1

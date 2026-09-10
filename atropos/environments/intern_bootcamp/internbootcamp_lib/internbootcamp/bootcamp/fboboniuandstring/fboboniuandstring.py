"""# 

### 谜题描述
Boboniu defines BN-string as a string s of characters 'B' and 'N'.

You can perform the following operations on the BN-string s:

  * Remove a character of s. 
  * Remove a substring \"BN\" or \"NB\" of s. 
  * Add a character 'B' or 'N' to the end of s. 
  * Add a string \"BN\" or \"NB\" to the end of s. 



Note that a string a is a substring of a string b if a can be obtained from b by deletion of several (possibly, zero or all) characters from the beginning and several (possibly, zero or all) characters from the end.

Boboniu thinks that BN-strings s and t are similar if and only if:

  * |s|=|t|. 
  * There exists a permutation p_1, p_2, …, p_{|s|} such that for all i (1≤ i≤ |s|), s_{p_i}=t_i. 



Boboniu also defines dist(s,t), the distance between s and t, as the minimum number of operations that makes s similar to t.

Now Boboniu gives you n non-empty BN-strings s_1,s_2,…, s_n and asks you to find a non-empty BN-string t such that the maximum distance to string s is minimized, i.e. you need to minimize max_{i=1}^n dist(s_i,t).

Input

The first line contains a single integer n (1≤ n≤ 3⋅ 10^5).

Each of the next n lines contains a string s_i (1≤ |s_i| ≤ 5⋅ 10^5). It is guaranteed that s_i only contains 'B' and 'N'. The sum of |s_i| does not exceed 5⋅ 10^5.

Output

In the first line, print the minimum max_{i=1}^n dist(s_i,t).

In the second line, print the suitable t.

If there are several possible t's, you can print any.

Examples

Input


3
B
N
BN


Output


1
BN


Input


10
N
BBBBBB
BNNNBBNBB
NNNNBNBNNBNNNBBN
NBNBN
NNNNNN
BNBNBNBBBBNNNNBBBBNNBBNBNBBNBBBBBBBB
NNNNBN
NBBBBBBBB
NNNNNN


Output


12
BBBBBBBBBBBBNNNNNNNNNNNN


Input


8
NNN
NNN
BBNNBBBN
NNNBNN
B
NNN
NNNNBNN
NNNNNNNNNNNNNNNBNNNNNNNBNB


Output


12
BBBBNNNNNNNNNNNN


Input


3
BNNNBNNNNBNBBNNNBBNNNNBBBBNNBBBBBBNBBBBBNBBBNNBBBNBNBBBN
BBBNBBBBNNNNNBBNBBBNNNBB
BBBBBBBBBBBBBBNBBBBBNBBBBBNBBBBNB


Output


12
BBBBBBBBBBBBBBBBBBBBBBBBBBNNNNNNNNNNNN

Note

In the first example dist(B,BN)=dist(N,BN)=1, dist(BN,BN)=0. So the maximum distance is 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
#pragma GCC target(\"avx2\")
#pragma GCC optimization(\"O3\")
#pragma GCC optimization(\"unroll-loops\")
const long long INFll = 1e18;
const int INFint = 1e9;
const long long MOD = 1e9 + 9;
void needForSpeed() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
}
bool isPrime(long long a) {
  if (a < 2) return 0;
  for (long long i = 2; i <= sqrt(a); i++)
    if (a % i == 0) return 0;
  return 1;
}
long long binpow(long long base, long long pow, long long mod = INFll) {
  if (pow == 0)
    return 1;
  else if (pow % 2)
    return base * binpow(base, pow - 1, mod) % mod;
  else {
    long long p = binpow(base, pow / 2, mod);
    return p * p % mod;
  }
}
template <class T>
void PR_VEC(const vector<T> &vec) {
  cout << \"size(\" << vec.size() << \"):[ '\";
  cout << vec[0] << \"'\";
  for (int i = 1; i < vec.size(); ++i) cout << \" , '\" << vec[i] << \"'\";
  cout << \" ]\n\";
}
int solve() {
  int n;
  cin >> n;
  vector<pair<int, int>> dots(n);
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    int x = 0, y = 0;
    for (int j = 0; j < s.length(); j++) {
      if (s[j] == 'B')
        ++x;
      else
        ++y;
    }
    dots[i] = {x, y};
  }
  int l = -1, r = 10000000;
  pair<int, int> final_dot = {0, 0};
  while (r - l > 1) {
    int mid = (l + r) / 2;
    int minx = 0, miny = 0, maxx = 10000000, maxy = 10000000;
    for (int i = 0; i < n; i++) {
      minx = max(minx, dots[i].first - mid);
      maxx = min(maxx, dots[i].first + mid);
      miny = max(miny, dots[i].second - mid);
      maxy = min(maxy, dots[i].second + mid);
    }
    int minXY = -10000000, maxXY = 10000000;
    for (int i = 0; i < n; i++) {
      minXY = max(minXY, dots[i].first - dots[i].second - mid);
      maxXY = min(maxXY, dots[i].first - dots[i].second + mid);
    }
    bool may_be = (minx <= maxx && miny <= maxy && minXY <= maxXY);
    if (minx - maxy > maxXY || maxx - miny < minXY) may_be = false;
    if (may_be) {
      final_dot = {minx, maxy};
      if (final_dot.first - final_dot.second < minXY) {
        int move = min(maxx - final_dot.first,
                       minXY - (final_dot.first - final_dot.second));
        final_dot.first += move;
      }
      if (final_dot.first - final_dot.second < minXY) {
        int move = min(final_dot.second - miny,
                       minXY - (final_dot.first - final_dot.second));
        final_dot.second -= move;
      }
      r = mid;
    } else
      l = mid;
  }
  cout << r << \"\n\";
  for (int i = 0; i < final_dot.first; i++) cout << 'B';
  for (int i = 0; i < final_dot.second; i++) cout << 'N';
  cout << \"\n\";
  return 0;
}
int main() {
  needForSpeed();
  int tests = 1;
  while (tests--) {
    solve();
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def solve_bn_case(dots):
    n = len(dots)
    l = -1
    r = 10**7
    final_dot = (0, 0)
    while r - l > 1:
        mid = (l + r) // 2
        minx = -10**7
        maxx = 10**7
        miny = -10**7
        maxy = 10**7
        minXY = -10**7
        maxXY = 10**7
        
        for x, y in dots:
            minx = max(minx, x - mid)
            maxx = min(maxx, x + mid)
            miny = max(miny, y - mid)
            maxy = min(maxy, y + mid)
            minXY = max(minXY, (x - y) - mid)
            maxXY = min(maxXY, (x - y) + mid)
        
        may_be = (minx <= maxx) and (miny <= maxy) and (minXY <= maxXY)
        if may_be:
            lower_bound = minx - maxy
            upper_bound = maxx - miny
            if lower_bound > maxXY or upper_bound < minXY:
                may_be = False
        
        if may_be:
            x_t = minx
            y_t = maxy
            if (x_t - y_t) < minXY:
                move = min(maxx - x_t, minXY - (x_t - y_t))
                x_t += move
                if (x_t - y_t) < minXY:
                    move = min(y_t - miny, minXY - (x_t - y_t))
                    y_t -= move
            x_t = max(x_t, 0)
            y_t = max(y_t, 0)
            if x_t == 0 and y_t == 0:
                x_t = 1
                y_t = 0
            final_dot = (x_t, y_t)
            r = mid
        else:
            l = mid
    return r, final_dot

class Fboboniuandstringbootcamp(Basebootcamp):
    def __init__(self, n=3, max_b=5, max_n=5):
        self.n = n
        self.max_b = max_b
        self.max_n = max_n
    
    def case_generator(self):
        dots = []
        for _ in range(self.n):
            x = random.randint(0, self.max_b)
            y = random.randint(0, self.max_n)
            while x + y == 0:
                x = random.randint(0, self.max_b)
                y = random.randint(0, self.max_n)
            dots.append((x, y))
        correct_d_max, (t_x, t_y) = solve_bn_case(dots)
        strings = []
        for x, y in dots:
            strings.append('B' * x + 'N' * y)
        return {
            'n': self.n,
            'strings': strings,
            'dots': dots,
            'correct_d_max': correct_d_max
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])] + question_case['strings']
        input_part = '\n'.join(input_lines)
        prompt = f"""You need to solve a BN-string optimization problem. Find a non-empty BN-string t such that the maximum distance to given strings is minimized. 

**Problem Details:**
- Distance 'dist(s, t)' is the minimum operations to make s similar to t, where similarity requires equal length and possible permutation of characters.
- Operations include adding/removing characters or substrings "BN"/"NB".

**Input:**
{input_part}

**Output Format:**
First line: the minimal maximum distance.
Second line: the optimal BN-string t.

Put your answer within [answer] and [/answer] tags. Example:
[answer]
3
BNNBB
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        d_str, t = lines[0], lines[1]
        if not t or any(c not in {'B', 'N'} for c in t):
            return None
        try:
            d = int(d_str)
        except:
            return None
        return {'d': d, 't': t}
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        try:
            user_d = solution['d']
            user_t = solution['t']
        except:
            return False
        if not user_t or any(c not in {'B', 'N'} for c in user_t):
            return False
        x_t = user_t.count('B')
        y_t = user_t.count('N')
        if x_t + y_t == 0:
            return False
        correct_d = identity['correct_d_max']
        max_dist = 0
        for x_i, y_i in identity['dots']:
            dx = abs(x_t - x_i)
            dy = abs(y_t - y_i)
            d_xy = abs((x_t - y_t) - (x_i - y_i))
            current = max(dx, dy, d_xy)
            if current > max_dist:
                max_dist = current
        return max_dist == correct_d and user_d == correct_d

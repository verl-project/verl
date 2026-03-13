"""# 

### 谜题描述
Little Timofey likes integers a lot. Unfortunately, he is very young and can't work with very big integers, so he does all the operations modulo his favorite prime m. Also, Timofey likes to look for arithmetical progressions everywhere.

One of his birthday presents was a sequence of distinct integers a1, a2, ..., an. Timofey wants to know whether he can rearrange the elements of the sequence so that is will be an arithmetical progression modulo m, or not.

Arithmetical progression modulo m of length n with first element x and difference d is sequence of integers x, x + d, x + 2d, ..., x + (n - 1)·d, each taken modulo m.

Input

The first line contains two integers m and n (2 ≤ m ≤ 109 + 7, 1 ≤ n ≤ 105, m is prime) — Timofey's favorite prime module and the length of the sequence.

The second line contains n distinct integers a1, a2, ..., an (0 ≤ ai < m) — the elements of the sequence.

Output

Print -1 if it is not possible to rearrange the elements of the sequence so that is will be an arithmetical progression modulo m.

Otherwise, print two integers — the first element of the obtained progression x (0 ≤ x < m) and its difference d (0 ≤ d < m).

If there are multiple answers, print any of them.

Examples

Input

17 5
0 2 4 13 15


Output

13 2


Input

17 5
0 2 4 13 14


Output

-1


Input

5 3
1 2 3


Output

3 4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 5;
long long p, n;
bool ind;
long long a[N];
set<long long> num;
long long add(long long a, long long b) { return (a + b) % p; }
long long sub(long long a, long long b) { return (a - b + p) % p; }
long long mult(long long a, long long b) { return a * b % p; }
long long power(long long a, int m) {
  if (m == 0) {
    return 1;
  }
  long long res = power(a, m / 2);
  res = res * res % p;
  if (m % 2 == 1) res = res * a % p;
  return res;
}
long long rev(long long a) { return power(a, p - 2); }
void print_ans(pair<long long, long long> res) {
  if (ind) {
    n = p - n;
    res.first = sub(res.first, mult(n, res.second));
  }
  cout << res.first << \" \" << res.second << endl;
}
bool check(int x, int pos_x, int y, int pos_y,
           pair<long long, long long>& res) {
  if (pos_x == pos_y) return false;
  res.second = mult(sub(x, y), rev(sub(pos_x, pos_y)));
  res.first = sub(x, mult(res.second, sub(pos_x, 1)));
  for (int i = 0; i < n; i++) {
    if (num.find(add(res.first, mult(i, res.second))) == num.end())
      return false;
  }
  return true;
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(nullptr);
  cin >> p >> n;
  for (int i = 0; i < n; i++) {
    long long t;
    cin >> t;
    num.insert(t);
    a[i] = t;
  }
  if (n == 1 || n == p) {
    cout << a[0] << \" 1\n\";
    return 0;
  }
  if (n == p - 1) {
    for (int i = 0; i < p; i++) {
      if (num.find(i) == num.end()) {
        cout << (i + 1) % p << \" 1\n\";
        return 0;
      }
    }
  }
  if (n > p / 2) {
    int cur = 0;
    for (int i = 0; i < p; i++) {
      if (num.find(i) == num.end()) {
        a[cur] = i;
        cur++;
      }
    }
    num.clear();
    for (int i = 0; i < cur; i++) {
      num.insert(a[i]);
    }
    ind = true;
    n = p - n;
  }
  long long x = a[0];
  long long pos_x = 0;
  for (int i = 0; i < n; i++) {
    if (num.find((2 * x - a[i] + p) % p) != num.end()) pos_x++;
  }
  pos_x = (pos_x + 1) / 2;
  long long y = a[1];
  long long pos_y = 0;
  for (int i = 0; i < n; i++) {
    if (num.find((2 * y - a[i] + p) % p) != num.end()) pos_y++;
  }
  pos_y = (pos_y + 1) / 2;
  pair<long long, long long> res;
  if (check(x, pos_x, y, pos_y, res)) {
    print_ans(res);
    return 0;
  }
  if (check(x, sub(add(n, 1), pos_x), y, pos_y, res)) {
    print_ans(res);
    return 0;
  }
  if (check(x, pos_x, y, sub(add(n, 1), pos_y), res)) {
    print_ans(res);
    return 0;
  }
  if (check(x, sub(add(n, 1), pos_x), y, sub(add(n, 1), pos_y), res)) {
    print_ans(res);
    return 0;
  }
  cout << \"-1\n\";
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
import re
from bootcamp import Basebootcamp

class Ctimofeyandremodulingbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.m = params.get('m', 17)
        self.n = params.get('n', 5)

    def case_generator(self):
        m = self.m
        n = self.n
        if m < 2:
            m = 2
        if n < 1 or n > 10**5:
            n = min(max(n, 1), 10**5)
        x = random.randint(0, m - 1)
        if n == 1:
            d = random.randint(0, m - 1)
        else:
            d = random.randint(1, m - 1)
        a = [(x + i * d) % m for i in range(n)]
        random.shuffle(a)
        return {
            "m": m,
            "n": n,
            "a": a
        }

    @staticmethod
    def prompt_func(question_case):
        m = question_case['m']
        n = question_case['n']
        a = question_case['a']
        a_str = ', '.join(map(str, a))
        prompt = f"Timofey有一个序列：{a_str}，模数m={m}。他想知道是否可以将这个序列重新排列成一个模{m}的算术级数。如果可以，请给出首项x和公差d；否则输出-1。答案格式：[answer]x d[/answer]。"
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if last_match == '-1':
            return -1
        else:
            parts = last_match.split()
            if len(parts) != 2:
                return None
            try:
                x = int(parts[0])
                d = int(parts[1])
                return (x, d)
            except ValueError:
                return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        m = identity['m']
        n = identity['n']
        a = identity['a']
        if solution == -1:
            return False
        else:
            x, d = solution
            if n == 1:
                return x == a[0]
            else:
                correct = [(x + i * d) % m for i in range(n)]
                correct_sorted = sorted(correct)
                a_sorted = sorted(a)
                return correct_sorted == a_sorted

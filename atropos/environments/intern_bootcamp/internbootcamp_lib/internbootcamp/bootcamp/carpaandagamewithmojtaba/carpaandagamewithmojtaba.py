"""# 

### 谜题描述
Mojtaba and Arpa are playing a game. They have a list of n numbers in the game.

In a player's turn, he chooses a number pk (where p is a prime number and k is a positive integer) such that pk divides at least one number in the list. For each number in the list divisible by pk, call it x, the player will delete x and add <image> to the list. The player who can not make a valid choice of p and k loses.

Mojtaba starts the game and the players alternatively make moves. Determine which one of players will be the winner if both players play optimally.

Input

The first line contains a single integer n (1 ≤ n ≤ 100) — the number of elements in the list.

The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 109) — the elements of the list.

Output

If Mojtaba wins, print \"Mojtaba\", otherwise print \"Arpa\" (without quotes).

You can print each letter in any case (upper or lower).

Examples

Input

4
1 1 1 1


Output

Arpa


Input

4
1 1 17 17


Output

Mojtaba


Input

4
1 1 17 289


Output

Arpa


Input

5
1 2 3 4 5


Output

Arpa

Note

In the first sample test, Mojtaba can't move.

In the second sample test, Mojtaba chooses p = 17 and k = 1, then the list changes to [1, 1, 1, 1].

In the third sample test, if Mojtaba chooses p = 17 and k = 1, then Arpa chooses p = 17 and k = 1 and wins, if Mojtaba chooses p = 17 and k = 2, then Arpa chooses p = 17 and k = 1 and wins.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long oo = 1000000000000000000;
const int N = 1000006;
long long v[N];
map<int, int> dp;
int mex(const set<int> &s) {
  int ans = 0;
  while (s.count(ans)) ans++;
  return ans;
}
int g(int bit) {
  if (dp.count(bit)) return dp[bit];
  int lg = 31 - __builtin_clz(bit);
  set<int> s;
  for (int i = 0; i <= lg; i++) {
    int q = 0;
    for (int j = 0; j <= lg; j++)
      if (bit & (1 << j)) {
        if (j < i) q |= 1 << j;
        if (j > i) q |= 1 << (j - i - 1);
      }
    s.insert(g(q));
  }
  return dp[bit] = mex(s);
}
int main() {
  int n;
  scanf(\"%d\", &n);
  for (int i = 0; i < n; i++) scanf(\"%lld\", v + i);
  map<int, int> m;
  for (int i = 0; i < n; i++) {
    long long x = v[i];
    for (long long j = 2; j * j <= x; j++) {
      if (x % j == 0) {
        int tmp = 0;
        while (x % j == 0) tmp++, x /= j;
        m[j] |= 1 << (tmp - 1);
      }
    }
    if (x > 1) m[x] |= 1;
  }
  dp[0] = 0;
  int ans = 0;
  for (auto x : m) ans ^= g(x.second);
  printf(\"%s\n\", ans ? \"Mojtaba\" : \"Arpa\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import math
from collections import defaultdict
from bootcamp import Basebootcamp

class Carpaandagamewithmojtababootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'min_n': 1,
            'max_n': 100,
            'max_prime': 50,    # 最大质数范围
            'max_factors': 3,   # 每个数的最大质因子数
            'max_exponent': 5   # 每个因子的最大指数
        }
        self.params.update(params)
    
    def case_generator(self):
        n = random.randint(self.params['min_n'], self.params['max_n'])
        primes = self._generate_primes(self.params['max_prime'], self.params['max_factors'] + 1)
        a = []
        for _ in range(n):
            num = self._generate_number(primes)
            a.append(num)
        return {'n': n, 'a': a}
    
    def _generate_primes(self, max_val, count):
        primes = []
        for num in range(2, max_val + 1):
            if self._is_prime(num):
                primes.append(num)
                if len(primes) >= count:
                    break
        return primes
    
    def _generate_number(self, available_primes):
        factors = random.sample(available_primes, random.randint(0, min(len(available_primes), self.params['max_factors'])))
        number = 1
        for p in factors:
            exponent = random.randint(1, self.params['max_exponent'])
            number *= p ** exponent
        return number if number != 1 else random.choice([1, 1, 1, 2])  # 增加1的概率但允许少量质数
    
    @staticmethod
    def _is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        return (
            "Mojtaba and Arpa are playing a number game. The rules are:\n"
            "1. On a turn, choose a prime power p^k that divides at least one number\n"
            "2. For each x divisible by p^k, replace x with x/(p^k)\n"
            "3. The player who cannot make a move loses\n\n"
            f"Input:\n{n}\n{a}\n\n"
            "Determine the winner (Mojtaba or Arpa). Put your final answer within [answer]...[/answer]."
        )
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE | re.DOTALL)
        if matches:
            ans = matches[-1].strip().lower()
            if ans in {'mojtaba', 'arpa'}:
                return ans.capitalize()
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        m = defaultdict(int)
        
        def factorize(x):
            factors = {}
            if x == 1:
                return factors
            while x % 2 == 0:
                factors[2] = factors.get(2, 0) + 1
                x //= 2
            i = 3
            while i * i <= x:
                while x % i == 0:
                    factors[i] = factors.get(i, 0) + 1
                    x //= i
                i += 2
            if x > 1:
                factors[x] = 1
            return factors
        
        for x in a:
            factors = factorize(x)
            for p, k in factors.items():
                m[p] |= 1 << (k - 1)
        
        dp = {}
        def mex(s):
            ans = 0
            while ans in s:
                ans += 1
            return ans
        
        def grundy(bit):
            if bit in dp:
                return dp[bit]
            if bit == 0:
                return 0
            lg = bit.bit_length() - 1
            s = set()
            for i in range(lg + 1):
                q = 0
                for j in range(lg + 1):
                    if bit & (1 << j):
                        if j < i:
                            q |= 1 << j
                        elif j > i:
                            new_pos = j - i - 1
                            if new_pos >= 0:
                                q |= 1 << new_pos
                s.add(grundy(q))
            dp[bit] = mex(s)
            return dp[bit]
        
        total_xor = 0
        for p in m:
            # 重置缓存确保不同质数独立计算
            dp.clear()
            total_xor ^= grundy(m[p])
        
        correct = 'Mojtaba' if total_xor != 0 else 'Arpa'
        return solution.strip().lower() == correct.lower()

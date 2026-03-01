"""# 

### 谜题描述
Let's call an array consisting of n integer numbers a1, a2, ..., an, beautiful if it has the following property:

  * consider all pairs of numbers x, y (x ≠ y), such that number x occurs in the array a and number y occurs in the array a; 
  * for each pair x, y must exist some position j (1 ≤ j < n), such that at least one of the two conditions are met, either aj = x, aj + 1 = y, or aj = y, aj + 1 = x. 



Sereja wants to build a beautiful array a, consisting of n integers. But not everything is so easy, Sereja's friend Dima has m coupons, each contains two integers qi, wi. Coupon i costs wi and allows you to use as many numbers qi as you want when constructing the array a. Values qi are distinct. Sereja has no coupons, so Dima and Sereja have made the following deal. Dima builds some beautiful array a of n elements. After that he takes wi rubles from Sereja for each qi, which occurs in the array a. Sereja believed his friend and agreed to the contract, and now he is wondering, what is the maximum amount of money he can pay.

Help Sereja, find the maximum amount of money he can pay to Dima.

Input

The first line contains two integers n and m (1 ≤ n ≤ 2·106, 1 ≤ m ≤ 105). Next m lines contain pairs of integers. The i-th line contains numbers qi, wi (1 ≤ qi, wi ≤ 105).

It is guaranteed that all qi are distinct.

Output

In a single line print maximum amount of money (in rubles) Sereja can pay.

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5 2
1 2
2 3


Output

5


Input

100 3
1 2
2 1
3 1


Output

4


Input

1 2
1 1
2 100


Output

100

Note

In the first sample Sereja can pay 5 rubles, for example, if Dima constructs the following array: [1, 2, 1, 2, 2]. There are another optimal arrays for this test.

In the third sample Sereja can pay 100 rubles, if Dima constructs the following array: [2].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long n, m;
long long a[2000007];
void input();
void solve();
int main() {
  input();
  solve();
  return 0;
}
void input() {
  scanf(\"%I64d%I64d\", &n, &m);
  int i;
  int x;
  for (i = 1; i <= m; i++) {
    scanf(\"%d%I64d\", &x, &a[i]);
  }
  sort(a + 1, a + m + 1);
  reverse(a + 1, a + m + 1);
}
void solve() {
  long long i, c;
  long long ans = 0;
  for (i = 1; i <= m; i++) {
    c = i * (i - 1) / 2;
    if (i % 2 == 0)
      c += i / 2;
    else
      c++;
    if (c <= n) ans += a[i];
  }
  printf(\"%I64d\n\", ans);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Eserejaandthearrangementofnumbersbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_m=100, max_w=1000):
        self.max_n = min(max_n, 2 * 10**6)
        self.max_m = min(max_m, 10**5)
        self.max_w = max_w
    
    def case_generator(self):
        # Enforce problem constraints for input parameters
        max_allowed_m = 10**5
        max_allowed_n = 2 * 10**6
        
        m = random.randint(1, min(self.max_m, max_allowed_m))
        n = random.randint(1, min(self.max_n, max_allowed_n))
        
        # Generate unique qi values within [1,1e5]
        q_samples = random.sample(range(1, 10**5 + 1), m)
        w_list = [random.randint(1, self.max_w) for _ in range(m)]
        
        # Create unsorted coupons list
        coupons = list(zip(q_samples, w_list))
        
        # Create sorted copy for answer calculation
        sorted_coupons = sorted(coupons, key=lambda x: -x[1])
        
        sum_result = 0
        for i in range(1, len(sorted_coupons) + 1):
            # Calculate required array length for i coupons
            c = i * (i - 1) // 2
            if i % 2 == 0:
                c += i // 2
            else:
                c += 1
            
            if c <= n:
                sum_result += sorted_coupons[i-1][1]
            else:
                break
        
        return {
            'n': n,
            'm': m,
            'coupons': coupons,  # Keep original input order
            'correct_answer': sum_result
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        coupons = question_case['coupons']
        
        input_lines = [f"{n} {m}"] + [f"{q} {w}" for q, w in coupons]
        input_str = '\n'.join(input_lines)
        
        prompt = f"""You are Sereja and need to determine the maximum amount you might have to pay Dima. Dima will construct a beautiful array using his coupons, and you must find the maximum possible total cost based on the coupons used.

A beautiful array a of length n satisfies the following condition:
- For every pair of distinct numbers x and y present in the array, there exists at least one position j (1 ≤ j < n) where either a[j] = x and a[j+1] = y, or a[j] = y and a[j+1] = x.

Dima has m coupons. Each coupon i allows unlimited use of a distinct number q_i in the array. Using coupon i costs w_i rubles, added to the total if q_i appears at least once in the array. The goal is to maximize the total cost by selecting coupons such that the resulting array is beautiful.

Input:
{input_str}

Your task is to compute the maximum possible total cost Sereja might pay. Place your final answer within [answer] and [/answer] tags, e.g., [answer]123[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

"""# 

### 谜题描述
On the math lesson a teacher asked each pupil to come up with his own lucky numbers. As a fan of number theory Peter chose prime numbers. Bob was more original. He said that number t is his lucky number, if it can be represented as: 

t = a2 + b2,  where a, b are arbitrary positive integers.

Now, the boys decided to find out how many days of the interval [l, r] (l ≤ r) are suitable for pair programming. They decided that the day i (l ≤ i ≤ r) is suitable for pair programming if and only if the number i is lucky for Peter and lucky for Bob at the same time. Help the boys to find the number of such days.

Input

The first line of the input contains integer numbers l, r (1 ≤ l, r ≤ 3·108).

Output

In the only line print the number of days on the segment [l, r], which are lucky for Peter and Bob at the same time.

Examples

Input

3 5


Output

1


Input

6 66


Output

7

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
bitset<300000001> gao;
int l, r;
int main() {
  gao.set();
  gao[1] = 0;
  scanf(\"%d%d\", &l, &r);
  for (int i = 3; i * i <= r; i += 2)
    if (gao[i]) {
      int t = i << 1;
      for (int j = i * i; j <= r; j += t) gao[j] = false;
    }
  int ans = l <= 2 && r > 1;
  for (int i = 1 + ((l - 1) / 4 + ((l - 1) % 4 != 0)) * 4; i <= r; i += 4)
    ans += gao[i];
  printf(\"%d\n\", ans);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from math import isqrt
from bootcamp import Basebootcamp

class Edoublehappinessbootcamp(Basebootcamp):
    def __init__(self, max_limit=100000):
        """
        Initialize bootcamp with safe parameters for case generation
        - max_limit capped at 1e6 to prevent memory overflow
        - Dynamic sieve generation for case validity
        """
        super().__init__()
        self.max_limit = min(max_limit, 1000000)  # Safety cap

    def case_generator(self):
        """Generate valid cases using optimized prime checking"""
        # Generate valid range with safe boundaries
        r = random.randint(2, self.max_limit)
        l = random.randint(1, r)
        
        # Count qualifying numbers using optimized method
        count = 0
        if l <= 2 <= r:
            count += 1
        
        # Check numbers congruent 1 mod4 for primality
        start = max(l, 5)
        for num in range(start, r + 1):
            if num % 4 != 1:
                continue
            if self.is_prime(num):
                count += 1
                
        return {'l': l, 'r': r, 'answer': count}

    @staticmethod
    def is_prime(n):
        """Optimized primality test"""
        if n <= 1:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for i in range(5, isqrt(n)+1, 6):
            if n % i == 0 or n % (i+2) == 0:
                return False
        return True

    @staticmethod
    def prompt_func(question_case) -> str:
        l, r = question_case['l'], question_case['r']
        return f"""Find numbers between {l} and {r} (inclusive) that are:
1. Prime numbers
2. Expressible as sum of two squares (a² + b²)

Output format: [answer]<number>[/answer]
Example: For input 3-5, correct answer is [answer]1[/answer]

Calculate for {l}-{r}:"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']

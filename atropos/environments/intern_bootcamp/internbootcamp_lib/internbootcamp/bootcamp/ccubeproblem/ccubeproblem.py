"""# 

### 谜题描述
Yaroslav, Andrey and Roman love playing cubes. Sometimes they get together and play cubes for hours and hours! 

Today they got together again and they are playing cubes. Yaroslav took unit cubes and composed them into an a × a × a cube, Andrey made a b × b × b cube and Roman made a c × c × c cube. After that the game was finished and the guys left. But later, Vitaly entered the room. He saw the cubes and wanted to make a cube as well. But what size should the cube be? Of course it should be a large cube with the side of length a + b + c. Besides, Vitaly decided to decompose the cubes built by Yaroslav, Andrey and Roman and compose his own large cube out of them. However, it turned out that the unit cubes he got from destroying the three cubes just weren't enough to make a large cube. We know that Vitaly was short of exactly n cubes. Vitaly got upset, demolished everything and left. As he was leaving, he met Petya and told him that there had been three cubes in the room and that he needed another n unit cubes to make his own large cube.

Petya entered the room and saw the messily scattered cubes. He wanted to make it neat and orderly again. But he only knows that there had been three cubes, made of small unit cubes and that Vitaly needed n more unit cubes to make a large one! Help Petya understand, how many ways of sizes a, b, c are there to restore Yaroslav's, Andrey's and Roman's cubes.

Input

The single line of the input contains integer n (1 ≤ n ≤ 1014). We know that all numbers a, b, c are positive integers.

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

In the single line print the required number of ways. If it turns out that there isn't a single way of suitable sizes of a, b, c, print 0. 

Examples

Input

24


Output

1


Input

648


Output

7


Input

5


Output

0


Input

93163582512000


Output

39090

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int main() {
  long long n;
  cin >> n;
  if (n % 3 != 0) {
    cout << 0 << endl;
    return 0;
  }
  n /= 3;
  int ans = 0;
  for (long long i = 2; i * i * i <= n; ++i) {
    if (n % i != 0) continue;
    long long r = n / i;
    for (long long j = max(
             i, (-i + (long long)sqrt((long double)i * i + 4 * r)) / 2 - 10);
         j * j <= r; ++j) {
      if (r % j != 0) continue;
      long long k = r / j;
      long long a = i - j + k;
      long long b = i + j - k;
      long long c = -i + j + k;
      if (a <= 0 || b <= 0 || c <= 0 || a % 2) continue;
      if (a == b && b == c) {
        ans++;
        continue;
      }
      if (a == b || a == c || b == c) {
        ans += 3;
        continue;
      }
      ans += 6;
    }
  }
  cout << ans << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Ccubeproblembootcamp(Basebootcamp):
    def __init__(self, max_abc=50):
        """
        Initialize with parameters to control problem generation.
        :param max_abc: Maximum value for a, b, c (default 50)
        """
        self.max_abc = max_abc

    def case_generator(self):
        """
        Generates valid or invalid test cases:
        - Valid: n = 3*(a+b)(a+c)(b+c) with positive integers a, b, c
        - Invalid: n not divisible by 3
        """
        # Randomly choose valid or invalid cases
        if random.random() < 0.5:
            # Generate valid (a, b, c)
            a = random.randint(1, self.max_abc)
            b = random.randint(1, self.max_abc)
            c = random.randint(1, self.max_abc)
            n = 3 * (a + b) * (a + c) * (b + c)
        else:
            # Generate invalid n (not divisible by 3)
            n = random.randint(1, 10**14)
            while n % 3 == 0:
                n = random.randint(1, 10**14)
        
        answer = self.compute_answer(n)
        return {'n': n, 'answer': answer}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        prompt = f"""You are a programming assistant solving a mathematical puzzle. 

Problem:
Given an integer n, compute the number of ordered triples (a, b, c) of positive integers that satisfy:
3 × (a + b)(a + c)(b + c) = {n}

Input:
A single integer n (1 ≤ n ≤ 1e14).

Output:
The number of valid triples. If none exist, output 0.

Examples:
Input: 24 → Output: 1
Input: 5 → Output: 0

Answer within [answer]...[/answer] tags. For example: [answer]0[/answer].

Your task:
Given n = {n}, provide the correct answer."""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last = matches[-1].strip()
        try:
            return int(last)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']

    @staticmethod
    def compute_answer(n):
        if n % 3 != 0:
            return 0
        m = n // 3
        ans = 0
        i = 2
        while i * i * i <= m:
            if m % i != 0:
                i += 1
                continue
            r = m // i
            # Calculate j start value
            sqrt_val = math.isqrt(i * i + 4 * r)
            j_start_candidate = (-i + sqrt_val) // 2 - 10
            j_start = max(i, j_start_candidate, 1)
            max_j = math.isqrt(r)
            j = j_start
            while j <= max_j:
                if r % j == 0:
                    k = r // j
                    a_comb = i - j + k
                    b_comb = i + j - k
                    c_comb = -i + j + k
                    if (a_comb > 0 and b_comb > 0 and c_comb > 0 
                        and a_comb % 2 == 0):
                        a = a_comb // 2
                        b = b_comb // 2
                        c = c_comb // 2
                        if a > 0 and b > 0 and c > 0:
                            if a == b and b == c:
                                ans += 1
                            elif a == b or a == c or b == c:
                                ans += 3
                            else:
                                ans += 6
                j += 1
            i += 1
        return ans

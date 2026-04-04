"""# 

### 谜题描述
The Little Elephant has two permutations a and b of length n, consisting of numbers from 1 to n, inclusive. Let's denote the i-th (1 ≤ i ≤ n) element of the permutation a as ai, the j-th (1 ≤ j ≤ n) element of the permutation b — as bj.

The distance between permutations a and b is the minimum absolute value of the difference between the positions of the occurrences of some number in a and in b. More formally, it's such minimum |i - j|, that ai = bj.

A cyclic shift number i (1 ≤ i ≤ n) of permutation b consisting from n elements is a permutation bibi + 1... bnb1b2... bi - 1. Overall a permutation has n cyclic shifts.

The Little Elephant wonders, for all cyclic shifts of permutation b, what is the distance between the cyclic shift and permutation a?

Input

The first line contains a single integer n (1 ≤ n ≤ 105) — the size of the permutations. The second line contains permutation a as n distinct numbers from 1 to n, inclusive. The numbers are separated with single spaces. The third line contains permutation b in the same format.

Output

In n lines print n integers — the answers for cyclic shifts. Print the answers to the shifts in the order of the shifts' numeration in permutation b, that is, first for the 1-st cyclic shift, then for the 2-nd, and so on.

Examples

Input

2
1 2
2 1


Output

1
0


Input

4
2 1 3 4
3 4 2 1


Output

2
1
0
1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int Maxn = 100 * 1000 + 10;
int n, inda[Maxn], indb[Maxn];
set<pair<int, int> > a, b, b1;
int main() {
  scanf(\"%d\", &n);
  int aa;
  for (int i = 0; i < n; i++) {
    scanf(\"%d\", &aa);
    aa--;
    inda[aa] = i;
  }
  for (int i = 0; i < n; i++) {
    scanf(\"%d\", &aa);
    aa--;
    indb[aa] = i;
  }
  for (int i = 0; i < n; i++) {
    if (indb[i] > inda[i])
      a.insert(make_pair(indb[i] - inda[i], inda[i]));
    else
      b.insert(make_pair(inda[i] - indb[i], indb[i])),
          b1.insert(make_pair(indb[i], inda[i] - indb[i]));
  }
  for (int i = 0; i < n; i++) {
    printf(\"%d\n\",
           min((b.begin()->first + i) % n, (a.begin()->first - i + n) % n));
    if (b1.begin()->first == i) {
      int ind = b1.begin()->second + i;
      a.insert(make_pair(n - ind + i, ind));
      b.erase(make_pair(b1.begin()->second, b1.begin()->first));
      b1.erase(b1.begin());
    }
    while (!a.empty() && a.begin()->first == i + 1) {
      b.insert(make_pair(-i - 1, a.begin()->second + i + 1));
      b1.insert(make_pair(a.begin()->second + i + 1, -i - 1));
      a.erase(a.begin());
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
from bootcamp import Basebootcamp

class Elittleelephantandshiftsbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        Initialize the Elittleelephantandshiftsbootcamp with parameters.
        Parameters:
            params (dict): Contains puzzle parameters, e.g., 'n' for permutation size.
        """
        super().__init__()  # Corrected: No parameters passed to super()
        self.n = params.get('n', 4)

    def case_generator(self):
        """
        Generates a problem instance with random permutations a and b, and precomputes the correct answers.
        Returns a JSON-serializable dictionary containing the instance data.
        """
        n = self.n
        a = list(range(1, n + 1))
        random.shuffle(a)
        b = list(range(1, n + 1))
        random.shuffle(b)

        # Precompute positions for each element in a and b (0-based)
        inda = {num-1: idx for idx, num in enumerate(a)}
        indb = {num-1: idx for idx, num in enumerate(b)}

        # Compute the correct output for each cyclic shift
        correct_outputs = []
        for k in range(n):
            min_distance = min(
                abs((indb[x] - k) % n - inda[x])
                for x in range(n)
            )
            correct_outputs.append(min_distance)

        return {
            'n': n,
            'a': a,
            'b': b,
            'correct_outputs': correct_outputs
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        Converts the generated problem instance into a formatted question string.
        """
        n = question_case['n']
        a_str = ' '.join(map(str, question_case['a']))
        b_str = ' '.join(map(str, question_case['b']))
        prompt = f"""You are a programming assistant. Solve the following puzzle:

Given two permutations a and b of length {n}, compute the distance between permutation a and each cyclic shift of permutation b. The distance is defined as the minimum absolute difference between the positions of the same number in a and the shifted b. 

Input format:
- First line: integer n
- Second line: permutation a (space-separated integers)
- Third line: permutation b (space-separated integers)

Output format:
Output {n} lines, each containing the distance for the corresponding cyclic shift of b. The shifts are numbered from 1 to n.

Example:
If n=2, a is 1 2, and b is 2 1, the correct output is:
1
0

Now, solve the following problem:
Input:
{n}
{a_str}
{b_str}

Write your answer as exactly {n} integers, each on a new line, enclosed within [answer] and [/answer] tags."""
        return prompt

    @staticmethod
    def extract_output(output):
        """
        Extracts the solution from the model's output, expecting the last [answer] block.
        Returns a list of integers or None if extraction fails.
        """
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        solution = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    solution.append(int(line))
                except ValueError:
                    return None
        return solution

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        Verifies if the extracted solution matches the precomputed correct outputs.
        """
        correct_outputs = identity.get('correct_outputs', [])
        return solution == correct_outputs if solution is not None else False

"""# 

### 谜题描述
Petya loves lucky numbers. Everybody knows that lucky numbers are positive integers whose decimal representation contains only the lucky digits 4 and 7. For example, numbers 47, 744, 4 are lucky and 5, 17, 467 are not.

One day Petya dreamt of a lexicographically k-th permutation of integers from 1 to n. Determine how many lucky numbers in the permutation are located on the positions whose indexes are also lucky numbers.

Input

The first line contains two integers n and k (1 ≤ n, k ≤ 109) — the number of elements in the permutation and the lexicographical number of the permutation.

Output

If the k-th permutation of numbers from 1 to n does not exist, print the single number \"-1\" (without the quotes). Otherwise, print the answer to the problem: the number of such indexes i, that i and ai are both lucky numbers.

Examples

Input

7 4


Output

1


Input

4 7


Output

1

Note

A permutation is an ordered set of n elements, where each integer from 1 to n occurs exactly once. The element of permutation in position with index i is denoted as ai (1 ≤ i ≤ n). Permutation a is lexicographically smaller that permutation b if there is such a i (1 ≤ i ≤ n), that ai < bi, and for any j (1 ≤ j < i) aj = bj. Let's make a list of all possible permutations of n elements and sort it in the order of lexicographical increasing. Then the lexicographically k-th permutation is the k-th element of this list of permutations.

In the first sample the permutation looks like that:

1 2 3 4 6 7 5

The only suitable position is 4.

In the second sample the permutation looks like that:

2 1 3 4

The only suitable position is 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
F = [1];
while len(F) <= 13:
    F.append(F[-1]*len(F));

def get_perm(k,S):
    if k >= F[len(S)]:
        print -1;
        exit(0);
    P = [];
    while k:
        i = 0;
        while F[len(S)-1]*(i+1) <= k:
            i += 1;
        k -= F[len(S)-1]*i;
        P.append(S[i]);
        S.pop(i);
    P += S;
    return P;

q = [0];
lucky = [];
while len(q):
    u = q.pop(0);
    if u > 10**9: continue;
    if u: lucky.append(u);
    q.append(u*10 + 4);
    q.append(u*10 + 7);

n,k = map(int,raw_input().split());
k -= 1;
while len(lucky) and lucky[-1] > n:
    lucky.pop();

L = min(13,n);
s = n - L + 1;
e = n;
ans = 0;
while len(lucky) and lucky[0] < s:
    ans += 1;
    lucky.pop(0);


P = get_perm(k,range(s,e+1));
for i in xrange(L):
    if (i + s) in lucky and P[i] in lucky:
        ans += 1;
print ans;
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque
from bootcamp import Basebootcamp

class Eluckypermutationbootcamp(Basebootcamp):
    _factorials = [1]
    while len(_factorials) <= 13:
        _factorials.append(_factorials[-1] * len(_factorials))

    def __init__(self, max_n=10**9, valid_case_prob=0.5):
        self.max_n = max_n
        self.valid_case_prob = valid_case_prob

    def case_generator(self):
        if random.random() < self.valid_case_prob:
            # Generate valid cases (k <= possible permutations)
            n = random.choice([
                random.randint(1, 12),
                random.randint(13, 20)  # Ensure coverage of n>=13 cases
            ])
            max_fact = self._factorials[min(n, 13)]
            k = random.randint(1, max_fact)
        else:
            # Generate invalid cases (k > possible permutations)
            n = random.randint(1, self.max_n)
            max_fact = self._factorials[min(n, 13)] if n <= 13 else 0
            k = random.randint(max(1, max_fact + 1), 10**9)
        return {'n': n, 'k': k}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        problem_desc = f"""Solve the following lucky permutation problem:
- Lucky numbers contain only 4 and 7 digits (e.g. 4, 7, 44, 747)
- Find the number of positions i (1-based) in the {k}-th lex permutation of 1..{n}
  where both i and a_i are lucky numbers
- If there are fewer than {k} permutations, output -1

Format your answer as [answer]N[/answer] where N is the result."""
        return problem_desc

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = cls._calculate_expected(identity['n'], identity['k'])
            return solution == expected
        except:
            return False

    @classmethod
    def _calculate_expected(cls, n, k_input):
        # Generate all lucky numbers up to n
        lucky = []
        q = deque([0])
        while q:
            u = q.popleft()
            if u > n:
                continue
            if u > 0:
                lucky.append(u)
            q.append(u * 10 + 4)
            q.append(u * 10 + 7)
        lucky = sorted(lucky)

        L = min(13, n)
        s = n - L + 1
        if s < 1:
            s = 1

        # Count lucky indices before s
        pre_count = sum(1 for x in lucky if x < s)

        # Check if permutation is possible
        if L == 0 or k_input > cls._factorials[L]:
            return -1 if L > 0 else 0

        # Generate permutation suffix
        suffix = list(range(s, n+1))
        k = k_input - 1
        perm = []
        while suffix and k > 0:
            fact = cls._factorials[len(suffix)-1]
            idx = 0
            while (idx + 1) * fact <= k:
                idx += 1
            perm.append(suffix[idx])
            del suffix[idx]
            k -= idx * fact
        perm += suffix

        # Count valid positions in permutation
        count = 0
        for i in range(len(perm)):
            pos = s + i
            if pos > n:
                break
            if pos in lucky and perm[i] in lucky:
                count += 1

        return pre_count + count if len(perm) == L else -1

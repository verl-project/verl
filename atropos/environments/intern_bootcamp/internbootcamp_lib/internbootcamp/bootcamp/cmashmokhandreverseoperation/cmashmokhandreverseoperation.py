"""# 

### 谜题描述
Mashmokh's boss, Bimokh, didn't like Mashmokh. So he fired him. Mashmokh decided to go to university and participate in ACM instead of finding a new job. He wants to become a member of Bamokh's team. In order to join he was given some programming tasks and one week to solve them. Mashmokh is not a very experienced programmer. Actually he is not a programmer at all. So he wasn't able to solve them. That's why he asked you to help him with these tasks. One of these tasks is the following.

You have an array a of length 2n and m queries on it. The i-th query is described by an integer qi. In order to perform the i-th query you must:

  * split the array into 2n - qi parts, where each part is a subarray consisting of 2qi numbers; the j-th subarray (1 ≤ j ≤ 2n - qi) should contain the elements a[(j - 1)·2qi + 1], a[(j - 1)·2qi + 2], ..., a[(j - 1)·2qi + 2qi]; 
  * reverse each of the subarrays; 
  * join them into a single array in the same order (this array becomes new array a); 
  * output the number of inversions in the new a. 



Given initial array a and all the queries. Answer all the queries. Please, note that the changes from some query is saved for further queries.

Input

The first line of input contains a single integer n (0 ≤ n ≤ 20). 

The second line of input contains 2n space-separated integers a[1], a[2], ..., a[2n] (1 ≤ a[i] ≤ 109), the initial array.

The third line of input contains a single integer m (1 ≤ m ≤ 106). 

The fourth line of input contains m space-separated integers q1, q2, ..., qm (0 ≤ qi ≤ n), the queries.

Note: since the size of the input and output could be very large, don't use slow output techniques in your language. For example, do not use input and output streams (cin, cout) in C++.

Output

Output m lines. In the i-th line print the answer (the number of inversions) for the i-th query.

Examples

Input

2
2 1 4 3
4
1 2 0 2


Output

0
6
6
0


Input

1
1 2
3
0 1 1


Output

0
1
0

Note

If we reverse an array x[1], x[2], ..., x[n] it becomes new array y[1], y[2], ..., y[n], where y[i] = x[n - i + 1] for each i.

The number of inversions of an array x[1], x[2], ..., x[n] is the number of pairs of indices i, j such that: i < j and x[i] > x[j].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
    n = int(raw_input())
    aa = list(map(int, raw_input().split()))
    raw_input()
    res = []
    if n < 2:
        l, f = (\"01\"[aa[0] > aa[-1]], \"01\"[aa[0] < aa[-1]]), False
        for q in raw_input()[::2]:
            f ^= q == '1'
            res.append(l[f])
        print('\n'.join(res))
        return
    n2 = 2 ** n
    a00 = a01 = a10 = a11 = 0
    for i in xrange(0, n2, 4):
        a, b, c, d = aa[i:i + 4]
        a00 += (b < a) + (d < c)
        a01 += (c < a) + (c < b) + (d < a) + (d < b)
        a10 += (b > a) + (d > c)
        a11 += (c > a) + (c > b) + (d > a) + (d > b)
    acc0 = [a00, a01]
    acc1 = [a10, a11]
    w = 4
    while w < n2:
        a00 = a10 = 0
        for i in xrange(0, n2, w * 2):
            le = sorted(aa[i:i + w])
            ri = sorted(aa[i + w:i + w * 2])
            i = j = 0
            try:
                while True:
                    if le[i] > ri[j]:
                        j += 1
                    else:
                        a00 += j
                        i += 1
            except IndexError:
                if i < j:
                    a00 += j * (w - i)
            i = j = 0
            try:
                while True:
                    if ri[i] > le[j]:
                        j += 1
                    else:
                        a10 += j
                        i += 1
            except IndexError:
                if i < j:
                    a10 += j * (w - i)
        acc0.append(a00)
        acc1.append(a10)
        w *= 2
    for q in map(int, raw_input().split()):
        while q:
            q -= 1
            acc0[q], acc1[q] = acc1[q], acc0[q]
        res.append(sum(acc0))
    print('\n'.join(map(str, res)))


if __name__ == '__main__':
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_answers(n, a, queries):
    aa = a.copy()
    m = len(queries)
    res = []
    
    if n == 0:
        return [0] * m  # Only one element, no possible inversions
    
    if n < 2:
        a1, a2 = aa[0], aa[1]
        original_inversion = 1 if a1 > a2 else 0
        reversed_inversion = 1 if a2 > a1 else 0
        current_inversion = original_inversion
        f = False  # Tracks whether the array is reversed
        for q in queries:
            if q != 0:
                f = not f
                current_inversion = reversed_inversion if f else original_inversion
            res.append(current_inversion)
        return res
    
    n2 = 2 ** n
    acc0 = []
    acc1 = []
    
    # Initialize for q=1 and q=2 levels
    a00 = a01 = a10 = a11 = 0
    for i in range(0, n2, 4):
        a_val = aa[i]
        b_val = aa[i+1] if i+1 < n2 else 0
        c_val = aa[i+2] if i+2 < n2 else 0
        d_val = aa[i+3] if i+3 < n2 else 0
        
        a00 += (b_val < a_val) + (d_val < c_val)
        a01 += (c_val < a_val) + (c_val < b_val) + (d_val < a_val) + (d_val < b_val)
        a10 += (b_val > a_val) + (d_val > c_val)
        a11 += (c_val > a_val) + (c_val > b_val) + (d_val > a_val) + (d_val > b_val)
    
    acc0 = [a00, a01]
    acc1 = [a10, a11]
    w = 4
    
    while w < n2:
        a00 = 0
        a10 = 0
        for i in range(0, n2, w * 2):
            le = sorted(aa[i:i + w])
            ri = sorted(aa[i + w:i + w * 2])
            
            # Compute a00 (inversions from left to right)
            i_le, j_ri, cnt = 0, 0, 0
            while i_le < len(le) and j_ri < len(ri):
                if le[i_le] > ri[j_ri]:
                    j_ri += 1
                else:
                    cnt += j_ri
                    i_le += 1
            cnt += j_ri * (len(le) - i_le)
            a00 += cnt
            
            # Compute a10 (inversions from right to left)
            i_ri, j_le, cnt = 0, 0, 0
            while i_ri < len(ri) and j_le < len(le):
                if ri[i_ri] > le[j_le]:
                    j_le += 1
                else:
                    cnt += j_le
                    i_ri += 1
            cnt += j_le * (len(ri) - i_ri)
            a10 += cnt
        
        acc0.append(a00)
        acc1.append(a10)
        w *= 2
    
    # Handling queries by swapping acc0 and acc1 as needed
    for q in queries:
        current_q = q
        # Flip all levels up to q
        for level in range(current_q):
            if level < len(acc0):
                acc0[level], acc1[level] = acc1[level], acc0[level]
        res.append(sum(acc0))
        # Restore original state for next query
        for level in range(current_q):
            if level < len(acc0):
                acc0[level], acc1[level] = acc1[level], acc0[level]
                
    return res

class Cmashmokhandreverseoperationbootcamp(Basebootcamp):
    def __init__(self, max_n=3, m_min=1, m_max=5, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n
        self.m_min = m_min
        self.m_max = m_max
    
    def case_generator(self):
        n = random.randint(0, self.max_n)
        size_a = 2 ** n
        a = [random.randint(1, 10) for _ in range(size_a)] if size_a > 0 else []
        m = random.randint(self.m_min, self.m_max)
        valid_queries = [random.randint(0, n) for _ in range(m)]
        expected_outputs = compute_answers(n, a, valid_queries)
        return {
            'n': n,
            'a': a,
            'queries': valid_queries,
            'expected_outputs': expected_outputs
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        queries = question_case['queries']
        m = len(queries)
        problem_desc = f"""Cmashmokhandreverseoperation needs your help to solve an array transformation problem. Process a series of queries on an array and determine inversion counts after each transformation.

**Problem Rules:**
1. Start with an array of {2**n} elements: {a}
2. For each query q in {queries}:
   - Split the array into 2^(n - q) subarrays, each consisting of 2^q elements
   - Reverse each subarray
   - Reassemble the array while preserving subarray order
   - Compute the number of inversions in the new array and output it

**Task:**
Process all {m} queries in order and output the inversion count after each step. Provide your answers in [answer] tags with each result on a new line.

Example format:
[answer]
0
6
6
0
[/answer]

**Your Problem:**
Initial array (n={n}): {a}
Queries (m={m}): {queries}

Place your answers between [answer] and [/answer] tags, with each answer on a separate line:"""
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1]
        lines = [line.strip() for line in last_match.splitlines() if line.strip()]
        solutions = []
        for line in lines:
            try:
                solutions.append(int(line))
            except ValueError:
                continue
        return solutions if solutions else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('expected_outputs', [])
        if not isinstance(solution, list) or len(solution) != len(expected):
            return False
        return all(s == e for s, e in zip(solution, expected))

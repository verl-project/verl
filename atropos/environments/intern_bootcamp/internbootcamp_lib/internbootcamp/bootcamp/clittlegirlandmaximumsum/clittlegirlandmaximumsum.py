"""# 

### 谜题描述
The little girl loves the problems on array queries very much.

One day she came across a rather well-known problem: you've got an array of n elements (the elements of the array are indexed starting from 1); also, there are q queries, each one is defined by a pair of integers l_i, r_i (1 ≤ l_i ≤ r_i ≤ n). You need to find for each query the sum of elements of the array with indexes from l_i to r_i, inclusive.

The little girl found the problem rather boring. She decided to reorder the array elements before replying to the queries in a way that makes the sum of query replies maximum possible. Your task is to find the value of this maximum sum.

Input

The first line contains two space-separated integers n (1 ≤ n ≤ 2⋅10^5) and q (1 ≤ q ≤ 2⋅10^5) — the number of elements in the array and the number of queries, correspondingly.

The next line contains n space-separated integers a_i (1 ≤ a_i ≤ 2⋅10^5) — the array elements.

Each of the following q lines contains two space-separated integers l_i and r_i (1 ≤ l_i ≤ r_i ≤ n) — the i-th query.

Output

In a single line print, a single integer — the maximum sum of query replies after the array elements are reordered.

Please, do not use the %lld specifier to read or write 64-bit integers in C++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

3 3
5 3 2
1 2
2 3
1 3


Output

25


Input

5 3
5 2 4 1 3
1 5
2 3
2 3


Output

33

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
tamanho , queries = map(int ,raw_input().split())

lista = sorted(map(int, raw_input().split()))

freq = [0] * tamanho

for i in xrange(queries):
	c, f = map(int, raw_input().split())
	freq[c-1] += 1
	if f<tamanho:
		freq[f] += -1

for i in xrange(1, tamanho):
	freq[i] += freq[i-1]
	
saida = 0

freq.sort()

for i in xrange(tamanho):
	
	saida += lista[i] * freq[i]
print saida
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Clittlegirlandmaximumsumbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, min_q=1, max_q=10, max_a=200000):
        self.min_n = min_n
        self.max_n = max_n
        self.min_q = min_q
        self.max_q = max_q
        self.max_a = max_a
    
    def case_generator(self):
        # Randomize parameters within given ranges
        n = random.randint(self.min_n, self.max_n)
        q = random.randint(self.min_q, self.max_q)
        
        # Generate array with varied elements
        a = [random.randint(1, self.max_a) for _ in range(n)]
        if random.random() < 0.2:  # 20% chance for identical elements
            a = [a[0]] * n
        
        queries = []
        for _ in range(q):
            # Generate diverse query patterns
            rand = random.random()
            if rand < 0.3 and n > 0:  # Full range queries
                queries.append([1, n])
            elif rand < 0.5 and n > 0:  # Single element queries
                pos = random.randint(1, n)
                queries.append([pos, pos])
            else:
                l = random.randint(1, n)
                r = random.randint(l, n)
                queries.append([l, r])
        
        # Calculate frequency using optimized method
        freq = [0] * (n + 2)  # Using 1-based indexing with buffer
        for l, r in queries:
            freq[l] += 1
            freq[r + 1] -= 1
        
        # Compute prefix sum
        current = 0
        frequency = []
        for i in range(1, n + 1):
            current += freq[i]
            frequency.append(current)
        
        # Sort and calculate expected sum
        a_sorted = sorted(a)
        frequency_sorted = sorted(frequency)
        expected_sum = sum(x * y for x, y in zip(a_sorted, frequency_sorted))
        
        return {
            'n': n,
            'q': q,
            'array': a,
            'queries': queries,
            'expected_sum': expected_sum
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        arr_str = ' '.join(map(str, question_case['array']))
        queries_str = '\n'.join([f"{l} {r}" for l, r in question_case['queries']])
        
        return f"""You are given an array optimization task. Reorder the array elements to maximize the sum of query results.

Input:
- Array: {arr_str}
- {question_case['q']} queries:
{queries_str}

Rules:
1. You must permute the array elements
2. Each query [l, r] sums elements from position l to r (inclusive)
3. The goal is to find the maximum possible total sum of all queries

Output the final maximum sum enclosed in [answer] tags. For example: [answer]167[/answer]

Provide your answer:"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_sum']

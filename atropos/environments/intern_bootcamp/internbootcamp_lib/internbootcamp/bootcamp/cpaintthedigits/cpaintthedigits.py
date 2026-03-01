"""# 

### 谜题描述
You are given a sequence of n digits d_1d_2 ... d_{n}. You need to paint all the digits in two colors so that:

  * each digit is painted either in the color 1 or in the color 2; 
  * if you write in a row from left to right all the digits painted in the color 1, and then after them all the digits painted in the color 2, then the resulting sequence of n digits will be non-decreasing (that is, each next digit will be greater than or equal to the previous digit). 



For example, for the sequence d=914 the only valid coloring is 211 (paint in the color 1 two last digits, paint in the color 2 the first digit). But 122 is not a valid coloring (9 concatenated with 14 is not a non-decreasing sequence).

It is allowed that either of the two colors is not used at all. Digits painted in the same color are not required to have consecutive positions.

Find any of the valid ways to paint the given sequence of digits or determine that it is impossible to do.

Input

The first line contains a single integer t (1 ≤ t ≤ 10000) — the number of test cases in the input.

The first line of each test case contains an integer n (1 ≤ n ≤ 2⋅10^5) — the length of a given sequence of digits.

The next line contains a sequence of n digits d_1d_2 ... d_{n} (0 ≤ d_i ≤ 9). The digits are written in a row without spaces or any other separators. The sequence can start with 0.

It is guaranteed that the sum of the values ​​of n for all test cases in the input does not exceed 2⋅10^5.

Output

Print t lines — the answers to each of the test cases in the input.

If there is a solution for a test case, the corresponding output line should contain any of the valid colorings written as a string of n digits t_1t_2 ... t_n (1 ≤ t_i ≤ 2), where t_i is the color the i-th digit is painted in. If there are several feasible solutions, print any of them.

If there is no solution, then the corresponding output line should contain a single character '-' (the minus sign).

Example

Input


5
12
040425524644
1
0
9
123456789
2
98
3
987


Output


121212211211
1
222222222
21
-

Note

In the first test case, d=040425524644. The output t=121212211211 is correct because 0022444 (painted in 1) concatenated with 44556 (painted in 2) is 002244444556 which is a sorted sequence of n given digits.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import collections
import bisect
import random
import logging
import math


def solve(N, A):
    logging.debug('start')

    small, mid, large = -1, -1, -1
    pos = [-1] * 10
    B = ['1'] * N
    state = 0
    for i, n in enumerate(A):
        if pos[n] == -1:
            pos[n] = i

        logging.info(
            'i: {}, n: {}, state: {}, small: {}, mid: {}, large: {}'.format(
                i, n, state, small, mid, large))

        if state == 0:
            if n >= small:
                small = n
            else:
                state = 1
                small = n
                for m in xrange(n + 1, 10):
                    if pos[m] != -1:
                        mid = m
                        for j in xrange(pos[m], i):
                            if A[j] >= mid:
                                B[j] = '2'
                                large = A[j]
                        break
        else:
            if n < small:
                return '-'
            elif n >= large:
                large = n
                B[i] = '2'
            elif n <= mid:
                small = n
            else:
                return '-'

    return ''.join(B)


def run():
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    T = int(raw_input())
    for t in xrange(1, T + 1):
        N = int(raw_input())
        A = list(map(int, list(raw_input())))
        re = solve(N, A)
        print('{}'.format(re))


def test():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    l = ['040425524644', '0', '123456789', '98', '987']
    for i in l:
        print('in {}\nout {}'.format(i, solve(len(i), list(map(int,
                                                               list(i))))))


if __name__ == '__main__':
    # init()
    run()
    # test()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cpaintthedigitsbootcamp(Basebootcamp):
    def __init__(self, solvable_probability=0.5, min_length=1, max_length=10):
        self.solvable_probability = solvable_probability
        self.min_length = max(1, min_length)
        self.max_length = max(max_length, self.min_length)

    def case_generator(self):
        if random.random() < self.solvable_probability:
            # Generate solvable case
            n = random.randint(self.min_length, self.max_length)
            sorted_digits = self.generate_non_decreasing(n)
            split_pos = random.randint(0, n)
            c1 = sorted_digits[:split_pos]
            c2 = sorted_digits[split_pos:]
            
            # Build original sequence with color assignment
            merged = []
            i = j = 0
            while i < len(c1) or j < len(c2):
                if random.choice([True, False]) and i < len(c1) or j >= len(c2):
                    merged.append(('1', c1[i]))
                    i += 1
                else:
                    merged.append(('2', c2[j]))
                    j += 1
            
            # Create final digits string
            digits = ''.join(str(d) for (_, d) in merged)
            return {'n': len(digits), 'digits': digits}
        else:
            # Generate unsolvable case (strictly decreasing with length >= 2)
            n = random.randint(max(2, self.min_length), self.max_length)
            digits = [str(9 - i % 10) for i in range(n)]
            return {'n': n, 'digits': ''.join(digits)}

    def generate_non_decreasing(self, length):
        if length == 0:
            return []
        sequence = [random.randint(0, 9)]
        for _ in range(length-1):
            sequence.append(random.randint(sequence[-1], 9))
        return sequence

    @staticmethod
    def prompt_func(question_case) -> str:
        digits = question_case['digits']
        return f"""Given the digit sequence {digits}, color each digit with 1 or 2 such that:
1. All 1-colored digits in their original order
2. All 2-colored digits in their original order
3. Concatenation of 1's then 2's is non-decreasing

Provide your answer as a string of '1's and '2's with the same length, or '-' if impossible. Enclose your final answer within [answer] tags.

Example:
Input: 914
Answer: [answer]211[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip().replace(' ', '')
        if answer == '-':
            return '-'
        return answer if all(c in '12' for c in answer) and len(answer) > 0 else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        digits = list(identity['digits'])
        n = len(digits)
        
        if solution == '-':
            return cls.solve(n, list(map(int, digits))) == '-'
        
        if len(solution) != n or not all(c in '12' for c in solution):
            return False
        
        seq1 = []
        seq2 = []
        for d, c in zip(digits, solution):
            digit = int(d)
            if c == '1':
                seq1.append(digit)
            else:
                seq2.append(digit)
        
        merged = seq1 + seq2
        for i in range(len(merged)-1):
            if merged[i] > merged[i+1]:
                return False
        return True

    @staticmethod
    def solve(N, A):
        B = ['1'] * N
        last_1 = -1
        transition_point = None
        last_2 = -1
        
        for i in range(N):
            current = A[i]
            
            if last_1 == -1:
                last_1 = current
                continue
                
            if current >= last_1:
                last_1 = current
                continue
                
            if transition_point is None:
                # Find transition point
                transition_point = i
                min_2_val = current
                for m in range(current + 1, 10):
                    for j in range(i):
                        if A[j] == m and B[j] == '1':
                            transition_point = j
                            min_2_val = m
                            break
                    else:
                        continue
                    break
                else:
                    return '-'
                
                # Update colors for transition segment
                for j in range(transition_point):
                    if A[j] >= min_2_val:
                        B[j] = '2'
                last_2 = max(A[transition_point:i], default=-1)
                last_1 = current
            else:
                if current < last_1 or (current > last_2 and last_2 != -1):
                    return '-'
                if current >= last_2:
                    B[i] = '2'
                    last_2 = current
                else:
                    last_1 = current
        return ''.join(B)

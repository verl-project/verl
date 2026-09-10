"""# 

### 谜题描述
Everybody knows what an arithmetic progression is. Let us remind you just in case that an arithmetic progression is such sequence of numbers a1, a2, ..., an of length n, that the following condition fulfills: 

a2 - a1 = a3 - a2 = a4 - a3 = ... = ai + 1 - ai = ... = an - an - 1.

For example, sequences [1, 5], [10], [5, 4, 3] are arithmetic progressions and sequences [1, 3, 2], [1, 2, 4] are not.

Alexander has n cards containing integers. Arthur wants to give Alexander exactly one more card with a number so that he could use the resulting n + 1 cards to make an arithmetic progression (Alexander has to use all of his cards).

Arthur has already bought a card but he hasn't written a number on it. Help him, print all integers that you can write on a card so that the described condition fulfilled.

Input

The first line contains integer n (1 ≤ n ≤ 105) — the number of cards. The next line contains the sequence of integers — the numbers on Alexander's cards. The numbers are positive integers, each of them doesn't exceed 108.

Output

If Arthur can write infinitely many distinct integers on the card, print on a single line -1.

Otherwise, print on the first line the number of integers that suit you. In the second line, print the numbers in the increasing order. Note that the numbers in the answer can exceed 108 or even be negative (see test samples).

Examples

Input

3
4 1 7


Output

2
-2 10


Input

1
10


Output

-1


Input

4
1 3 5 9


Output

1
7


Input

4
4 3 4 5


Output

0


Input

2
2 4


Output

3
0 3 6

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/python
from __future__ import division
from sys import stdin, stdout, maxint
import fractions, math, bisect

N = int (stdin.readline ())
arr = map (int, stdin.readline ().split ())

if N == 1:
    print -1
else:
    arr.sort ()

    mindiff = maxint

    for n in xrange (N - 1):
        mindiff = min (mindiff, arr[n + 1] - arr[n])
    
    i, j, cnt = 0, 0, 0

    for n in xrange (N - 1):
        if arr[n + 1] - arr[n] != mindiff:
            cnt += 1
            i, j = n, n + 1
    
    ans = set ()
    
    if cnt == 0:
        ans.add (arr[0] - mindiff)
        ans.add (arr[N - 1] + mindiff)

        if N == 2 and (arr[1] - arr[0]) % 2 == 0:
            ans.add (arr[0] + (arr[1] - arr[0]) // 2)
    elif cnt == 1:
        insertion = arr[i] + mindiff

        if insertion + mindiff == arr[j]:
            ans.add (insertion)
    
    print len (ans)
    lst = sorted (list (ans))

    for e in lst:
        stdout.write (str (e) + ' ')
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Carithmeticprogressionbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, min_number=1, max_number=100):
        self.min_n = min_n
        self.max_n = max_n
        self.min_number = min_number
        self.max_number = max_number

    def case_generator(self):
        case_type = random.choices([1,2,3,4,5], weights=[0.1,0.2,0.3,0.3,0.1])[0]
        
        if case_type == 1:  # n=1 的测试用例
            n = 1
            numbers = [random.randint(self.min_number, self.max_number)]
        elif case_type == 2:  # n=2 且差为偶数的测试用例
            n = 2
            a = random.randint(self.min_number, self.max_number)
            d = random.choice([2,4,6,8,10])
            numbers = [a, a+d]
        elif case_type == 3:  # 需要插入元素的测试用例
            m = random.randint(3, 5)
            n = m-1
            a = random.randint(self.min_number, self.max_number)
            d = random.randint(1, 5)
            arr = [a + i*d for i in range(m)]
            del_index = random.randint(1, m-2)
            del arr[del_index]
            numbers = arr
        elif case_type == 4:  # 无解的测试用例
            n = 3
            while True:
                a = random.randint(self.min_number, self.max_number)
                d1 = random.randint(1, 5)
                d2 = random.randint(1, 5)
                if d1 != d2:
                    numbers = sorted([a, a+d1, a+d1+d2+1])
                    break
        else:  # 随机测试用例
            n = random.randint(self.min_n, self.max_n)
            numbers = [random.randint(self.min_number, self.max_number) for _ in range(n)]
        
        numbers = random.sample(numbers, len(numbers))  # 打乱顺序
        solutions = self._solve_problem(n, numbers)
        return {
            'n': n,
            'numbers': numbers,
            'solutions': solutions
        }

    @staticmethod
    def _solve_problem(n, numbers):
        if n == 1:
            return -1
        
        arr = sorted(numbers)
        mindiff = min(arr[i+1]-arr[i] for i in range(len(arr)-1))
        
        cnt = 0
        need_insert = None
        for i in range(len(arr)-1):
            diff = arr[i+1] - arr[i]
            if diff != mindiff:
                cnt += 1
                if diff == 2*mindiff:
                    need_insert = arr[i] + mindiff
                else:
                    return 0  # 存在无法修正的差值
        
        if cnt == 0:
            ans = {arr[0]-mindiff, arr[-1]+mindiff}
            if n == 2 and (arr[1]-arr[0])%2 == 0:
                ans.add(arr[0] + (arr[1]-arr[0])//2)
            return sorted(ans)
        elif cnt == 1 and need_insert is not None:
            return [need_insert]
        else:
            return 0

    @staticmethod
    def prompt_func(question_case) -> str:
        numbers_str = ' '.join(map(str, question_case['numbers']))
        return (
            f"Given {question_case['n']} numbers: {numbers_str}\n"
            "Find all possible numbers to add to form an arithmetic sequence.\n"
            "Output format:\n"
            "- If infinite solutions: [answer]-1[/answer]\n"
            "- If no solution: [answer]0[/answer]\n"
            "- Else: [answer]sorted_numbers[/answer]\n"
            "Example 1: [answer]-2 10[/answer]\n"
            "Example 2: [answer]-1[/answer]"
        )

    @staticmethod
    def extract_output(output):
        match = re.search(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()
        
        if content == '-1':
            return -1
        if content == '0':
            return 0
        
        try:
            return sorted(map(int, content.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['solutions']

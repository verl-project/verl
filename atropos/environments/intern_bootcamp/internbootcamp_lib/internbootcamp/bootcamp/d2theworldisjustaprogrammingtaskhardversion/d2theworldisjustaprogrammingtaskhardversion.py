"""# 

### 谜题描述
This is a harder version of the problem. In this version, n ≤ 300 000.

Vasya is an experienced developer of programming competitions' problems. As all great minds at some time, Vasya faced a creative crisis. To improve the situation, Petya gifted him a string consisting of opening and closing brackets only. Petya believes, that the beauty of the bracket string is a number of its cyclical shifts, which form a correct bracket sequence.

To digress from his problems, Vasya decided to select two positions of the string (not necessarily distinct) and swap characters located at this positions with each other. Vasya will apply this operation exactly once. He is curious what is the maximum possible beauty he can achieve this way. Please help him.

We remind that bracket sequence s is called correct if: 

  * s is empty; 
  * s is equal to \"(t)\", where t is correct bracket sequence; 
  * s is equal to t_1 t_2, i.e. concatenation of t_1 and t_2, where t_1 and t_2 are correct bracket sequences. 



For example, \"(()())\", \"()\" are correct, while \")(\" and \"())\" are not.

The cyclical shift of the string s of length n by k (0 ≤ k < n) is a string formed by a concatenation of the last k symbols of the string s with the first n - k symbols of string s. For example, the cyclical shift of string \"(())()\" by 2 equals \"()(())\".

Cyclical shifts i and j are considered different, if i ≠ j.

Input

The first line contains an integer n (1 ≤ n ≤ 300 000), the length of the string.

The second line contains a string, consisting of exactly n characters, where each of the characters is either \"(\" or \")\".

Output

The first line should contain a single integer — the largest beauty of the string, which can be achieved by swapping some two characters.

The second line should contain integers l and r (1 ≤ l, r ≤ n) — the indices of two characters, which should be swapped in order to maximize the string's beauty.

In case there are several possible swaps, print any of them.

Examples

Input


10
()()())(()


Output


5
8 7


Input


12
)(()(()())()


Output


4
5 10


Input


6
)))(()


Output


0
1 1

Note

In the first example, we can swap 7-th and 8-th character, obtaining a string \"()()()()()\". The cyclical shifts by 0, 2, 4, 6, 8 of this string form a correct bracket sequence.

In the second example, after swapping 5-th and 10-th character, we obtain a string \")(())()()(()\". The cyclical shifts by 11, 7, 5, 3 of this string form a correct bracket sequence.

In the third example, swap of any two brackets results in 0 cyclical shifts being correct bracket sequences. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def do(i):
    if i == \"(\":
        return 1
    else:
        return -1
def find(i,idx):
    if i<n-idx:
        return idx+i+1
    else:
        return i-(n-idx)+1
n = input()
arr = map(do,raw_input())
s = [0]*n
s[n-1] = arr[n-1]
maxi = n-1
maxv = s[n-1]

for i in range(n-1)[::-1]:
    s[i] = s[i+1] + arr[i]
    if s[i] > maxv:
        maxv = s[i]
        maxi = i
newv = arr[maxi:]+arr[:maxi]
if sum(newv) != 0:
    print 0
    print 1,1
else:
    cnt = 0
    cnt1 = -1
    cnt2 = -1
    maxv1,maxv2 = 0,0
    l1,l2,r1,r2 = 0,0,0,0
    last1,last2 = 0,0
    st = 0
    for i in range(n):
        st += newv[i]
        if st == 0:
            cnt += 1
            cnt1 = -1
            last1 = i+1

        elif st == 1:
            cnt1 += 1
            if cnt1 >= maxv1:
                maxv1 = cnt1
                l1 = last1
                r1 = i+1

            last2 = i+1
            cnt2 = -1
        elif st == 2:
            cnt2 += 1
            if cnt2 >= maxv2:
                maxv2 = cnt2
                l2 = last2
                r2 = i+1
    if maxv1 == 0:
        print cnt
        print 1,1
    elif maxv1>maxv2+cnt:
        print maxv1+1
        print find(l1,maxi),find(r1,maxi)
    else:
        print maxv2+cnt+1
        print find(l2,maxi),find(r2,maxi)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

class D2theworldisjustaprogrammingtaskhardversionbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 10)
        self.seed = params.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)
    
    def _generate_random_string(self, n):
        return ''.join(random.choice('()') for _ in range(n))
    
    def case_generator(self):
        n = self.n
        s = self._generate_random_string(n)
        max_beauty = 0
        best_l = 1
        best_r = 1
        for l in range(1, n+1):
            for r in range(l, n+1):
                s_list = list(s)
                s_list[l-1], s_list[r-1] = s_list[r-1], s_list[l-1]
                current_beauty = self.compute_beauty(''.join(s_list))
                if current_beauty > max_beauty:
                    max_beauty = current_beauty
                    best_l = l
                    best_r = r
        return {
            'n': n,
            's': s,
            'correct_l': best_l,
            'correct_r': best_r,
            'max_beauty': max_beauty
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['s']
        return f"Given a bracket string of length {n}: {s}, find two positions to swap to maximize the beauty. Output the maximum beauty, followed by the positions (1-based) on the next line, within [answer] tags."
    
    @staticmethod
    def extract_output(output):
        import re
        match = re.search(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()
        numbers = list(map(int, re.findall(r'\d+', content)))
        if len(numbers) < 3:
            return None
        return (numbers[1], numbers[2])
    
    @staticmethod
    def compute_beauty(s):
        n = len(s)
        if n == 0:
            return 0
        arr = [1 if c == '(' else -1 for c in s]
        total = sum(arr)
        if total != 0:
            return 0
        # Precompute the suffix sums
        suffix_sum = [0] * (n + 1)
        for i in range(n-1, -1, -1):
            suffix_sum[i] = suffix_sum[i+1] + arr[i]
        # Find the maximum suffix sum
        max_suffix = -float('inf')
        for i in range(n):
            if suffix_sum[i] > max_suffix:
                max_suffix = suffix_sum[i]
        # Compute the number of valid shifts
        count = 0
        for k in range(n):
            valid = True
            balance = 0
            for i in range(n):
                balance += arr[(k + i) % n]
                if balance < 0:
                    valid = False
                    break
            if valid and balance == 0:
                count += 1
        return count
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        l, r = solution
        n = identity['n']
        s = identity['s']
        s_list = list(s)
        l -= 1
        r -= 1
        s_list[l], s_list[r] = s_list[r], s_list[l]
        new_s = ''.join(s_list)
        beauty = cls.compute_beauty(new_s)
        return beauty == identity['max_beauty']

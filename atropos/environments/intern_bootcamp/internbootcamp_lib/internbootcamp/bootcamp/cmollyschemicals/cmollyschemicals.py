"""# 

### 谜题描述
Molly Hooper has n different kinds of chemicals arranged in a line. Each of the chemicals has an affection value, The i-th of them has affection value ai.

Molly wants Sherlock to fall in love with her. She intends to do this by mixing a contiguous segment of chemicals together to make a love potion with total affection value as a non-negative integer power of k. Total affection value of a continuous segment of chemicals is the sum of affection values of each chemical in that segment.

Help her to do so in finding the total number of such segments.

Input

The first line of input contains two integers, n and k, the number of chemicals and the number, such that the total affection value is a non-negative power of this number k. (1 ≤ n ≤ 105, 1 ≤ |k| ≤ 10).

Next line contains n integers a1, a2, ..., an ( - 109 ≤ ai ≤ 109) — affection values of chemicals.

Output

Output a single integer — the number of valid segments.

Examples

Input

4 2
2 2 2 2


Output

8


Input

4 -3
3 -6 -3 12


Output

3

Note

Do keep in mind that k0 = 1.

In the first sample, Molly can get following different affection values: 

  * 2: segments [1, 1], [2, 2], [3, 3], [4, 4];

  * 4: segments [1, 2], [2, 3], [3, 4];

  * 6: segments [1, 3], [2, 4];

  * 8: segments [1, 4]. 



Out of these, 2, 4 and 8 are powers of k = 2. Therefore, the answer is 8.

In the second sample, Molly can choose segments [1, 2], [3, 3], [3, 4].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import itertools
from fractions import gcd
from math import sqrt
from bisect import bisect_left , bisect_right
import heapq
from collections import deque
from itertools import combinations as C

def Ls():
	return list(raw_input())
def get(a):
	return map(a , raw_input().split())
def Int():
	return int(raw_input())
def Str():
	return raw_input()
from collections import defaultdict , deque

#Time to rise :)
def find_lt(a, x):
    i = bisect_left(a, x)
    if i:
        return i
    return -1


n , k  = get(int)
har = get(int)
pre = [0] * (n)
ocr = dict()
for i in xrange(n):
	if i == 0:
		pre[i] = har[i]
	else:
		pre[i] = pre[i-1] + har[i]
	if pre[i] not in ocr:
		ocr[pre[i]] = [i]
	else:
		ocr[pre[i]].append(i)
ans = 0
INF = 10**14+10#slow pypy
set_ = set()
for i in xrange(n):
	at_ = pre[i]
	par = i
	#print at_,
	#set_ = set()
	for j in xrange(0,51):
		to_ = k ** j
		#if to_ in set_:break
		#set_.add(to_)
		if abs(to_) > INF and k != 1 and k != -1:break
		if har[i] == to_:
			ans += 1
			#print i , 'yayyyy'
		if pre[i] == to_ and i != 0:
			ans += 1
			#print i , 'yoyoy'
		check_ = at_ - to_
		#print check_
		if check_ in ocr:
			arr = ocr[check_]
			ax = find_lt(arr,i)
			#print arr,check_,ax
			if ax != -1:
				atx = arr[ax-1]
				if (i - atx > 1):ans += ax
				else:ans += max(0,(ax-1)) 
				#print i , ax,to_,atx
		#print '->',ans
		if (k == 1):break
		if (k == -1 and j == 1):break
	#print '------',i

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
from collections import defaultdict
from bootcamp import Basebootcamp
import random

class Cmollyschemicalsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, min_abs_k=1, max_abs_k=10, array_min=-100, array_max=100):
        self.min_n = min_n
        self.max_n = max_n
        self.min_abs_k = max(1, min_abs_k)  # 确保最小绝对值为1
        self.max_abs_k = max_abs_k
        self.array_min = array_min
        self.array_max = array_max
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        # 强制k_abs的取值范围在有效区间
        k_abs = random.randint(self.min_abs_k, self.max_abs_k)
        k = k_abs if random.random() < 0.5 else -k_abs
        array = [random.randint(self.array_min, self.array_max) for _ in range(n)]
        correct_answer = self._calculate_solution(n, k, array)
        return {
            'n': n,
            'k': k,
            'array': array,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def _calculate_solution(n, k, array):
        pre = []
        current_sum = 0
        for num in array:
            current_sum += num
            pre.append(current_sum)
        ocr = defaultdict(list)
        for idx, s in enumerate(pre):
            ocr[s].append(idx)
        ans = 0
        INF = 10**14 + 10

        for i in range(n):
            at_ = pre[i]
            for j in range(0, 51):
                to_ = k ** j
                if k not in (1, -1) and abs(to_) > INF:
                    break
                # 处理单元素段
                if array[i] == to_:
                    ans += 1
                # 处理完整前缀段
                if i != 0 and at_ == to_:
                    ans += 1
                check_ = at_ - to_
                if check_ in ocr:
                    arr = ocr[check_]
                    ax = bisect.bisect_left(arr, i)
                    if ax > 0:
                        atx = arr[ax-1]
                        if (i - atx) > 1:
                            ans += ax
                        else:
                            ans += max(0, ax-1)
                # 处理k的特殊情况
                if k == 1:
                    break
                if k == -1 and j == 1:
                    break
        return ans
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        array = question_case['array']
        array_str = ' '.join(map(str, array))
        prompt = f"""Molly Hooper has {n} chemicals arranged in a line. Each chemical has an affection value. Find the number of contiguous segments where the total affection value is a non-negative integer power of {k}.

Input:
{n} {k}
{array_str}

Your answer must be a single integer placed between [answer] and [/answer] tags. Example: [answer]8[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
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

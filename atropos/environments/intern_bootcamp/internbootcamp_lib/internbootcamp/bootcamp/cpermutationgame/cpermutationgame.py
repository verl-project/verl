"""# 

### 谜题描述
After a long day, Alice and Bob decided to play a little game. The game board consists of n cells in a straight line, numbered from 1 to n, where each cell contains a number a_i between 1 and n. Furthermore, no two cells contain the same number. 

A token is placed in one of the cells. They take alternating turns of moving the token around the board, with Alice moving first. The current player can move from cell i to cell j only if the following two conditions are satisfied: 

  * the number in the new cell j must be strictly larger than the number in the old cell i (i.e. a_j > a_i), and 
  * the distance that the token travels during this turn must be a multiple of the number in the old cell (i.e. |i-j|mod a_i = 0). 



Whoever is unable to make a move, loses. For each possible starting position, determine who wins if they both play optimally. It can be shown that the game is always finite, i.e. there always is a winning strategy for one of the players.

Input

The first line contains a single integer n (1 ≤ n ≤ 10^5) — the number of numbers.

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ n). Furthermore, there are no pair of indices i ≠ j such that a_i = a_j.

Output

Print s — a string of n characters, where the i-th character represents the outcome of the game if the token is initially placed in the cell i. If Alice wins, then s_i has to be equal to \"A\"; otherwise, s_i has to be equal to \"B\". 

Examples

Input

8
3 6 5 4 2 7 1 8


Output

BAAAABAB


Input

15
3 11 2 5 10 9 7 13 15 8 4 12 6 1 14


Output

ABAAAABBBAABAAB

Note

In the first sample, if Bob puts the token on the number (not position): 

  * 1: Alice can move to any number. She can win by picking 7, from which Bob has no move. 
  * 2: Alice can move to 3 and 5. Upon moving to 5, Bob can win by moving to 8. If she chooses 3 instead, she wins, as Bob has only a move to 4, from which Alice can move to 8. 
  * 3: Alice can only move to 4, after which Bob wins by moving to 8. 
  * 4, 5, or 6: Alice wins by moving to 8. 
  * 7, 8: Alice has no move, and hence she loses immediately. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=int(raw_input())
arr=list(map(int,raw_input().split()))
dict1={}
arr1=[0]*n
for i in range(n):
	arr1[arr[i]-1]=i
for i in range(n):
	dict1[i+1]=[]
for i in range(n):
	for j in range(i-arr[i],-1,-arr[i]):
		if(arr[j]>arr[i]):
			dict1[arr[i]].append(arr[j])
	for j in range(i+arr[i],n,arr[i]):
		if(arr[j]>arr[i]):
			dict1[arr[i]].append(arr[j])
strarr=['.']*n
#print(dict1)
for i in range(n-1,-1,-1):
	if(len(dict1[arr[arr1[i]]])==0):
		strarr[arr1[i]]='B'
	else:
		if(len(dict1[arr[arr1[i]]])==1 and len(dict1[dict1[arr[arr1[i]]][0]])==0):
			strarr[arr1[i]]='A'
		else:
			flag=0
			for j in dict1[arr[arr1[i]]]:
				#print(j)
				#print(arr1[j-1])
				if(strarr[arr1[j-1]]=='B'):
					flag=1
					break
			if(flag==1):
				strarr[arr1[i]]='A'
			else:
				strarr[arr1[i]]='B'
	#print(*strarr)
print(\"\".join(x for x in strarr))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Cpermutationgamebootcamp(Basebootcamp):
    def __init__(self, n=8):
        self.n = n
    
    def case_generator(self):
        n = self.n
        a = list(range(1, n+1))
        random.shuffle(a)
        s = self.compute_s_optimized(n, a)
        return {
            'n': n,
            'a': a,
            's': s
        }
    
    @staticmethod
    def compute_s_optimized(n, arr):
        pos_map = {num: idx for idx, num in enumerate(arr)}
        moves = [[] for _ in range(n)]
        
        # 预处理合法移动（优化版本）
        for i in range(n):
            ai = arr[i]
            # 向左遍历（步长ai）
            for j in range(i - ai, -1, -ai):
                if arr[j] > ai:
                    moves[i].append(j)
            # 向右遍历（步长ai）
            for j in range(i + ai, n, ai):
                if arr[j] > ai:
                    moves[i].append(j)
        
        # 动态规划从后往前处理
        dp = ['B'] * n
        sorted_indices = sorted(range(n), key=lambda x: -arr[x])
        
        for idx in sorted_indices:
            for move in moves[idx]:
                if dp[move] == 'B':
                    dp[idx] = 'A'
                    break
        return ''.join(dp)
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        prompt = f"""Alice和Bob正在玩一个策略游戏。游戏规则如下：

- 棋盘包含{n}个单元格，按1到{n}编号。每个单元格有一个唯一数字（1到{n}之间）。
- 玩家轮流移动令牌，Alice先手。
- 移动规则：新位置的数字必须严格大于当前数字，且移动距离是当前数字的倍数。
- 无法移动的玩家输。

当前谜题的数组a为：[{', '.join(map(str, a))}]

请针对每个起始位置i（1到{n}），判断Alice获胜的情况。输出一个长度为{n}的字符串，其中第i个字符为'A'（Alice胜）或'B'（Bob胜）。

答案请放在[answer]标签内，例如：[answer]ABAB[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]([A-B]+)\[/answer\]', output, re.IGNORECASE)
        return matches[-1].upper().strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['s']
        return solution == expected if solution else False

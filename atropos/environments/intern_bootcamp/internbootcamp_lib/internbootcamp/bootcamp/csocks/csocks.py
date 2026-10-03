"""# 

### 谜题描述
Arseniy is already grown-up and independent. His mother decided to leave him alone for m days and left on a vacation. She have prepared a lot of food, left some money and washed all Arseniy's clothes. 

Ten minutes before her leave she realized that it would be also useful to prepare instruction of which particular clothes to wear on each of the days she will be absent. Arseniy's family is a bit weird so all the clothes is enumerated. For example, each of Arseniy's n socks is assigned a unique integer from 1 to n. Thus, the only thing his mother had to do was to write down two integers li and ri for each of the days — the indices of socks to wear on the day i (obviously, li stands for the left foot and ri for the right). Each sock is painted in one of k colors.

When mother already left Arseniy noticed that according to instruction he would wear the socks of different colors on some days. Of course, that is a terrible mistake cause by a rush. Arseniy is a smart boy, and, by some magical coincidence, he posses k jars with the paint — one for each of k colors.

Arseniy wants to repaint some of the socks in such a way, that for each of m days he can follow the mother's instructions and wear the socks of the same color. As he is going to be very busy these days he will have no time to change the colors of any socks so he has to finalize the colors now.

The new computer game Bota-3 was just realised and Arseniy can't wait to play it. What is the minimum number of socks that need their color to be changed in order to make it possible to follow mother's instructions and wear the socks of the same color during each of m days.

Input

The first line of input contains three integers n, m and k (2 ≤ n ≤ 200 000, 0 ≤ m ≤ 200 000, 1 ≤ k ≤ 200 000) — the number of socks, the number of days and the number of available colors respectively.

The second line contain n integers c1, c2, ..., cn (1 ≤ ci ≤ k) — current colors of Arseniy's socks.

Each of the following m lines contains two integers li and ri (1 ≤ li, ri ≤ n, li ≠ ri) — indices of socks which Arseniy should wear during the i-th day.

Output

Print one integer — the minimum number of socks that should have their colors changed in order to be able to obey the instructions and not make people laugh from watching the socks of different colors.

Examples

Input

3 2 3
1 2 3
1 2
2 3


Output

2


Input

3 2 2
1 1 2
1 2
2 1


Output

0

Note

In the first sample, Arseniy can repaint the first and the third socks to the second color.

In the second sample, there is no need to change any colors.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import heapq
import bisect

INF = 10**9+7
OFFLINE = 0
sys.setrecursionlimit(INF)

##
if OFFLINE:
	sys.stdin = open(\"fin.txt\", \"r\")
	sys.stdout = open(\"fout.txt\", \"w\")
	print \"OFFLINE\"
##

#main

seen = set()

def bfs(s):
	global seen

	if s in seen:
		return set()

	seen.add(s)

	S = set()
	S.add(s)

	level = 1
	frt = [s]

	while frt:
		nxt = []
		for u in frt:
			for v in G[u]:
				if v not in S:
					S.add(v)
					seen.add(v)
					nxt.append(v)
		level += 1
		frt = nxt

	return S



n, m, k = map(int, sys.stdin.readline().split())
color = [-1]
G = {}
line = raw_input()
line = line.split()
for u in range(1, n+1):
	color.append(int(line[u-1]))
	G[u] = [] 

for i in range(m):
	u, v = map(int, sys.stdin.readline().split())

	G[u].append(v)
	G[v].append(u)


sets = []

for u in G:
	s = bfs(u)
	if len(s) == 0:
		continue
	else:
		sets.append(s)

#print sets

_s = []
for s in sets:
	a = []
	for u in s:
		a.append(color[u])
	_s.append(a)


ans = 0

cnt = [0 for i in range(k+1)]
for a in _s:
	for c in a:
		cnt[c] += 1

	_max = 0
	for c in a:
		if cnt[c] > _max:
			_max = cnt[c]

	for c in a:
		cnt[c] = 0

	ans += (len(a) - _max)

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict, deque
from bootcamp import Basebootcamp
import re

class Csocksbootcamp(Basebootcamp):
    def __init__(self, min_socks=2, max_socks=5, min_days=0, max_days=20, min_colors=1, max_colors=5):
        self.min_socks = min_socks
        self.max_socks = max_socks
        self.min_days = min_days
        self.max_days = max_days
        self.min_colors = min_colors
        self.max_colors = max_colors
    
    def case_generator(self):
        n = random.randint(self.min_socks, self.max_socks)
        k = random.randint(self.min_colors, self.max_colors)
        socks = list(range(1, n + 1))
        random.shuffle(socks)
        
        max_groups = min(n, 3)
        num_groups = random.randint(1, max_groups)
        
        groups = []
        if num_groups == 1:
            groups = [socks]
        else:
            group_size = n // num_groups
            remaining = n % num_groups
            start = 0
            for i in range(num_groups):
                g_size = group_size + (1 if i < remaining else 0)
                groups.append(socks[start:start+g_size])
                start += g_size
        
        groups = [g for g in groups if g]
        pairs = []
        for group in groups:
            if len(group) >= 2:
                random.shuffle(group)
                for i in range(len(group)-1):
                    pairs.append([group[i], group[i+1]])
        
        m = len(pairs)
        colors = [0] * n
        for group in groups:
            group_colors = [random.randint(1, k) for _ in group]
            for idx, sock in enumerate(group):
                colors[sock-1] = group_colors[idx]
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'colors': colors,
            'pairs': pairs
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        colors = question_case['colors']
        pairs = question_case['pairs']
        prompt = f"""Arseniy needs to repaint socks so each day's pair has the same color. Find the minimum repaints needed.

Details:
- Socks: {n}
- Days: {m}
- Colors: {k}
- Current colors (socks 1-{n}): {colors}
- Daily pairs:
"""
        for i, (li, ri) in enumerate(pairs, 1):
            prompt += f"  Day {i}: Socks {li} and {ri}\n"
        prompt += "\nOutput the minimum repaints within [answer] and [/answer]."
        return prompt
    
    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answers:
            return None
        try:
            return int(answers[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        colors = identity['colors']
        pairs = identity['pairs']
        G = defaultdict(list)
        for li, ri in pairs:
            G[li].append(ri)
            G[ri].append(li)
        
        seen = set()
        total = 0
        for sock in range(1, n+1):
            if sock in seen:
                continue
            queue = deque([sock])
            seen.add(sock)
            component = []
            while queue:
                u = queue.popleft()
                component.append(u)
                for v in G[u]:
                    if v not in seen:
                        seen.add(v)
                        queue.append(v)
            color_count = defaultdict(int)
            for u in component:
                color_count[colors[u-1]] += 1
            max_c = max(color_count.values(), default=0)
            total += len(component) - max_c
        
        try:
            return int(solution) == total
        except:
            return False

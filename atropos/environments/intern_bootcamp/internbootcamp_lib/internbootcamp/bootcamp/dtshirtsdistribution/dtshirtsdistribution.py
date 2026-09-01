"""# 

### 谜题描述
The organizers of a programming contest have decided to present t-shirts to participants. There are six different t-shirts sizes in this problem: S, M, L, XL, XXL, XXXL (sizes are listed in increasing order). The t-shirts are already prepared. For each size from S to XXXL you are given the number of t-shirts of this size.

During the registration, the organizers asked each of the n participants about the t-shirt size he wants. If a participant hesitated between two sizes, he could specify two neighboring sizes — this means that any of these two sizes suits him.

Write a program that will determine whether it is possible to present a t-shirt to each participant of the competition, or not. Of course, each participant should get a t-shirt of proper size: 

  * the size he wanted, if he specified one size; 
  * any of the two neibouring sizes, if he specified two sizes. 



If it is possible, the program should find any valid distribution of the t-shirts.

Input

The first line of the input contains six non-negative integers — the number of t-shirts of each size. The numbers are given for the sizes S, M, L, XL, XXL, XXXL, respectively. The total number of t-shirts doesn't exceed 100 000.

The second line contains positive integer n (1 ≤ n ≤ 100 000) — the number of participants.

The following n lines contain the sizes specified by the participants, one line per participant. The i-th line contains information provided by the i-th participant: single size or two sizes separated by comma (without any spaces). If there are two sizes, the sizes are written in increasing order. It is guaranteed that two sizes separated by comma are neighboring.

Output

If it is not possible to present a t-shirt to each participant, print «NO» (without quotes).

Otherwise, print n + 1 lines. In the first line print «YES» (without quotes). In the following n lines print the t-shirt sizes the orginizers should give to participants, one per line. The order of the participants should be the same as in the input.

If there are multiple solutions, print any of them.

Examples

Input

0 1 0 1 1 0
3
XL
S,M
XL,XXL


Output

YES
XL
M
XXL


Input

1 1 2 0 1 1
5
S
M
S,M
XXL,XXXL
XL,XXL


Output

NO

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def read(t=None):
	string = raw_input()
	return string if t is None else [t(x) for x in string.split()]

a, n = [], 0
demands = []

sizeno = {'S':0, 'M':1, 'L':2, 'XL':3, 'XXL':4, 'XXXL':5}
sizenm = ['S', 'M', 'L', 'XL', 'XXL', 'XXXL']

class demand(object):
	def __init__(self, arr):
		self.one = True
		self.x = sizeno[arr[0]]
		self.y = -1
		self.sizenm = None

		if len(arr) == 2:
			self.one = False
			self.y = sizeno[arr[1]]

	def __str__(self):
		return \"(%s, %s)\"%(sizenm[self.x], -1 if self.y == -1 else sizenm[self.y])

def distribute(i):
	global a
	for d in demands:
		x, y, one = d.x, d.y, d.one
		if x != i or one:
			continue
		#print \"\t\tdistribut for: %s\"%d
		#print \"\t\ta: %s\"%a
		if a[x] > 0:
			a[x] -= 1
			d.sizenm = sizenm[x]
		elif a[y] > 0:
			a[y] -= 1
			d.sizenm = sizenm[y]
		else:
			#print \"\t\tFailed\"
			return False
		#print \"\t\tdistribute: %s: %s\"%(d, d.sizenm)

	return True

def solve():
	global a, n, demands

	a = read(int)
	n = int(read())

	for i in xrange(n):
		arr = read().split(',')
		#print \"arr: %s\"%arr
		demands.append(demand(arr))

	for d in demands:
		x, one = d.x, d.one
		if one:
			d.sizenm = sizenm[x]
			a[x] -= 1
	for x in a:
		if x < 0:
			print 'NO'
			return

	#print \"--->\"
	#for d in demands:
		#print d
	#print \"--->\"

	for i in range(5):
		res = distribute(i)
		if not res:
			print 'NO'
			return

	print 'YES'
	for d in demands:
		print d.sizenm

if __name__ == \"__main__\":
	solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Dtshirtsdistributionbootcamp(Basebootcamp):
    SIZES = ['S', 'M', 'L', 'XL', 'XXL', 'XXXL']
    SIZE_IDX = {s:i for i,s in enumerate(SIZES)}
    
    def __init__(self, **params):
        self.min_participants = params.get('min_participants', 1)
        self.max_participants = params.get('max_participants', 10)
        self.double_prob = params.get('double_prob', 0.3)
    
    def case_generator(self):
        """保证生成有解的实例"""
        n = random.randint(self.min_participants, self.max_participants)
        
        # 首先生成实际分配方案
        assignments = []
        for _ in range(n):
            size_idx = random.randint(0, 5)
            assignments.append(size_idx)
        
        # 计算库存
        inventory = [0]*6
        for idx in assignments:
            inventory[idx] += 1
        
        # 生成需求描述
        demands = []
        for idx in assignments:
            if random.random() < self.double_prob and idx < 5:
                # 生成相邻需求（确保至少包含实际尺寸）
                neighbor = random.choice([idx, idx+1])
                min_idx = min(idx, neighbor)
                max_idx = max(idx, neighbor)
                demands.append(f"{self.SIZES[min_idx]},{self.SIZES[max_idx]}")
            else:
                demands.append(self.SIZES[idx])
        
        return {
            'available': inventory,
            'n': n,
            'demands': demands
        }
    
    @staticmethod
    def prompt_func(question_case):
        sizes = ' '.join(map(str, question_case['available']))
        demands = '\n'.join(question_case['demands'])
        return f"""编程竞赛T恤分配问题：
可用尺寸库存（S M L XL XXL XXXL）：{sizes}
参与者数量：{question_case['n']}
需求列表：
{demands}

请判断是否可以满足所有需求，并将答案包裹在[answer]标签中。例如：
[answer]
YES
XL
M
XXL
[/answer]
或：
[answer]
NO
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        solution = solution.strip().upper().split('\n')
        if not solution:
            return False
        
        # 验证NO情况
        if solution[0] == 'NO':
            try:
                # 调用参考解法验证是否确实无解
                from copy import deepcopy
                from io import StringIO
                import sys
                
                class Demand:
                    def __init__(self, parts):
                        self.x = cls.SIZE_IDX[parts[0]]
                        self.y = cls.SIZE_IDX[parts[1]] if len(parts)>1 else -1
                        self.assigned = None
                
                # 构建输入数据
                a = deepcopy(identity['available'])
                demands = [Demand(d.split(',')) for d in identity['demands']]
                
                # 处理单一需求
                valid = True
                for d in demands:
                    if d.y == -1:
                        a[d.x] -= 1
                        if a[d.x] < 0:
                            valid = False
                
                if not valid:
                    return True
                
                # 处理双需求
                for i in range(5):
                    for d in demands:
                        if d.y != -1 and d.x == i and d.assigned is None:
                            if a[i] > 0:
                                a[i] -= 1
                                d.assigned = i
                            elif a[i+1] > 0:
                                a[i+1] -= 1
                                d.assigned = i+1
                            else:
                                valid = False
                    if not valid:
                        break
                
                return not valid
            except:
                return False
        
        # 验证YES情况
        if solution[0] != 'YES' or len(solution) != identity['n']+1:
            return False
        
        # 检查格式有效性
        try:
            assigned = [cls.SIZE_IDX[s.upper()] for s in solution[1:]]
        except KeyError:
            return False
        
        # 检查库存消耗
        inventory = list(identity['available'])
        for idx in assigned:
            inventory[idx] -= 1
            if inventory[idx] < 0:
                return False
        
        # 检查需求匹配
        for i, (size_idx, demand) in enumerate(zip(assigned, identity['demands'])):
            allowed = [cls.SIZE_IDX[p] for p in demand.split(',')]
            if size_idx not in allowed:
                return False
        
        return True

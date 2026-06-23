"""# 

### 谜题描述
Little beaver is a beginner programmer, so informatics is his favorite subject. Soon his informatics teacher is going to have a birthday and the beaver has decided to prepare a present for her. He planted n flowers in a row on his windowsill and started waiting for them to grow. However, after some time the beaver noticed that the flowers stopped growing. The beaver thinks it is bad manners to present little flowers. So he decided to come up with some solutions. 

There are m days left to the birthday. The height of the i-th flower (assume that the flowers in the row are numbered from 1 to n from left to right) is equal to ai at the moment. At each of the remaining m days the beaver can take a special watering and water w contiguous flowers (he can do that only once at a day). At that each watered flower grows by one height unit on that day. The beaver wants the height of the smallest flower be as large as possible in the end. What maximum height of the smallest flower can he get?

Input

The first line contains space-separated integers n, m and w (1 ≤ w ≤ n ≤ 105; 1 ≤ m ≤ 105). The second line contains space-separated integers a1, a2, ..., an (1 ≤ ai ≤ 109).

Output

Print a single integer — the maximum final height of the smallest flower.

Examples

Input

6 2 3
2 2 2 2 1 1


Output

2


Input

2 5 1
5 8


Output

9

Note

In the first sample beaver can water the last 3 flowers at the first day. On the next day he may not to water flowers at all. In the end he will get the following heights: [2, 2, 2, 3, 2, 2]. The smallest flower has height equal to 2. It's impossible to get height 3 in this test.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
n,want,lim = map(int,sys.stdin.readline().split())
a = map(int,sys.stdin.readline().split())
cursum = [0]*200005
def check(m):
	temp=a
	day=0
	cur=0
	for i in xrange(n):
		if(a[i]+cur<m):
			cursum[i]+=m-(a[i]+cur)
			day+=m-(a[i]+cur)
			cursum[i+lim-1]+=-(m-(a[i]+cur))
		cur+=cursum[i]
		cursum[i]=0
	if(day>want):return 0
	return 1
def local():
	f=0
	l=1000100005
	m=0
	while(f<l):
		m=(f+l)/2
		if(check(m)==1):f=m+1
		else: l=m
	print f-1
local()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def compute_max_min_height(n, m, w, a_list):
    low = min(a_list)
    high = low + m
    best = low

    while low <= high:
        mid = (low + high) // 2
        required = 0
        current_add = 0
        diff = [0] * (n + 2)
        valid = True

        for i in range(n):
            current_add += diff[i]
            current_height = a_list[i] + current_add

            if current_height < mid:
                need = mid - current_height
                required += need
                if required > m:
                    valid = False
                    break

                current_add += need
                end = i + w
                # 精确处理浇水范围
                effective_end = min(end, n)
                diff[effective_end] -= need

        if valid and required <= m:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best

class Cpresentbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=100, max_a=10):
        self.max_n = max(3, max_n)  # 确保生成足够复杂的案例
        self.max_m = max(2, max_m)
        self.max_a = max_a
    
    def case_generator(self):
        while True:
            n = random.randint(1, self.max_n)
            w = random.randint(1, n) if n > 0 else 1
            m = random.randint(0, self.max_m)  # 包含m=0边界情况
            a = [random.randint(1, self.max_a) for _ in range(n)]
            
            # 确保至少有一个非平凡解
            if m == 0:
                correct = min(a)
            else:
                correct = compute_max_min_height(n, m, w, a)
            
            # 强制20%案例包含等差数列
            if random.random() < 0.2:
                a = sorted([correct + i for i in range(n)])
            
            return {
                'n': n,
                'm': m,
                'w': w,
                'a': a,
                'correct_answer': correct
            }
    
    @staticmethod
    def prompt_func(case):
        return (
            f"Problem Statement:\n"
            f"Given {case['n']} flowers arranged in a row with initial heights {case['a']}, "
            f"you have {case['m']} days to water them. Each day you can choose a contiguous "
            f"block of {case['w']} flowers to water (each watered flower grows by 1 unit).\n\n"
            f"Task: Determine the maximum possible height of the shortest flower after optimal watering.\n\n"
            f"Output Format: Put your final integer answer within [answer] tags like: [answer]42[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\s*\](.*?)\[\/answer\s*\]', output, re.DOTALL|re.IGNORECASE)
        if matches:
            last_match = matches[-1].strip()
            if last_match.isdigit():
                return int(last_match)
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 针对边界case重新计算验证
        recalculated = compute_max_min_height(
            identity['n'],
            identity['m'],
            identity['w'],
            identity['a']
        )
        return solution == recalculated

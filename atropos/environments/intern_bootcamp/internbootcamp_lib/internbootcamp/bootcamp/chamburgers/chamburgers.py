"""# 

### 谜题描述
Polycarpus loves hamburgers very much. He especially adores the hamburgers he makes with his own hands. Polycarpus thinks that there are only three decent ingredients to make hamburgers from: a bread, sausage and cheese. He writes down the recipe of his favorite \"Le Hamburger de Polycarpus\" as a string of letters 'B' (bread), 'S' (sausage) и 'C' (cheese). The ingredients in the recipe go from bottom to top, for example, recipe \"ВSCBS\" represents the hamburger where the ingredients go from bottom to top as bread, sausage, cheese, bread and sausage again.

Polycarpus has nb pieces of bread, ns pieces of sausage and nc pieces of cheese in the kitchen. Besides, the shop nearby has all three ingredients, the prices are pb rubles for a piece of bread, ps for a piece of sausage and pc for a piece of cheese.

Polycarpus has r rubles and he is ready to shop on them. What maximum number of hamburgers can he cook? You can assume that Polycarpus cannot break or slice any of the pieces of bread, sausage or cheese. Besides, the shop has an unlimited number of pieces of each ingredient.

Input

The first line of the input contains a non-empty string that describes the recipe of \"Le Hamburger de Polycarpus\". The length of the string doesn't exceed 100, the string contains only letters 'B' (uppercase English B), 'S' (uppercase English S) and 'C' (uppercase English C).

The second line contains three integers nb, ns, nc (1 ≤ nb, ns, nc ≤ 100) — the number of the pieces of bread, sausage and cheese on Polycarpus' kitchen. The third line contains three integers pb, ps, pc (1 ≤ pb, ps, pc ≤ 100) — the price of one piece of bread, sausage and cheese in the shop. Finally, the fourth line contains integer r (1 ≤ r ≤ 1012) — the number of rubles Polycarpus has.

Please, do not write the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

Print the maximum number of hamburgers Polycarpus can make. If he can't make any hamburger, print 0.

Examples

Input

BBBSSC
6 4 1
1 2 3
4


Output

2


Input

BBC
1 10 1
1 10 1
21


Output

7


Input

BSC
1 1 1
1 1 3
1000000000000


Output

200000000001

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
linea=raw_input()
nb,ns,nc=map(int,raw_input().split())
pb,ps,pc=map(int,raw_input().split())
r=int(raw_input())

cb=0
cs=0
cc=0

for char in linea:
	if char=='B':
		cb+=1
	elif char=='S':
		cs+=1
	else:
		cc+=1

def check(n):
	rb=n*cb-nb
	rs=n*cs-ns
	rc=n*cc-nc
	if rb<0:
		rb=0
	if rs<0:
		rs=0
	if rc<0:
		rc=0
	return r>=(rb*pb+rs*ps+rc*pc)

def binarySearch(l,r):
	if (l==r):
		return l
	mid=(l+r+1)/2
	if (check(mid)):
		return binarySearch(mid,r)
	else:
		return binarySearch(l,mid-1)

print binarySearch(0,100000000000000000)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def calculate_max_hamburgers(recipe, nb, ns, nc, pb, ps, pc, r):
    cb = recipe.count('B')
    cs = recipe.count('S')
    cc = recipe.count('C')
    
    def check(n):
        rb = max(0, cb * n - nb)
        rs = max(0, cs * n - ns)
        rc = max(0, cc * n - nc)
        cost = rb * pb + rs * ps + rc * pc
        return cost <= r
    
    low = 0
    high = 10**18
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    return ans

class Chamburgersbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)  # 添加基类初始化调用
        self.params = params
    
    def case_generator(self):
        def generate_recipe():
            length = random.randint(1, 100)
            chars = ['B', 'S', 'C']
            # 确保食谱至少包含一种必要成分
            while True:
                recipe = ''.join(random.choices(chars, k=length))
                if any(c in recipe for c in ['B', 'S', 'C']):
                    return recipe
        
        recipe = generate_recipe()
        nb = random.randint(1, 100)
        ns = random.randint(1, 100)
        nc = random.randint(1, 100)
        pb = random.randint(1, 100)
        ps = random.randint(1, 100)
        pc = random.randint(1, 100)
        r = random.randint(1, 10**12)
        
        return {
            'recipe': recipe,
            'nb': nb,
            'ns': ns,
            'nc': nc,
            'pb': pb,
            'ps': ps,
            'pc': pc,
            'r': r
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        recipe = question_case['recipe']
        nb = question_case['nb']
        ns = question_case['ns']
        nc = question_case['nc']
        pb = question_case['pb']
        ps = question_case['ps']
        pc = question_case['pc']
        r = question_case['r']
        
        prompt = f"""You are a chef assistant. Polycarpus loves making hamburgers following specific recipes. Each hamburger's ingredients are layered bottom to top as per the recipe string consisting of 'B' (bread), 'S' (sausage), and 'C' (cheese).

The recipe for the hamburger is: {recipe}.

Polycarpus currently has:
- {nb} pieces of bread (B),
- {ns} pieces of sausage (S),
- {nc} pieces of cheese (C).

The shop sells each ingredient at the following prices:
- Bread: {pb} rubles per piece,
- Sausage: {ps} rubles per piece,
- Cheese: {pc} rubles per piece.

He has {r} rubles to spend. 

What's the maximum number of hamburgers he can make? You cannot break any pieces; you can buy additional ingredients if needed. 

Please provide your answer as an integer inside [answer] and [/answer] tags. For example, [answer]5[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            solution_num = int(solution)
            correct = calculate_max_hamburgers(
                identity['recipe'],
                identity['nb'],
                identity['ns'],
                identity['nc'],
                identity['pb'],
                identity['ps'],
                identity['pc'],
                identity['r']
            )
            return solution_num == correct
        except (ValueError, KeyError):
            return False

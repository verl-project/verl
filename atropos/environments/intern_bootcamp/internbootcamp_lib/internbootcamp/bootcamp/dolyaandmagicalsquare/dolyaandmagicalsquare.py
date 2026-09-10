"""# 

### 谜题描述
Recently, Olya received a magical square with the size of 2^n× 2^n.

It seems to her sister that one square is boring. Therefore, she asked Olya to perform exactly k splitting operations.

A Splitting operation is an operation during which Olya takes a square with side a and cuts it into 4 equal squares with side a/2. If the side of the square is equal to 1, then it is impossible to apply a splitting operation to it (see examples for better understanding).

Olya is happy to fulfill her sister's request, but she also wants the condition of Olya's happiness to be satisfied after all operations.

The condition of Olya's happiness will be satisfied if the following statement is fulfilled:

Let the length of the side of the lower left square be equal to a, then the length of the side of the right upper square should also be equal to a. There should also be a path between them that consists only of squares with the side of length a. All consecutive squares on a path should have a common side.

Obviously, as long as we have one square, these conditions are met. So Olya is ready to fulfill her sister's request only under the condition that she is satisfied too. Tell her: is it possible to perform exactly k splitting operations in a certain order so that the condition of Olya's happiness is satisfied? If it is possible, tell also the size of the side of squares of which the path from the lower left square to the upper right one will consist.

Input

The first line contains one integer t (1 ≤ t ≤ 10^3) — the number of tests.

Each of the following t lines contains two integers n_i and k_i (1 ≤ n_i ≤ 10^9, 1 ≤ k_i ≤ 10^{18}) — the description of the i-th test, which means that initially Olya's square has size of 2^{n_i}× 2^{n_i} and Olya's sister asks her to do exactly k_i splitting operations.

Output

Print t lines, where in the i-th line you should output \"YES\" if it is possible to perform k_i splitting operations in the i-th test in such a way that the condition of Olya's happiness is satisfied or print \"NO\" otherwise. If you printed \"YES\", then also print the log_2 of the length of the side of the squares through space, along which you can build a path from the lower left square to the upper right one.

You can output each letter in any case (lower or upper).

If there are multiple answers, print any.

Example

Input


3
1 1
2 2
2 12


Output


YES 0
YES 1
NO

Note

In each of the illustrations, the pictures are shown in order in which Olya applied the operations. The recently-created squares are highlighted with red.

In the first test, Olya can apply splitting operations in the following order:

<image> Olya applies one operation on the only existing square.

The condition of Olya's happiness will be met, since there is a path of squares of the same size from the lower left square to the upper right one:

<image>

The length of the sides of the squares on the path is 1. log_2(1) = 0.

In the second test, Olya can apply splitting operations in the following order:

<image> Olya applies the first operation on the only existing square. She applies the second one on the right bottom square.

The condition of Olya's happiness will be met, since there is a path of squares of the same size from the lower left square to the upper right one:

<image>

The length of the sides of the squares on the path is 2. log_2(2) = 1.

In the third test, it takes 5 operations for Olya to make the square look like this:

<image>

Since it requires her to perform 7 splitting operations, and it is impossible to perform them on squares with side equal to 1, then Olya cannot do anything more and the answer is \"NO\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
t = int(raw_input())

f = [0]
for __ in xrange(100):
	f.append(f[-1] * 4 + 1)

p = [0]
for g in xrange(100):
	p.append(p[-1] + (pow(2,g+1)-1))

for __ in xrange(t):
	n,k = map(int, raw_input().split())
	if k == 1:
		print \"YES\", n-1
		continue

	if n < len(f) and k > f[n]:
		print \"NO\"
		continue
	on = n

	n = min(n, len(f))
	for j in xrange(n-1,-1,-1):
		if p[n-j] > k:
			continue

		other = pow(2, n-j)
		avail = (other - 1) * (other - 1) * f[j]
		if f[n-j] + avail >= k:
			print \"YES\", on - (n - j)
			break
	else:
		print \"NO\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def solve_case(n_input, k_input):
    MAX_PRECOMPUTE = 100
    f = [0]
    for _ in range(MAX_PRECOMPUTE):
        f.append(f[-1] * 4 + 1)
    p = [0]
    for g in range(MAX_PRECOMPUTE):
        p.append(p[-1] + (2 ** (g + 1) - 1))
    
    n, k = n_input, k_input

    if k == 1:
        return f"YES {n-1}"
    
    # 计算最大可能的分割次数（不考虑路径条件）
    max_f = (4**n - 1) // 3
    if k > max_f:
        return "NO"
    
    original_n = n
    
    # 直接遍历所有可能的j（不截断n）
    for j in range(original_n - 1, -1, -1):
        m_segment = original_n - j
        
        # 计算当前段的p值
        if m_segment < len(p):
            current_p = p[m_segment]
        else:
            current_p = 2 * (2**m_segment - 1) - m_segment
        
        if current_p > k:
            continue
        
        # 计算剩余可用分割次数
        other = 2 ** m_segment
        if j < len(f):
            f_j = f[j]
        else:
            f_j = (4**j - 1) // 3
        
        avail = (other - 1) ** 2 * f_j
        
        # 判断是否满足总分割次数
        if current_p + avail >= k:
            answer_m = original_n - m_segment
            return f"YES {answer_m}"
    
    return "NO"

class Dolyaandmagicalsquarebootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = params
    
    def case_generator(self):
        # 生成平衡的测试案例，涵盖边界情况
        if random.random() < 0.5:
            # 生成有效案例（有解）
            n = random.randint(1, 20)
            max_f = (4**n - 1) // 3
            if max_f == 0:  # 防止n=0的情况
                n = 1
                max_f = 1
            k = random.randint(1, max_f)
        else:
            # 生成无效案例（无解）
            n = random.randint(1, 15)
            max_f = (4**n - 1) // 3
            k = random.randint(max_f + 1, max_f * 2)
        return {'n': n, 'k': k}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        return f"""给定初始2^{n}×2^{n}的正方形，请判断是否可以进行恰好{k}次分裂操作，使得存在一条从左下到右上的同尺寸方块路径。答案格式：[answer]YES x[/answer]或[answer]NO[/answer]，其中x为log2(边长)。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL | re.IGNORECASE)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = solve_case(identity['n'], identity['k'])
        processed_sol = ' '.join(solution.strip().split()).upper()
        processed_exp = ' '.join(expected.strip().split()).upper()
        return processed_sol == processed_exp

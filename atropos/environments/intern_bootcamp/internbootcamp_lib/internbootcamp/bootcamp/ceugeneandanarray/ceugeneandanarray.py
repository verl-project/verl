"""# 

### 谜题描述
Eugene likes working with arrays. And today he needs your help in solving one challenging task.

An array c is a subarray of an array b if c can be obtained from b by deletion of several (possibly, zero or all) elements from the beginning and several (possibly, zero or all) elements from the end.

Let's call a nonempty array good if for every nonempty subarray of this array, sum of the elements of this subarray is nonzero. For example, array [-1, 2, -3] is good, as all arrays [-1], [-1, 2], [-1, 2, -3], [2], [2, -3], [-3] have nonzero sums of elements. However, array [-1, 2, -1, -3] isn't good, as his subarray [-1, 2, -1] has sum of elements equal to 0.

Help Eugene to calculate the number of nonempty good subarrays of a given array a.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 2 × 10^5) — the length of array a.

The second line of the input contains n integers a_1, a_2, ..., a_n (-10^9 ≤ a_i ≤ 10^9) — the elements of a. 

Output

Output a single integer — the number of good subarrays of a.

Examples

Input


3
1 2 -3


Output


5


Input


3
41 -41 41


Output


3

Note

In the first sample, the following subarrays are good: [1], [1, 2], [2], [2, -3], [-3]. However, the subarray [1, 2, -3] isn't good, as its subarray [1, 2, -3] has sum of elements equal to 0.

In the second sample, three subarrays of size 1 are the only good subarrays. At the same time, the subarray [41, -41, 41] isn't good, as its subarray [41, -41] has sum of elements equal to 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
a = map(int, raw_input().split())
p = [0 for _ in xrange(n + 1)]
for i in xrange(1, n + 1):
	p[i] = p[i - 1] + a[i - 1]

L = 0
R = 0

st = set()
ans = 0

while L < n:
	while R <= n and p[R] not in st:
		st.add(p[R])
		R += 1
	ans += R - L - 1
	st.remove(p[L])
	L += 1

print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from bootcamp import Basebootcamp

class Ceugeneandanarraybootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.n = params.get('n', 3)  # 默认数组长度为3
    
    def case_generator(self):
        # 随机生成数组元素，范围在-10^9到10^9之间
        n = self.n
        a = [random.randint(-10**9, 10**9) for _ in range(n)]
        
        # 计算正确答案
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + a[i - 1]
        
        L = 0
        R = 0
        seen = set()
        ans = 0
        
        while L < n:
            while R <= n and prefix[R] not in seen:
                seen.add(prefix[R])
                R += 1
            ans += R - L - 1
            seen.remove(prefix[L])
            L += 1
        
        # 返回可JSON序列化的字典
        case = {
            'n': n,
            'array': a,
            'correct_answer': ans
        }
        return case
    
    @staticmethod
    def prompt_func(question_case):
        array = question_case['array']
        n = question_case['n']
        prompt = (
            f"You are given an array of integers. Your task is to calculate the number of nonempty good subarrays of the given array. A good subarray is defined as one where every nonempty subarray of it has a nonzero sum. The array has length {n} and its elements are {array}. Please output the number of good subarrays in the following format: [answer]X[/answer], where X is the count."
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        start_tag = "[answer]"
        end_tag = "[/answer]"
        start = output.rfind(start_tag)
        if start == -1:
            return None
        end = output.find(end_tag, start + len(start_tag))
        if end == -1:
            return None
        answer_str = output[start + len(start_tag):end].strip()
        try:
            return int(answer_str)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        correct_answer = identity['correct_answer']
        return solution == correct_answer

# 示例使用
if __name__ == "__main__":
    # 初始化训练场，默认n=3
    bootcamp = Ceugeneandanarraybootcamp()
    
    # 生成谜题实例
    case = bootcamp.case_generator()
    print("Generated Case:", case)
    
    # 生成问题提示
    prompt = Ceugeneandanarraybootcamp.prompt_func(case)
    print("Prompt:", prompt)
    
    # 模拟LLM的输出
    model_output = f"The number of good subarrays is [answer]{case['correct_answer']}[/answer]."
    
    # 提取答案
    extracted_answer = Ceugeneandanarraybootcamp.extract_output(model_output)
    print("Extracted Answer:", extracted_answer)
    
    # 验证答案
    correct = Ceugeneandanarraybootcamp._verify_correction(extracted_answer, case)
    print("Correct:", correct)

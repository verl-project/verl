"""# 

### 谜题描述
Let's call the following process a transformation of a sequence of length n.

If the sequence is empty, the process ends. Otherwise, append the [greatest common divisor](https://en.wikipedia.org/wiki/Greatest_common_divisor) (GCD) of all the elements of the sequence to the result and remove one arbitrary element from the sequence. Thus, when the process ends, we have a sequence of n integers: the greatest common divisors of all the elements in the sequence before each deletion.

You are given an integer sequence 1, 2, ..., n. Find the lexicographically maximum result of its transformation.

A sequence a_1, a_2, …, a_n is lexicographically larger than a sequence b_1, b_2, …, b_n, if there is an index i such that a_j = b_j for all j < i, and a_i > b_i.

Input

The first and only line of input contains one integer n (1≤ n≤ 10^6).

Output

Output n integers — the lexicographically maximum result of the transformation.

Examples

Input

3


Output

1 1 3 

Input

2


Output

1 2 

Input

1


Output

1 

Note

In the first sample the answer may be achieved this way:

  * Append GCD(1, 2, 3) = 1, remove 2. 
  * Append GCD(1, 3) = 1, remove 1. 
  * Append GCD(3) = 3, remove 3. 



We get the sequence [1, 1, 3] as the result.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys


def solve(n):
    if n == 1:
        return [1]
    elif n == 0:
        return []
    elif n == 3:
        return [1, 1, 3]

    res = [1 for _ in range(0, (n + 1) / 2)]
    res2 = solve(n - (n + 1) / 2)
    return res + map(lambda a: a * 2, res2)

def gcd(a, b):
    if a > b:
        return gcd(b, a)
    if a == 0:
        return b
    return gcd(b % a, a)

def read_ints():
    return map(lambda a: int(a), sys.stdin.readline().split())

n, = read_ints()
res = solve(n)
print(\" \".join(map(lambda a: str(a), res)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from bootcamp import Basebootcamp

class Csequencetransformationbootcamp(Basebootcamp):
    def __init__(self, n=None, min_n=1, max_n=10):
        """
        初始化谜题参数，可指定固定的n，或随机生成n的范围。
        
        参数:
            n (int, optional): 固定的n值。若提供，则生成的实例均为该n。
            min_n (int): 随机生成n时的最小值。
            max_n (int): 随机生成n时的最大值。
        """
        self.n = n
        self.min_n = min_n
        self.max_n = max_n

    def case_generator(self):
        """
        生成谜题实例，返回包含n值的字典。
        """
        import random
        if self.n is not None:
            return {'n': self.n}
        else:
            n = random.randint(self.min_n, self.max_n)
            return {'n': n}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        将谜题实例转换为详细的文本问题，指导用户按指定格式回答。
        """
        n = question_case['n']
        prompt = f"""你是一个编程竞赛选手，需要解决以下问题：

给定一个初始序列1, 2, ..., {n}，通过以下过程生成结果序列：
1. 计算当前序列的GCD并添加到结果末尾。
2. 删除序列中的一个元素。
重复上述步骤直到序列为空，要求找到字典序最大的结果序列。

输入：n = {n}
输出：{n}个由空格分隔的整数，表示字典序最大的结果。

示例：
当n=3时，正确输出为：1 1 3，过程如下：
1. GCD(1,2,3)=1，删除2 → 剩余[1,3]
2. GCD(1,3)=1，删除1 → 剩余[3]
3. GCD(3)=3，删除3 → 结果[1,1,3]

请将答案放在[answer]标签内，例如：[answer]1 1 3[/answer]。
你的任务是解决n={n}的情况，并将最终答案按指定格式放置。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取最后一个[answer]标签内的内容，并标准化格式。
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        normalized = ' '.join(last_match.split())
        return normalized
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证用户答案是否与正确结果一致。
        """
        n = identity['n']
        try:
            correct = cls.solve(n)
            correct_str = ' '.join(map(str, correct))
            user_str = solution.strip()
            user_str = ' '.join(user_str.split())
            return user_str == correct_str
        except Exception:
            return False
    
    @staticmethod
    def solve(n):
        """
        递归生成字典序最大的结果序列。
        """
        if n == 1:
            return [1]
        elif n == 0:
            return []
        elif n == 3:
            return [1, 1, 3]
        res_length = (n + 1) // 2
        res = [1] * res_length
        remaining = n - res_length
        res2 = Csequencetransformationbootcamp.solve(remaining)
        res2_doubled = [x * 2 for x in res2]
        return res + res2_doubled

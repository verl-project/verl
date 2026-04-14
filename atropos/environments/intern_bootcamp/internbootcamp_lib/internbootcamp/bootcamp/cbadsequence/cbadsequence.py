"""# 

### 谜题描述
Petya's friends made him a birthday present — a bracket sequence. Petya was quite disappointed with his gift, because he dreamed of correct bracket sequence, yet he told his friends nothing about his dreams and decided to fix present himself. 

To make everything right, Petya is going to move at most one bracket from its original place in the sequence to any other position. Reversing the bracket (e.g. turning \"(\" into \")\" or vice versa) isn't allowed. 

We remind that bracket sequence s is called correct if: 

  * s is empty; 
  * s is equal to \"(t)\", where t is correct bracket sequence; 
  * s is equal to t_1 t_2, i.e. concatenation of t_1 and t_2, where t_1 and t_2 are correct bracket sequences. 



For example, \"(()())\", \"()\" are correct, while \")(\" and \"())\" are not. Help Petya to fix his birthday present and understand whether he can move one bracket so that the sequence becomes correct.

Input

First of line of input contains a single number n (1 ≤ n ≤ 200 000) — length of the sequence which Petya received for his birthday.

Second line of the input contains bracket sequence of length n, containing symbols \"(\" and \")\".

Output

Print \"Yes\" if Petya can make his sequence correct moving at most one bracket. Otherwise print \"No\".

Examples

Input


2
)(


Output


Yes


Input


3
(()


Output


No


Input


2
()


Output


Yes


Input


10
)))))(((((


Output


No

Note

In the first example, Petya can move first bracket to the end, thus turning the sequence into \"()\", which is correct bracket sequence.

In the second example, there is no way to move at most one bracket so that the sequence becomes correct.

In the third example, the sequence is already correct and there's no need to move brackets.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
'''input
8
)(()())(


'''

from bisect import bisect_right as bl
from random import randint as R
RI = lambda : [int(_x) for _x in raw_input().split()]

n=input() 
s = raw_input().strip()
n1 = s.count(\"(\")
go = (n1 == n-n1)
c = 0

count = 0
for i in s:
	if i==\"(\":
		c+=1
	if i==\")\":
		c-=1


	if c == -1:
		count+=1
		c = 0

	elif c <-1:
		count+=2


if count<=1 and go:
	print \"Yes\"
else:
	print \"No\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cbadsequencebootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化训练场类，设置默认参数
        """
        self.params = {
            'n': 100  # 默认生成的括号序列长度为100
        }
        self.params.update(params)
        # 确保n为偶数
        self.params['n'] = self.params['n'] if self.params['n'] % 2 == 0 else self.params['n'] - 1
    
    def case_generator(self):
        """
        生成一个括号序列的问题实例，确保可以通过移动最多一个括号使其正确
        """
        n = self.params['n']
        k = n // 2
        
        # 确保生成的括号数量是平衡的
        while True:
            s = []
            open_count = 0
            close_count = 0
            # 生成一个随机的括号序列
            for _ in range(n):
                if random.random() < 0.5:
                    s.append('(')
                    open_count += 1
                else:
                    s.append(')')
                    close_count += 1
            # 确保括号数量平衡
            if open_count == close_count:
                break
        
        s = ''.join(s)
        
        return {
            "n": n,
            "s": s
        }
    
    @staticmethod
    def prompt_func(question_case):
        """
        将问题实例转换为文本形式的问题描述
        """
        n = question_case['n']
        s = question_case['s']
        prompt = f"括号序列长度为{n}，序列为：{s}\n\nPetya可以移动最多一个括号，使得序列变成正确的括号序列吗？\n\n正确的括号序列的定义是：\n1. 空序列；\n2. 形如(t)，其中t是正确的序列；\n3. 形如t1t2，其中t1和t2都是正确的序列。\n\n你的任务是判断是否可以通过移动最多一个括号使得序列正确。如果是，输出'Yes'，否则输出'No'。请将答案放在[answer]标签中。例如：\n[answer]Yes[/answer] 或者 [answer]No[/answer]"
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        从LLM的回复中提取答案
        """
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if matches:
            return matches[-1].strip().lower()
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案是否正确
        """
        s = identity['s']
        n = identity['n']
        
        # 括号数量不平衡，直接返回False
        left = s.count('(')
        right = s.count(')')
        if left != right:
            return False
        
        # 计算需要调整的最少移动次数
        current_depth = 0
        required_moves = 0
        max_depth = 0
        
        for char in s:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            else:
                current_depth -= 1
                if current_depth < 0:
                    # 需要一个移动来纠正这种情况
                    required_moves += 1
                    # 假设我们移动了一个括号，深度回到0
                    current_depth = 0
        
        # 最大深度超过1的情况需要额外的移动次数
        if max_depth > 1:
            required_moves += 1
        
        # 是否可以通过最多一次移动解决
        expected = "Yes" if required_moves <= 1 else "No"
        return solution.lower() == expected.lower()

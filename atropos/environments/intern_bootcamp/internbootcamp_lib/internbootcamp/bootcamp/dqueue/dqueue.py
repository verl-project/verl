"""# 

### 谜题描述
Little girl Susie went shopping with her mom and she wondered how to improve service quality. 

There are n people in the queue. For each person we know time ti needed to serve him. A person will be disappointed if the time he waits is more than the time needed to serve him. The time a person waits is the total time when all the people who stand in the queue in front of him are served. Susie thought that if we swap some people in the queue, then we can decrease the number of people who are disappointed. 

Help Susie find out what is the maximum number of not disappointed people can be achieved by swapping people in the queue.

Input

The first line contains integer n (1 ≤ n ≤ 105).

The next line contains n integers ti (1 ≤ ti ≤ 109), separated by spaces.

Output

Print a single number — the maximum number of not disappointed people in the queue.

Examples

Input

5
15 2 1 5 3


Output

4

Note

Value 4 is achieved at such an arrangement, for example: 1, 2, 3, 5, 15. Thus, you can make everything feel not disappointed except for the person with time 5.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
t = map(int,raw_input().split())
t.sort()
s = t[0]
o = 1
for x in range(len(t)-1):
    if s <= t[x+1]:
	s+=t[x+1]
	o+=1
print o
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dqueuebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, ti_min=1, ti_max=1000):
        super().__init__()
        self.min_n = min_n
        self.max_n = max_n
        self.ti_min = ti_min
        self.ti_max = ti_max  # Adjusted default to avoid range errors
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        ti = [random.randint(self.ti_min, self.ti_max) for _ in range(n)]
        return {'n': n, 'ti': ti}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        ti_str = ' '.join(map(str, question_case['ti']))
        prompt = f"""你是一位排队优化专家，请帮助Susie找出调整队列顺序后可以得到的最大不失望人数。规则如下：

队列中有n个人，每个人有一个服务时间ti。当一个人等待的时间（即他前面所有人的服务时间总和）超过他的ti时，他会感到失望。我们的目标是通过调整队列顺序，使得不失望的人数最大化。请根据下面的输入数据，计算正确的答案，并按照指定格式返回结果。

输入格式：
第一行是一个整数n，表示人数。
第二行包含n个整数，表示每个人的服务时间ti，用空格分隔。

请仔细阅读输入数据，并输出最大可能的不失望人数，将结果放在[answer]和[/answer]标签之间。

示例输入：
5
15 2 1 5 3

示例输出：
[answer]4[/answer]

现在，根据以下输入数据，给出你的解答：

输入：
{n}
{ti_str}

请确保答案的格式正确，并将最终结果放置在[answer]标签中。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            return int(last_answer)
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        ti = identity['ti']
        ti_sorted = sorted(ti)
        if not ti_sorted:
            return solution == 0
        s = ti_sorted[0]
        o = 1
        for x in range(len(ti_sorted) - 1):
            next_t = ti_sorted[x + 1]
            if s <= next_t:
                s += next_t
                o += 1
        return solution == o

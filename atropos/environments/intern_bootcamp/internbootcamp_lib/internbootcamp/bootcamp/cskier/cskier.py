"""# 

### 谜题描述
Skier rides on a snowy field. Its movements can be described by a string of characters 'S', 'N', 'W', 'E' (which correspond to 1 meter movement in the south, north, west or east direction respectively).

It is known that if he moves along a previously unvisited segment of a path (i.e. this segment of the path is visited the first time), then the time of such movement is 5 seconds. If he rolls along previously visited segment of a path (i.e., this segment of the path has been covered by his path before), then it takes 1 second.

Find the skier's time to roll all the path.

Input

The first line contains an integer t (1 ≤ t ≤ 10^4) — the number of test cases in the input. Then t test cases follow.

Each set is given by one nonempty string of the characters 'S', 'N', 'W', 'E'. The length of the string does not exceed 10^5 characters.

The sum of the lengths of t given lines over all test cases in the input does not exceed 10^5.

Output

For each test case, print the desired path time in seconds.

Example

Input


5
NNN
NS
WWEN
WWEE
NWNWS


Output


15
6
16
12
25

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
t = int(raw_input())
for _ in range(t):
    ans = 0
    s = raw_input()
    x, y = 0, 0
    d = {}
    for c in s:
        ox, oy = x, y
        if c == 'N': y += 1
        if c == 'S': y -= 1
        if c == 'E': x += 1
        if c == 'W': x -= 1
        if ((x, y, ox, oy)) in d:
            ans += 1
        else:
            ans += 5
            d[(x, y, ox, oy)] = 1
            d[(ox, oy, x, y)] = 1
    print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
import re

from bootcamp import Basebootcamp

class Cskierbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化Cskierbootcamp类，设置生成测试用例的参数。
        """
        self.min_length = params.get('min_length', 1)
        self.max_length = params.get('max_length', 100)
        self.directions = ['N', 'S', 'E', 'W']
    
    def case_generator(self):
        """
        生成一个Cskier谜题的实例。
        """
        # 生成随机长度的字符串
        length = random.randint(self.min_length, self.max_length)
        s = ''.join(random.choices(self.directions, k=length))
        
        # 计算正确答案
        correct_answer = self._calculate_time(s)
        
        # 返回可JSON序列化的字典
        return {
            "input_string": s,
            "correct_answer": correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        将问题实例转换为文本形式的问题。
        """
        s = question_case["input_string"]
        prompt = (
            "你是一名滑雪者，正在雪地上滑行。你的移动路径由字符串表示，每个字符代表一个方向：\n"
            "N - 北（y坐标增加1）\n"
            "S - 南（y坐标减少1）\n"
            "E - 东（x坐标增加1）\n"
            "W - 西（x坐标减少1）\n"
            "\n"
            "规则说明：\n"
            "1. 每次移动1米。\n"
            "2. 如果移动到一个从未经过的路径段，花费5秒。\n"
            "3. 如果移动到一个已经经过的路径段，花费1秒。\n"
            "\n"
            f"你的任务是计算以下路径的总时间：{s}\n"
            "请将答案放在[answer]标签中，格式为：\n"
            "[answer]总时间[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        从LLM的回复中提取答案。
        """
        # 查找所有匹配项，取最后一个
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案是否正确。
        """
        # 将提取的答案转换为整数
        try:
            solution_int = int(solution)
        except ValueError:
            return False
        
        # 获取正确的答案
        correct_answer = identity["correct_answer"]
        
        return solution_int == correct_answer
    
    def _calculate_time(self, s):
        """
        计算滑雪者的时间。
        """
        x, y = 0, 0
        visited = set()
        time = 0
        
        for c in s:
            ox, oy = x, y
            if c == 'N':
                y += 1
            elif c == 'S':
                y -= 1
            elif c == 'E':
                x += 1
            elif c == 'W':
                x -= 1
            
            # 生成当前路径段的两种表示方式
            segment1 = (x, y, ox, oy)
            segment2 = (ox, oy, x, y)
            
            if segment1 in visited or segment2 in visited:
                time += 1
            else:
                time += 5
                visited.add(segment1)
                visited.add(segment2)
        
        return time

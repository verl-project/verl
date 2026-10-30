"""# 

### 谜题描述
A very brave explorer Petya once decided to explore Paris catacombs. Since Petya is not really experienced, his exploration is just walking through the catacombs.

Catacombs consist of several rooms and bidirectional passages between some pairs of them. Some passages can connect a room to itself and since the passages are built on different depths they do not intersect each other. Every minute Petya arbitrary chooses a passage from the room he is currently in and then reaches the room on the other end of the passage in exactly one minute. When he enters a room at minute i, he makes a note in his logbook with number ti: 

  * If Petya has visited this room before, he writes down the minute he was in this room last time; 
  * Otherwise, Petya writes down an arbitrary non-negative integer strictly less than current minute i. 



Initially, Petya was in one of the rooms at minute 0, he didn't write down number t0.

At some point during his wandering Petya got tired, threw out his logbook and went home. Vasya found his logbook and now he is curious: what is the minimum possible number of rooms in Paris catacombs according to Petya's logbook?

Input

The first line contains a single integer n (1 ≤ n ≤ 2·105) — then number of notes in Petya's logbook.

The second line contains n non-negative integers t1, t2, ..., tn (0 ≤ ti < i) — notes in the logbook.

Output

In the only line print a single integer — the minimum possible number of rooms in Paris catacombs.

Examples

Input

2
0 0


Output

2


Input

5
0 1 0 1 3


Output

3

Note

In the first sample, sequence of rooms Petya visited could be, for example 1 → 1 → 2, 1 → 2 → 1 or 1 → 2 → 3. The minimum possible number of rooms is 2.

In the second sample, the sequence could be 1 → 2 → 3 → 1 → 2 → 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
n = input()
arr = list(map(int, raw_input().split()))
tail = [0 for i in range(n+1)]
ans = 1
for i in range(1, len(arr)+1):
	x = arr[i-1]
	if tail[x]:
		ans += 1
	else:
		tail[x] = 1

print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Apetyaandcatacombsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        """严格符合ti < i约束的案例生成"""
        n = random.randint(self.min_n, self.max_n)
        return {
            'n': n,
            't': [random.randint(0, i-1) for i in range(1, n+1)]
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        t_values = ' '.join(map(str, question_case['t']))
        return f"""你是地下墓穴路径分析专家，需要根据Petya的移动日志确定最小房间数。以下是任务详情：

## 背景规则
1. 每分钟移动到相邻房间
2. 新房间：记录任意小于当前时间的数
3. 旧房间：记录上次访问时间

## 输入案例
第一行（房间数）：{n}
第二行（时间序列）：{t_values}

## 答案要求
将最终答案放在[answer]标签内，如：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        """增强格式兼容性的答案提取"""
        matches = re.findall(r'\[answer\s*](.*?)\[/answer\s*]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            return int(matches[-1].strip().split()[0])  # 处理可能的多余内容
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """动态内存的验证算法"""
        last_occurrence = {}
        room_count = 1
        current_time = 1
        
        for ti in identity['t']:
            if ti in last_occurrence and last_occurrence[ti] >= current_time - len(last_occurrence):
                room_count += 1
                last_occurrence.clear()
            last_occurrence[ti] = current_time
            current_time += 1
            
        return solution == room_count

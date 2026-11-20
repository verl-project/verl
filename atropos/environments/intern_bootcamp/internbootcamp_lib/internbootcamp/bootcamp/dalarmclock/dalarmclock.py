"""# 

### 谜题描述
Every evening Vitalya sets n alarm clocks to wake up tomorrow. Every alarm clock rings during exactly one minute and is characterized by one integer ai — number of minute after midnight in which it rings. Every alarm clock begins ringing at the beginning of the minute and rings during whole minute. 

Vitalya will definitely wake up if during some m consecutive minutes at least k alarm clocks will begin ringing. Pay attention that Vitalya considers only alarm clocks which begin ringing during given period of time. He doesn't consider alarm clocks which started ringing before given period of time and continues ringing during given period of time.

Vitalya is so tired that he wants to sleep all day long and not to wake up. Find out minimal number of alarm clocks Vitalya should turn off to sleep all next day. Now all alarm clocks are turned on. 

Input

First line contains three integers n, m and k (1 ≤ k ≤ n ≤ 2·105, 1 ≤ m ≤ 106) — number of alarm clocks, and conditions of Vitalya's waking up. 

Second line contains sequence of distinct integers a1, a2, ..., an (1 ≤ ai ≤ 106) in which ai equals minute on which i-th alarm clock will ring. Numbers are given in arbitrary order. Vitalya lives in a Berland in which day lasts for 106 minutes. 

Output

Output minimal number of alarm clocks that Vitalya should turn off to sleep all next day long.

Examples

Input

3 3 2
3 5 1


Output

1


Input

5 10 3
12 8 18 25 1


Output

0


Input

7 7 2
7 3 4 1 6 5 2


Output

6


Input

2 2 2
1 3


Output

0

Note

In first example Vitalya should turn off first alarm clock which rings at minute 3.

In second example Vitalya shouldn't turn off any alarm clock because there are no interval of 10 consequence minutes in which 3 alarm clocks will ring.

In third example Vitalya should turn off any 6 alarm clocks.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m,k=map(int,raw_input().split())
lst=map(int,raw_input().split())
lst.sort()
from bisect import * 
i=1
j=0
c=0
ans=0
s=[]
for i in range(0,n):
  while len(s)>0 and lst[i]-s[0]>=m:
    s=s[1:]
  if len(s)+1<k:
    s.append(lst[i])
  else:
    ans=ans+1
    
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bisect import bisect_left, bisect_right
from bootcamp import Basebootcamp

class Dalarmclockbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=100, max_time=1000):
        self.max_n = max(max_n, 1)
        self.max_m = max(max_m, 1)
        self.max_time = max(max_time, self.max_n + 2)  # 保证足够的时间点生成

    def case_generator(self):
        n = random.randint(1, self.max_n)
        k = random.randint(1, n)
        m = random.randint(1, self.max_m)
        
        # 生成n个唯一且可能的时间点（确保足够的时间空间）
        possible_times = list(range(1, self.max_time + 1))
        ai = random.sample(possible_times, n)
        
        # 计算正确答案
        ai_sorted = sorted(ai)
        ans = self.calculate_min_turn_off(n, m, k, ai_sorted)
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'alarms': ai,  # 保持原始输入顺序
            'ans': ans
        }

    @staticmethod
    def calculate_min_turn_off(n, m, k, a_sorted):
        s = []
        ans = 0
        for x in a_sorted:
            # 维护滑动窗口
            while s and x - s[0] >= m:
                del s[0]
            if len(s) + 1 < k:
                s.append(x)
            else:
                ans += 1
        return ans

    @staticmethod
    def prompt_func(question_case) -> str:
        alarm_list = ' '.join(map(str, question_case['alarms']))
        return f"""根据以下条件计算需要关闭的最小闹钟数：
- 总闹钟数：{question_case['n']}
- 危险窗口时长：{question_case['m']}分钟
- 唤醒阈值：{question_case['k']}个闹钟
- 闹钟时间（原始无序输入）：{alarm_list}

请将最终答案放在[answer]和[/answer]标记之间，例如：[answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        # 严格匹配标签大小写，允许跨行匹配
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        # 处理可能的换行和空格
        last_match = matches[-1].strip().replace('\n', ' ')
        # 合并连续空格
        return ' '.join(last_match.split())

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['ans']
        except (ValueError, TypeError):
            return False

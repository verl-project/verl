"""# 

### 谜题描述
Emuskald is addicted to Codeforces, and keeps refreshing the main page not to miss any changes in the \"recent actions\" list. He likes to read thread conversations where each thread consists of multiple messages.

Recent actions shows a list of n different threads ordered by the time of the latest message in the thread. When a new message is posted in a thread that thread jumps on the top of the list. No two messages of different threads are ever posted at the same time.

Emuskald has just finished reading all his opened threads and refreshes the main page for some more messages to feed his addiction. He notices that no new threads have appeared in the list and at the i-th place in the list there is a thread that was at the ai-th place before the refresh. He doesn't want to waste any time reading old messages so he wants to open only threads with new messages.

Help Emuskald find out the number of threads that surely have new messages. A thread x surely has a new message if there is no such sequence of thread updates (posting messages) that both conditions hold: 

  1. thread x is not updated (it has no new messages); 
  2. the list order 1, 2, ..., n changes to a1, a2, ..., an. 

Input

The first line of input contains an integer n, the number of threads (1 ≤ n ≤ 105). The next line contains a list of n space-separated integers a1, a2, ..., an where ai (1 ≤ ai ≤ n) is the old position of the i-th thread in the new list. It is guaranteed that all of the ai are distinct.

Output

Output a single integer — the number of threads that surely contain a new message.

Examples

Input

5
5 2 1 3 4


Output

2


Input

3
1 2 3


Output

0


Input

4
4 3 2 1


Output

3

Note

In the first test case, threads 2 and 5 are placed before the thread 1, so these threads must contain new messages. Threads 1, 3 and 4 may contain no new messages, if only threads 2 and 5 have new messages.

In the second test case, there may be no new messages at all, since the thread order hasn't changed.

In the third test case, only thread 1 can contain no new messages.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
nums = map(int,raw_input().split())[::-1]

idx = 0
pre = n + 1
for num in nums:
    if num < pre:
        idx += 1
        pre = num
    else:
        break
print n-idx
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Bmultithreadingbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        """生成符合题意的随机测试用例"""
        n = random.randint(self.min_n, self.max_n)
        a = list(range(1, n+1))
        random.shuffle(a)
        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        """准确传递输入数据的自然语言问题描述"""
        n = question_case['n']
        a_str = ' '.join(map(str, question_case['a']))  # 关键修复：不要反转数组
        return f"""你正在帮助Emuskald分析Codeforces的最近活动列表。列表中有n个不同的线程，当某个线程收到新消息时，它会跳到列表最前面。现在列表顺序已更新，已知刷新后的第i个位置(1≤i≤n)对应的线程在刷新前的位置是a_i（所有a_i构成1到n的排列）。

输入格式：
第一行：整数n
第二行：n个互不相同的整数a_1到a_n（1≤a_i≤n）

当前测试用例：
n = {n}
a = {a_str}

请严格按以下步骤分析：
1. 识别所有必须包含新消息的线程
2. 将答案的整数值放在[answer]和[/answer]之间，例如：[answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        """增强鲁棒性的答案提取"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格遵循参考代码逻辑的验证"""
        a = identity['a'][::-1]  # 此处进行正确的数组方向处理
        pre_min, count = len(a) + 1, 0
        
        for num in a:
            if num < pre_min:
                pre_min = num
                count += 1
            else:
                break
        return solution == (len(a) - count)

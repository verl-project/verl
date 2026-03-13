"""# 

### 谜题描述
Polycarp is a music editor at the radio station. He received a playlist for tomorrow, that can be represented as a sequence a1, a2, ..., an, where ai is a band, which performs the i-th song. Polycarp likes bands with the numbers from 1 to m, but he doesn't really like others. 

We define as bj the number of songs the group j is going to perform tomorrow. Polycarp wants to change the playlist in such a way that the minimum among the numbers b1, b2, ..., bm will be as large as possible.

Find this maximum possible value of the minimum among the bj (1 ≤ j ≤ m), and the minimum number of changes in the playlist Polycarp needs to make to achieve it. One change in the playlist is a replacement of the performer of the i-th song with any other group.

Input

The first line of the input contains two integers n and m (1 ≤ m ≤ n ≤ 2000).

The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 109), where ai is the performer of the i-th song.

Output

In the first line print two integers: the maximum possible value of the minimum among the bj (1 ≤ j ≤ m), where bj is the number of songs in the changed playlist performed by the j-th band, and the minimum number of changes in the playlist Polycarp needs to make.

In the second line print the changed playlist.

If there are multiple answers, print any of them.

Examples

Input

4 2
1 2 3 2


Output

2 1
1 2 1 2 


Input

7 3
1 3 2 2 2 2 1


Output

2 1
1 3 3 2 2 2 1 


Input

4 4
1000000000 100 7 1000000000


Output

1 4
1 2 3 4 

Note

In the first sample, after Polycarp's changes the first band performs two songs (b1 = 2), and the second band also performs two songs (b2 = 2). Thus, the minimum of these values equals to 2. It is impossible to achieve a higher minimum value by any changes in the playlist. 

In the second sample, after Polycarp's changes the first band performs two songs (b1 = 2), the second band performs three songs (b2 = 3), and the third band also performs two songs (b3 = 2). Thus, the best minimum value is 2. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def foo():
    n, m = map(int, raw_input().split(' '))
    a = map(int, raw_input().split(' '))
    ans1, ans2 = n / m, 0
    b = [[] for i in xrange(m + 1)]
    available = []
    for i in xrange(n):
        if a[i] > m:
            available.append(i)
        else:
            b[a[i]].append(i)
    for i in xrange(1, m + 1):
        if len(b[i]) > ans1:
            for j in xrange(len(b[i]) - ans1):
                available.append(b[i][j])
    for i in xrange(1, m + 1):
        if len(b[i]) < ans1:
            for j in xrange(ans1 - len(b[i])):
                a[available[-1]] = i
                del available[-1]
                ans2 += 1
    print ans1, ans2
    print \" \".join(map(str, a))

if __name__ == '__main__': foo()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cpolycarpattheradiobootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = {
            'min_n': 1,
            'max_n': 2000,
            'min_m': 1,
            'max_m': 2000,
            **params
        }
    
    def case_generator(self):
        min_n = self.params.get('min_n', 1)
        max_n = self.params.get('max_n', 2000)
        min_m = self.params.get('min_m', 1)
        max_m = self.params.get('max_m', 2000)
        
        n = random.randint(min_n, max_n)
        m = random.randint(min_m, min(n, max_m))
        
        a = []
        for _ in range(n):
            if random.random() < 0.7:
                a.append(random.randint(1, m))
            else:
                a.append(random.randint(m + 1, 10**9))
        
        k = n // m
        b = [[] for _ in range(m + 1)]
        available = []
        for i in range(n):
            if a[i] > m:
                available.append(i)
            else:
                b[a[i]].append(i)
        
        for i in range(1, m + 1):
            if len(b[i]) > k:
                for j in range(len(b[i]) - k):
                    available.append(b[i][j])
        
        changes = 0
        new_a = a.copy()
        for i in range(1, m + 1):
            if len(b[i]) < k:
                needed = k - len(b[i])
                for j in range(needed):
                    if not available:
                        break
                    idx = available.pop()
                    new_a[idx] = i
                    changes += 1
        
        identity = {
            'n': n,
            'm': m,
            'a': a,
            'k': k,
            'changes': changes,
            'new_a': new_a
        }
        return identity
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        a = question_case['a']
        prompt = f"你是一位音乐编辑，Cpolycarpattheradio。你收到一个包含{n}首歌的播放列表，每首歌由一个乐队编号表示：{a}。你需要调整这个列表，使得前{m}个乐队的歌曲数量中的最小值尽可能大。同时，改变次数要尽可能少。改变次数是指将歌曲的乐队编号替换为另一个的次数。你的任务是找到最大的可能的最小值k，以及最少的改变次数，然后输出修改后的播放列表。\n\n规则：\n1. 播放列表的长度是{n}。\n2. 乐队编号可以是任意正整数，但只有前{m}个乐队是被喜欢的。\n3. 目标是使得每个前{m}个乐队的歌曲数量至少为k，且k尽可能大。\n4. 在k最大的前提下，改变次数要尽可能少。\n\n请提供以下内容，并将答案放在[answer]标签中：\n1. 第一行：k和改变次数，用空格分隔。\n2. 第二行：修改后的播放列表，用空格分隔。\n\n例如，答案格式为：\n[answer]\n2 1\n1 2 1 2\n[/answer]\n"
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = content.split('\n')
        if len(lines) < 2:
            return None
        first_line = lines[0].strip()
        k_changes = first_line.split()
        if len(k_changes) != 2:
            return None
        try:
            k = int(k_changes[0])
            changes = int(k_changes[1])
        except ValueError:
            return None
        new_a_line = lines[1].strip()
        new_a = list(map(int, new_a_line.split()))
        return {
            'k': k,
            'changes': changes,
            'new_a': new_a
        }
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or not identity:
            return False
        n = identity['n']
        m = identity['m']
        a = identity['a']
        expected_k = identity['k']
        expected_changes = identity['changes']
        
        if 'k' not in solution or 'changes' not in solution or 'new_a' not in solution:
            return False
        k = solution['k']
        changes = solution['changes']
        new_a = solution['new_a']
        
        if len(new_a) != n:
            return False
        if k != expected_k:
            return False
        if changes != expected_changes:
            return False
        
        count = {}
        for i in range(1, m + 1):
            count[i] = 0
        for num in new_a:
            if 1 <= num <= m:
                count[num] += 1
        min_b = min(count.values())
        if min_b < expected_k:
            return False
        
        return True

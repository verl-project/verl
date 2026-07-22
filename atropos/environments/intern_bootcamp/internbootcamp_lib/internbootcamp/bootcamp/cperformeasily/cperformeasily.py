"""# 

### 谜题描述
After battling Shikamaru, Tayuya decided that her flute is too predictable, and replaced it with a guitar. The guitar has 6 strings and an infinite number of frets numbered from 1. Fretting the fret number j on the i-th string produces the note a_{i} + j.

Tayuya wants to play a melody of n notes. Each note can be played on different string-fret combination. The easiness of performance depends on the difference between the maximal and the minimal indices of used frets. The less this difference is, the easier it is to perform the technique. Please determine the minimal possible difference.

For example, if a = [1, 1, 2, 2, 3, 3], and the sequence of notes is 4, 11, 11, 12, 12, 13, 13 (corresponding to the second example), we can play the first note on the first string, and all the other notes on the sixth string. Then the maximal fret will be 10, the minimal one will be 3, and the answer is 10 - 3 = 7, as shown on the picture.

<image>

Input

The first line contains 6 space-separated numbers a_{1}, a_{2}, ..., a_{6} (1 ≤ a_{i} ≤ 10^{9}) which describe the Tayuya's strings.

The second line contains the only integer n (1 ≤ n ≤ 100 000) standing for the number of notes in the melody.

The third line consists of n integers b_{1}, b_{2}, ..., b_{n} (1 ≤ b_{i} ≤ 10^{9}), separated by space. They describe the notes to be played. It's guaranteed that b_i > a_j for all 1≤ i≤ n and 1≤ j≤ 6, in other words, you can play each note on any string.

Output

Print the minimal possible difference of the maximal and the minimal indices of used frets.

Examples

Input


1 4 100 10 30 5
6
101 104 105 110 130 200


Output


0


Input


1 1 2 2 3 3
7
13 4 11 12 11 13 12


Output


7

Note

In the first sample test it is optimal to play the first note on the first string, the second note on the second string, the third note on the sixth string, the fourth note on the fourth string, the fifth note on the fifth string, and the sixth note on the third string. In this case the 100-th fret is used each time, so the difference is 100 - 100 = 0.

<image>

In the second test it's optimal, for example, to play the second note on the first string, and all the other notes on the sixth string. Then the maximal fret will be 10, the minimal one will be 3, and the answer is 10 - 3 = 7.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import operator
import sys

a = list(set(map(int, sys.stdin.readline().split())))
n = int(input())
c2 = list(set(map(int, sys.stdin.readline().split())))

s = 0
v = []
index = 0
n = len(c2)
c = [0]*n
qm = 10e8+1
for q in range(len(a)) :
    for i in range(len(c2)) :
        v.append([c2[i]-a[q], i])
v.sort(key=operator.itemgetter(0))

for i in range(len(v)) :
    c[v[i][1]] += 1
    if c[v[i][1]] == 1 : s += 1
    if s == n :
        while c[v[index][1]] > 1 :
            c[v[index][1]] -= 1
            index += 1
        if v[i][0] - v[index][0] < qm : qm = v[i][0] - v[index][0]
        c[v[index][1]] -= 1
        if c[v[index][1]] == 0 : s -= 1
        index += 1

sys.stdout.write(str(qm))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Cperformeasilybootcamp(Basebootcamp):
    def __init__(self, a_params=None, n_min=1, n_max=100):
        # 参数校验与初始化逻辑强化
        if a_params:
            if len(a_params) < 6:
                a_params += [a_params[-1]] * (6 - len(a_params))
            self.a_list = sorted(a_params[:6], reverse=True)  # 降序排列提升生成效率
        else:
            self.a_list = sorted([random.randint(1, 1e3) for _ in range(6)], reverse=True)
        
        self.n_min = max(1, n_min)
        self.n_max = max(self.n_min, n_max)
        self.params = {
            'a_list': self.a_list,
            'n_range': (self.n_min, self.n_max)
        }

    def case_generator(self):
        """强化测试用例生成逻辑"""
        a = self.a_list
        a_max = a[0]  # 降序排列后第一个元素最大
        
        # 生成符合 b_i > a_j 的音符序列（对所有j成立）
        n = random.randint(self.n_min, self.n_max)
        min_fret = 1
        max_fret = 100
        
        # 生成包含多样性的测试用例
        b = []
        for _ in range(n):
            # 选择3种生成模式保证多样性
            mode = random.choice([1, 2, 3])
            
            if mode == 1:  # 基于最大基准音
                fret = random.randint(min_fret, max_fret)
                b_val = a_max + fret
            elif mode == 2:  # 基于中间基准音
                base = random.choice(a[1:-1])
                fret = random.randint(min_fret, max_fret)
                b_val = base + fret
            else:  # 强制产生统一解
                common_fret = random.randint(50, 100)
                b_val = a_max + common_fret
                
            b.append(b_val)
        
        # 保证至少存在一个统一解
        b.append(a_max + (max_fret // 2))
        
        # 计算结果时使用去重基准音
        a_unique = list(set(a))
        correct_answer = self.compute_min_difference(a_unique, b)
        
        return {
            'a': a,
            'n': len(b),
            'b': b,
            'correct_answer': correct_answer
        }

    @staticmethod
    def compute_min_difference(a_unique, b):
        """优化滑动窗口算法"""
        candidates = []
        for idx, note in enumerate(b):
            for base in a_unique:
                if note > base:
                    candidates.append((note - base, idx))
        
        if not candidates:
            return 0
        
        candidates.sort()
        min_diff = float('inf')
        counter = defaultdict(int)
        unique_count = 0
        left = 0
        
        for right in range(len(candidates)):
            r_val, r_idx = candidates[right]
            if counter[r_idx] == 0:
                unique_count += 1
            counter[r_idx] += 1
            
            # 窗口收缩优化
            while unique_count == len(b) and left <= right:
                current_diff = r_val - candidates[left][0]
                min_diff = min(min_diff, current_diff)
                
                l_val, l_idx = candidates[left]
                counter[l_idx] -= 1
                if counter[l_idx] == 0:
                    unique_count -= 1
                left += 1
                
        return min_diff if min_diff != float('inf') else 0

    @staticmethod
    def prompt_func(question_case) -> str:
        """增强问题描述规范性"""
        a = question_case['a']
        b = question_case['b']
        return f"""
# 吉他演奏优化问题

## 题目要求
你正在为吉他演奏设计最佳指法方案。吉他共有6根弦，各弦基础音高为（按弦编号1-6）：
{a[0]}, {a[1]}, {a[2]}, {a[3]}, {a[4]}, {a[5]}

## 演奏规则
1. 在第i根弦按压第j品时，实际音高 = 基础音a[i] + j（j ≥ 1）
2. 需要演奏的音符序列：{b}
3. 每个音符必须且只能选择一个弦进行演奏

## 优化目标
找到使得所有使用品位的最大值与最小值的差最小的演奏方案，返回这个最小差值。

请将最终答案用[answer]标签包裹，例如：[answer]42[/answer]
        """.strip()

    @staticmethod
    def extract_output(output):
        """强化答案提取鲁棒性"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格答案校验"""
        return solution == identity['correct_answer']

# 验证测试样例
if __name__ == "__main__":
    # 测试第一个示例
    test1 = {
        'a': [100, 30, 10, 5, 4, 1],  # 降序排列
        'b': [101, 104, 105, 110, 130, 200, 150]
    }
    assert Cperformeasilybootcamp.compute_min_difference(
        list(set(test1['a'])),
        test1['b']
    ) == 0

    # 测试第二个示例
    test2 = {
        'a': [3, 3, 2, 2, 1, 1],
        'b': [13, 4, 11, 12, 11, 13, 12, 7]  # 添加边界值
    }
    assert Cperformeasilybootcamp.compute_min_difference(
        list(set(test2['a'])),
        test2['b']
    ) == 7

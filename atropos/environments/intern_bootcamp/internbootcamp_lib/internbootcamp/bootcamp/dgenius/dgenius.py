"""# 

### 谜题描述
Please note the non-standard memory limit.

There are n problems numbered with integers from 1 to n. i-th problem has the complexity c_i = 2^i, tag tag_i and score s_i.

After solving the problem i it's allowed to solve problem j if and only if IQ < |c_i - c_j| and tag_i ≠ tag_j. After solving it your IQ changes and becomes IQ = |c_i - c_j| and you gain |s_i - s_j| points.

Any problem can be the first. You can solve problems in any order and as many times as you want.

Initially your IQ = 0. Find the maximum number of points that can be earned.

Input

The first line contains a single integer t (1 ≤ t ≤ 100) — the number of test cases. 

The first line of each test case contains an integer n (1 ≤ n ≤ 5000) — the number of problems.

The second line of each test case contains n integers tag_1, tag_2, …, tag_n (1 ≤ tag_i ≤ n) — tags of the problems.

The third line of each test case contains n integers s_1, s_2, …, s_n (1 ≤ s_i ≤ 10^9) — scores of the problems.

It's guaranteed that sum of n over all test cases does not exceed 5000.

Output

For each test case print a single integer — the maximum number of points that can be earned.

Example

Input


5
4
1 2 3 4
5 10 15 20
4
1 2 1 2
5 10 15 20
4
2 2 4 1
2 8 19 1
2
1 1
6 9
1
1
666


Output


35
30
42
0
0

Note

In the first test case optimal sequence of solving problems is as follows: 

  1. 1 → 2, after that total score is 5 and IQ = 2 
  2. 2 → 3, after that total score is 10 and IQ = 4 
  3. 3 → 1, after that total score is 20 and IQ = 6 
  4. 1 → 4, after that total score is 35 and IQ = 14 



In the second test case optimal sequence of solving problems is as follows: 

  1. 1 → 2, after that total score is 5 and IQ = 2 
  2. 2 → 3, after that total score is 10 and IQ = 4 
  3. 3 → 4, after that total score is 15 and IQ = 8 
  4. 4 → 1, after that total score is 35 and IQ = 14 



In the third test case optimal sequence of solving problems is as follows: 

  1. 1 → 3, after that total score is 17 and IQ = 6 
  2. 3 → 4, after that total score is 35 and IQ = 8 
  3. 4 → 2, after that total score is 42 and IQ = 12 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
T = int(raw_input())

for case_ in xrange(T):
    n = int(raw_input())
    ts = map(int, raw_input().split())
    ss = map(int, raw_input().split())

    dp = [0 for i in xrange(n)]

    for i in xrange(n):
        for j in xrange(i - 1, -1, -1):
            if ts[i] == ts[j]:
                continue
            delta = abs(ss[i] - ss[j])
            dp[i], dp[j] = max(dp[i], dp[j] + delta), max(dp[j], dp[i] + delta)
    print max(dp)

'''
^^^TEST^^^
5
4
1 2 3 4
5 10 15 20
4
1 2 1 2
5 10 15 20
4
2 2 4 1
2 8 19 1
2
1 1
6 9
1
1
666
----
35
30
42
0
0
$$$TEST$$$
'''
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dgeniusbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20, **params):
        super().__init__(**params)
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        tags = [random.randint(1, n) for _ in range(n)]
        scores = [random.randint(1, 10**9) for _ in range(n)]
        expected_max = self._calculate_max_score(n, tags, scores)
        return {
            'n': n,
            'tags': tags,
            'scores': scores,
            'expected_max': expected_max
        }
    
    @staticmethod
    def _calculate_max_score(n, tags, scores):
        dp = [0] * n
        for i in range(n):
            for j in range(i-1, -1, -1):
                if tags[i] != tags[j]:
                    delta = abs(scores[i] - scores[j])
                    dp_i_new = max(dp[i], dp[j] + delta)
                    dp_j_new = max(dp[j], dp[i] + delta)
                    dp[i], dp[j] = dp_i_new, dp_j_new
        return max(dp)
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        tags = ' '.join(map(str, question_case['tags']))
        scores = ' '.join(map(str, question_case['scores']))
        problem_desc = f"""你正在参加一个编程竞赛，需要解决以下问题：

有 {n} 个问题，编号1~{n}。每个问题i的复杂度为2^i，标签数组为[{tags}]，分数数组为[{scores}]。初始IQ=0。

解题规则：
1. 解完问题i后，只能解满足以下条件的问题j：
   - 当前IQ < |2^i - 2^j|
   - tag_i ≠ tag_j
2. 解完j后IQ变为|2^i - 2^j|，获得|s_i - s_j|分
3. 可重复解题，但需满足上述条件

求能获得的最大分数。

输入数据：
- 测试用例数：1
- n = {n}
- tags = {tags}
- scores = {scores}

请输出最大分数，格式为[answer]答案[/answer]，如[answer]35[/answer]。"""
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answers:
            return None
        try:
            return int(answers[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_max']

"""# 

### 谜题描述
One day, at the \"Russian Code Cup\" event it was decided to play football as an out of competition event. All participants was divided into n teams and played several matches, two teams could not play against each other more than once.

The appointed Judge was the most experienced member — Pavel. But since he was the wisest of all, he soon got bored of the game and fell asleep. Waking up, he discovered that the tournament is over and the teams want to know the results of all the matches.

Pavel didn't want anyone to discover about him sleeping and not keeping an eye on the results, so he decided to recover the results of all games. To do this, he asked all the teams and learned that the real winner was friendship, that is, each team beat the other teams exactly k times. Help Pavel come up with chronology of the tournir that meets all the conditions, or otherwise report that there is no such table.

Input

The first line contains two integers — n and k (1 ≤ n, k ≤ 1000).

Output

In the first line print an integer m — number of the played games. The following m lines should contain the information about all the matches, one match per line. The i-th line should contain two integers ai and bi (1 ≤ ai, bi ≤ n; ai ≠ bi). The numbers ai and bi mean, that in the i-th match the team with number ai won against the team with number bi. You can assume, that the teams are numbered from 1 to n.

If a tournir that meets the conditions of the problem does not exist, then print -1.

Examples

Input

3 1


Output

3
1 2
2 3
3 1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# -*- coding: utf-8 -*-
\"\"\"
Created on Thu Apr 17 12:19:39 2014

@author: Joy
\"\"\"


nk=raw_input().split()
n=int(nk[0])
k=int(nk[1])

matches=[]
if n-1-k<k:
    print -1
else:
    for i in range(1,n+1):
        for j in range(1,k+1):
            beat=i+j
            if i+j>n:
                beat=beat%n
            matches.append([i, beat])
            
    print len(matches)
    for m in matches:
        print m[0],m[1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cfootballbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_k=1000):
        """
        初始化参数设置生成n和k的最大值，确保案例多样性。
        """
        self.max_n = max_n
        self.max_k = max_k

    def case_generator(self):
        """
        生成随机案例，覆盖有效（k合法）和无效（k过大）两种情况。
        """
        n = random.randint(1, self.max_n)
        if random.random() < 0.5:  # 50%概率生成有效案例
            if n == 1:
                k = 0  # n=1时k只能为0
            else:
                max_valid_k = (n - 1) // 2
                k = random.randint(0, max_valid_k)
        else:  # 50%概率生成无效案例
            if n == 1:
                k = random.randint(1, self.max_k)  # 若n=1，k>0则无解
            else:
                min_invalid_k = (n - 1) // 2 + 1
                k = random.randint(min_invalid_k, min(self.max_k, min_invalid_k + 100))
        return {"n": n, "k": k}

    @staticmethod
    def prompt_func(question_case):
        n = question_case["n"]
        k = question_case["k"]
        return f"""You are to solve a football tournament problem where each team must beat exactly k others. The teams are numbered 1 to {n}, and each pair can play at most once.

Input:
n = {n}, k = {k}

Output:
- If possible, the number of matches followed by each match (winner and loser).
- If impossible, output -1.

Format your answer within [answer] and [/answer] tags. Example for n=3, k=1:
[answer]
3
1 2
2 3
3 1
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r"\[answer\](.*?)\[/answer\]", output, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity["n"]
        k_val = identity["k"]
        # 判断当前案例是否有解
        case_has_solution = (n == 1 and k_val == 0) or (n > 1 and k_val <= (n - 1) // 2)
        sol_str = str(solution).strip()

        if sol_str == "-1":
            return not case_has_solution  # 模型输出-1且案例无解时正确

        if not case_has_solution:
            return False  # 案例无解但模型未输出-1

        # 解析模型输出的比赛记录
        lines = sol_str.split("\n")
        if not lines:
            return False

        try:
            m = int(lines[0])
        except ValueError:
            return False

        expected_m = n * k_val  # 每队赢k次，总比赛数应为n*k
        if m != expected_m:
            return False

        if n == 1 and k_val == 0:
            # 特例：仅一队且k=0时应无比赛
            return m == 0 and len(lines) == 1

        if len(lines) != m + 1:
            return False  # 比赛数量与声明不符

        matches = []
        # 验证每行比赛记录合法性
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 2:
                return False
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                return False
            if a == b or a < 1 or a > n or b < 1 or b > n:
                return False
            matches.append((a, b))

        # 检查每对球队是否至多比赛一次
        pairs = set()
        for a, b in matches:
            pair = tuple(sorted((a, b)))  # 标准化比赛对表示
            if pair in pairs:
                return False
            pairs.add(pair)

        # 检查每队胜利次数是否为k
        win_counts = {team: 0 for team in range(1, n + 1)}
        for a, _ in matches:
            win_counts[a] += 1

        return all(count == k_val for count in win_counts.values())

"""# 

### 谜题描述
There are n employees in Alternative Cake Manufacturing (ACM). They are now voting on some very important question and the leading world media are trying to predict the outcome of the vote.

Each of the employees belongs to one of two fractions: depublicans or remocrats, and these two fractions have opposite opinions on what should be the outcome of the vote. The voting procedure is rather complicated: 

  1. Each of n employees makes a statement. They make statements one by one starting from employees 1 and finishing with employee n. If at the moment when it's time for the i-th employee to make a statement he no longer has the right to vote, he just skips his turn (and no longer takes part in this voting). 
  2. When employee makes a statement, he can do nothing or declare that one of the other employees no longer has a right to vote. It's allowed to deny from voting people who already made the statement or people who are only waiting to do so. If someone is denied from voting he no longer participates in the voting till the very end. 
  3. When all employees are done with their statements, the procedure repeats: again, each employees starting from 1 and finishing with n who are still eligible to vote make their statements. 
  4. The process repeats until there is only one employee eligible to vote remaining and he determines the outcome of the whole voting. Of course, he votes for the decision suitable for his fraction. 



You know the order employees are going to vote and that they behave optimal (and they also know the order and who belongs to which fraction). Predict the outcome of the vote.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 200 000) — the number of employees. 

The next line contains n characters. The i-th character is 'D' if the i-th employee is from depublicans fraction or 'R' if he is from remocrats.

Output

Print 'D' if the outcome of the vote will be suitable for depublicans and 'R' if remocrats will win.

Examples

Input

5
DDRRR


Output

D


Input

6
DDRRRR


Output

R

Note

Consider one of the voting scenarios for the first sample: 

  1. Employee 1 denies employee 5 to vote. 
  2. Employee 2 denies employee 3 to vote. 
  3. Employee 3 has no right to vote and skips his turn (he was denied by employee 2). 
  4. Employee 4 denies employee 2 to vote. 
  5. Employee 5 has no right to vote and skips his turn (he was denied by employee 1). 
  6. Employee 1 denies employee 4. 
  7. Only employee 1 now has the right to vote so the voting ends with the victory of depublicans. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/python
import sys
n = int(raw_input())

s = raw_input()
seq = list(s)
d = []
r = []
rd = 0
dd = 0

for i in range(n):
	if seq[i]=='D':
		d.append('D')
	else:
		r.append('R')

first  = True
while (r and d) or first:
	first = False
	for i in range(n):
		if seq[i]=='D':
			if dd:
				dd-=1
				d.pop()
				seq[i] = 1
				continue
			else:
				rd+=1
		if seq[i]=='R':
			if rd:
				rd-=1
				r.pop()
				seq[i] = 1
				continue
			else:
				dd+=1

if d:
	print \"D\"
else:
	print \"R\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Cvotingbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=200000, d_prob=0.5):
        """
        初始化投票训练场参数。
        
        参数:
            min_n (int): 最小员工数，默认1
            max_n (int): 最大员工数，默认200000
            d_prob (float): 生成D的概率，默认0.5
        """
        self.min_n = min_n
        self.max_n = max_n
        self.d_prob = d_prob
    
    def case_generator(self):
        """
        生成投票案例，包含随机员工序列和正确答案。
        """
        n = random.randint(self.min_n, self.max_n)
        sequence = ''.join(['D' if random.random() < self.d_prob else 'R' for _ in range(n)])
        correct_answer = self._solve_puzzle(n, sequence)
        return {
            'n': n,
            'sequence': sequence,
            'correct_answer': correct_answer
        }
        
    @staticmethod
    def _solve_puzzle(n, sequence):
        """
        准确复现参考代码逻辑的解答方法。
        """
        dq = deque()
        rq = deque()
        for i, c in enumerate(sequence):
            if c == 'D':
                dq.append(i)
            else:
                rq.append(i)
        
        while dq and rq:
            for i in range(n):
                if not (dq and rq):
                    break
                if sequence[i] == 'D' and i == dq[0]:
                    if dq and rq:
                        rq.popleft()
                        dq.popleft()
                elif sequence[i] == 'R' and i == rq[0]:
                    if dq and rq:
                        dq.popleft()
                        rq.popleft()
        
        return 'D' if dq else 'R'

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        生成包含详细规则和问题实例的提示文本。
        """
        n = question_case['n']
        sequence = question_case['sequence']
        return f"""你是Alternative Cake Manufacturing (ACM)的投票结果预测专家。公司有{n}名员工正在就一个重要问题进行投票，他们的投票顺序和所属派系如下：每位员工按顺序依次属于派系{sequence}（第i个字符代表第i位员工的派系，D代表depublicans，R代表remocrats）。

投票规则如下：
1. 投票过程分为多轮进行。每一轮中，未被淘汰的员工按照初始顺序依次发言。
2. 每位员工在发言时，可以选择淘汰对方派系的下一个即将发言的成员，或者不采取行动。
3. 当员工被淘汰后，将不再参与后续的投票过程。
4. 此过程重复进行，直到只剩下一名员工未被淘汰，该员工所属的派系将赢得投票。
5. 所有员工都将采取最优策略，即优先淘汰对方派系的下一个可能发言者以确保己方胜利。

请预测最终的投票结果，即胜利方是D还是R？请将最终答案严格放置在[answer]标签内，例如[answer]D[/answer]。"""

    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取最后一个符合格式的答案。
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        last_match = matches[-1].strip().upper()
        return last_match if last_match in ('D', 'R') else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        严格验证答案准确性。
        """
        return solution == identity['correct_answer']

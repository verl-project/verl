"""# 

### 谜题描述
Student Valera is an undergraduate student at the University. His end of term exams are approaching and he is to pass exactly n exams. Valera is a smart guy, so he will be able to pass any exam he takes on his first try. Besides, he can take several exams on one day, and in any order.

According to the schedule, a student can take the exam for the i-th subject on the day number ai. However, Valera has made an arrangement with each teacher and the teacher of the i-th subject allowed him to take an exam before the schedule time on day bi (bi < ai). Thus, Valera can take an exam for the i-th subject either on day ai, or on day bi. All the teachers put the record of the exam in the student's record book on the day of the actual exam and write down the date of the mark as number ai.

Valera believes that it would be rather strange if the entries in the record book did not go in the order of non-decreasing date. Therefore Valera asks you to help him. Find the minimum possible value of the day when Valera can take the final exam if he takes exams so that all the records in his record book go in the order of non-decreasing date.

Input

The first line contains a single positive integer n (1 ≤ n ≤ 5000) — the number of exams Valera will take.

Each of the next n lines contains two positive space-separated integers ai and bi (1 ≤ bi < ai ≤ 109) — the date of the exam in the schedule and the early date of passing the i-th exam, correspondingly.

Output

Print a single integer — the minimum possible number of the day when Valera can take the last exam if he takes all the exams so that all the records in his record book go in the order of non-decreasing date.

Examples

Input

3
5 2
3 1
4 2


Output

2


Input

3
6 1
5 2
4 3


Output

6

Note

In the first sample Valera first takes an exam in the second subject on the first day (the teacher writes down the schedule date that is 3). On the next day he takes an exam in the third subject (the teacher writes down the schedule date, 4), then he takes an exam in the first subject (the teacher writes down the mark with date 5). Thus, Valera takes the last exam on the second day and the dates will go in the non-decreasing order: 3, 4, 5.

In the second sample Valera first takes an exam in the third subject on the fourth day. Then he takes an exam in the second subject on the fifth day. After that on the sixth day Valera takes an exam in the first subject.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
R=lambda:map(int,raw_input().split())
def cmp(x,y):
	if x[0]>y[0] or (x[0]==y[0] and x[1]>y[1]): return 1
	return -1
ans=0
for x in sorted([R() for i in range(input())],cmp=cmp):
	if ans<=x[1]: ans=x[1]
	else: ans=x[0]
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Aexamsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, max_ai=10**9):
        self.min_n = min_n
        self.max_n = max_n
        self.max_ai = max_ai
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        exams = []
        for _ in range(n):
            ai = random.randint(2, self.max_ai)
            bi = random.randint(1, ai - 1)
            exams.append((ai, bi))
        return {
            'n': n,
            'exams': exams
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        exams = question_case['exams']
        exam_lines = '\n'.join([f"{a} {b}" for a, b in exams])
        prompt = f"""Valera需要参加{n}场考试。每场考试可以选择在提前日bi或原定日ai进行考试，其中bi < ai。记录本上的日期按ai非递减排列，求最后考试的最早可能日期。

输入格式：
第一行是n，表示考试数量。
接下来n行每行两个整数ai和bi（bi < ai）。

例如输入：
3
5 2
3 1
4 2
正确输出是2，最后考试在第2天。

当前输入：
{n}
{exam_lines}

请计算答案并放在[answer]和[/answer]之间。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        exams = identity['exams']
        sorted_exams = sorted(exams, key=lambda x: (x[0], x[1]))
        ans = 0
        for a, b in sorted_exams:
            if ans <= b:
                ans = b
            else:
                ans = a
        return isinstance(solution, int) and solution == ans

"""# 

### 谜题描述
Vanya wants to pass n exams and get the academic scholarship. He will get the scholarship if the average grade mark for all the exams is at least avg. The exam grade cannot exceed r. Vanya has passed the exams and got grade ai for the i-th exam. To increase the grade for the i-th exam by 1 point, Vanya must write bi essays. He can raise the exam grade multiple times.

What is the minimum number of essays that Vanya needs to write to get scholarship?

Input

The first line contains three integers n, r, avg (1 ≤ n ≤ 105, 1 ≤ r ≤ 109, 1 ≤ avg ≤ min(r, 106)) — the number of exams, the maximum grade and the required grade point average, respectively.

Each of the following n lines contains space-separated integers ai and bi (1 ≤ ai ≤ r, 1 ≤ bi ≤ 106).

Output

In the first line print the minimum number of essays.

Examples

Input

5 5 4
5 2
4 7
3 1
3 2
2 5


Output

4


Input

2 5 4
5 2
5 2


Output

0

Note

In the first sample Vanya can write 2 essays for the 3rd exam to raise his grade by 2 points and 2 essays for the 4th exam to raise his grade by 1 point.

In the second sample, Vanya doesn't need to write any essays as his general point average already is above average.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
[n, r, avg] = map(int, raw_input('').split(' '))
grade_essays = []
for i in xrange(n):
    nums = map(int, raw_input('').split(' '))
    grade_essays.append(nums)


required_total = avg * n
current_total = 0
for i in xrange(n):
    current_total += grade_essays[i][0]

required_essays = 0
grade_essays = sorted(grade_essays, key=lambda grade_essay: grade_essay[1])
current_grade_essay = 0
while current_total < required_total:
    grade = grade_essays[current_grade_essay][0]
    essay = grade_essays[current_grade_essay][1]
    amt_to_add = min(r - grade, required_total - current_total)
    required_essays += ((amt_to_add) * essay)
    current_total += (amt_to_add)
    current_grade_essay += 1
print required_essays
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cvanyaandexamsbootcamp(Basebootcamp):
    def __init__(self, max_exams=10, max_r=1000, max_bi=10**6):
        super().__init__()
        self.max_exams = max_exams
        self.max_r = max_r
        self.max_bi = max_bi

    def case_generator(self):
        # 生成基础参数（确保avg <= r）
        n = random.randint(1, self.max_exams)
        r = random.randint(1, self.max_r)
        avg = random.randint(1, min(r, 10**6))  # 正确约束avg范围

        # 生成考试数据（存在初始分数可能达标的情况）
        exams = []
        for _ in range(n):
            ai = random.randint(1, r)
            bi = random.randint(1, self.max_bi)
            exams.append([ai, bi])

        return {
            'n': n,
            'r': r,
            'avg': avg,
            'exams': exams
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        exams_desc = "\n".join(
            [f"- 第{i+1}门考试：当前分数 {ai} 分（最高可至{r}分），每提升1分需要 {bi} 篇小作文"
             for i, (ai, (r, bi)) in enumerate(zip(
                 [e[0] for e in question_case['exams']],
                 [(question_case['r'], e[1]) for e in question_case['exams']]
             ))]
        )
        
        return f"""Vanya希望通过提高考试分数来获得奖学金。当前情况：
- 已参加考试科目：{question_case['n']}门
- 每科最高分数限制：{question_case['r']}分
- 目标平均分：{question_case['avg']}分

各科详情：
{exams_desc}

请计算Vanya需要撰写的最小作文数量。答案格式为单独一个整数，置于[answer]标签内，如：[answer]1234[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 参数解析
        n, r, avg = identity['n'], identity['r'], identity['avg']
        exams = identity['exams']
        
        # 计算理论需求
        required_total = avg * n
        current_total = sum(ai for ai, _ in exams)
        
        # 情况1：初始已达标
        if current_total >= required_total:
            return solution == 0
        
        # 情况2：需要提升
        sorted_exams = sorted(exams, key=lambda x: x[1])
        deficit = required_total - current_total
        essays = 0
        
        for ai, bi in sorted_exams:
            increment = min(r - ai, deficit)
            essays += increment * bi
            deficit -= increment
            if deficit <= 0:
                break
        
        return essays == solution

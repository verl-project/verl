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
import sys

if __name__ == '__main__':
  n = int(sys.stdin.readline())
  times = []
  for _ in xrange(n):
    a, b = map(int, sys.stdin.readline().strip().split())
    times.append((a, b))

  times.sort()
  current_day = 0
  for a, b in times:
    current_day = b if b >= current_day else a

  print current_day
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class Cexamsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5000, max_a=10**9):
        """
        参数调整支持边界条件
        :param min_n: 考试数下限 (1 ≤ n ≤ 5000)
        :param max_n: 考试数上限 (约束至5000)
        :param max_a: ai最大值 (支持题目要求的1e9)
        """
        self.min_n = max(1, min_n)
        self.max_n = min(5000, max_n)
        self.max_a = max_a
    
    def case_generator(self):
        """生成保证数据有效性的测试案例"""
        n = random.randint(self.min_n, self.max_n)
        exams = []
        
        # 生成有效数据的主逻辑
        for _ in range(n):
            # 生成合法ai（≥2）和bi（1 ≤ bi < ai）
            a = random.randint(2, self.max_a)
            b = random.randint(1, a-1)
            exams.append((a, b))
        
        # 添加30%概率的边界案例
        if random.random() < 0.3:
            # 生成全逆序案例（排序时会自动校正）
            exams.sort(reverse=True)
        else:
            # 随机打乱数据顺序
            random.shuffle(exams)
        
        return {
            "n": n,
            "exams": exams
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """优化格式描述的准确性"""
        exams = question_case['exams']
        n = question_case['n']
        
        problem = (
            "你需要在满足时间顺序约束的条件下，找到完成所有考试的最早最后一天。\n\n"
            "**规则说明**\n"
            "1. 每个考试有两个可选日期：官方日期a_i和提前日期b_i（b_i < a_i）\n"
            "2. 必须按照官方日期的非递减顺序参加考试\n"
            "3. 实际参加日期可以任选b_i或a_i\n"
            "4. 要求找到完成所有考试的最早可能最后一天\n\n"
            "**输入格式**\n"
            f"第一行：{n}\n" +
            "\n".join(f"{a} {b}" for a, b in exams) +
            "\n\n**答案要求**\n将最终答案放在[answer]标签内，例如：[answer]5[/answer]"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        """增强答案提取鲁棒性"""
        matches = re.findall(
            r'\[answer\s*\]\s*(\d+)\s*\[/answer\s*\]',
            output,
            re.IGNORECASE | re.DOTALL
        )
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格验证逻辑"""
        try:
            # 按官方日期排序（核心验证逻辑）
            sorted_exams = sorted(identity['exams'], key=lambda x: x[0])
            last_day = 0
            for a, b in sorted_exams:
                last_day = max(b, last_day) if b >= last_day else a
            return solution == last_day
        except Exception as e:
            print(f"验证异常：{str(e)}")
            return False

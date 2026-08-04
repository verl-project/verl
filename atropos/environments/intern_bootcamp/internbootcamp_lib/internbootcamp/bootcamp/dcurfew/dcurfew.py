"""# 

### 谜题描述
Instructors of Some Informatics School make students go to bed.

The house contains n rooms, in each room exactly b students were supposed to sleep. However, at the time of curfew it happened that many students are not located in their assigned rooms. The rooms are arranged in a row and numbered from 1 to n. Initially, in i-th room there are ai students. All students are currently somewhere in the house, therefore a1 + a2 + ... + an = nb. Also 2 instructors live in this house.

The process of curfew enforcement is the following. One instructor starts near room 1 and moves toward room n, while the second instructor starts near room n and moves toward room 1. After processing current room, each instructor moves on to the next one. Both instructors enter rooms and move simultaneously, if n is odd, then only the first instructor processes the middle room. When all rooms are processed, the process ends.

When an instructor processes a room, she counts the number of students in the room, then turns off the light, and locks the room. Also, if the number of students inside the processed room is not equal to b, the instructor writes down the number of this room into her notebook (and turns off the light, and locks the room). Instructors are in a hurry (to prepare the study plan for the next day), so they don't care about who is in the room, but only about the number of students.

While instructors are inside the rooms, students can run between rooms that are not locked and not being processed. A student can run by at most d rooms, that is she can move to a room with number that differs my at most d. Also, after (or instead of) running each student can hide under a bed in a room she is in. In this case the instructor will not count her during the processing. In each room any number of students can hide simultaneously.

Formally, here is what's happening:

  * A curfew is announced, at this point in room i there are ai students. 
  * Each student can run to another room but not further than d rooms away from her initial room, or stay in place. After that each student can optionally hide under a bed. 
  * Instructors enter room 1 and room n, they count students there and lock the room (after it no one can enter or leave this room). 
  * Each student from rooms with numbers from 2 to n - 1 can run to another room but not further than d rooms away from her current room, or stay in place. Each student can optionally hide under a bed. 
  * Instructors move from room 1 to room 2 and from room n to room n - 1. 
  * This process continues until all rooms are processed. 



Let x1 denote the number of rooms in which the first instructor counted the number of non-hidden students different from b, and x2 be the same number for the second instructor. Students know that the principal will only listen to one complaint, therefore they want to minimize the maximum of numbers xi. Help them find this value if they use the optimal strategy.

Input

The first line contains three integers n, d and b (2 ≤ n ≤ 100 000, 1 ≤ d ≤ n - 1, 1 ≤ b ≤ 10 000), number of rooms in the house, running distance of a student, official number of students in a room.

The second line contains n integers a1, a2, ..., an (0 ≤ ai ≤ 109), i-th of which stands for the number of students in the i-th room before curfew announcement.

It is guaranteed that a1 + a2 + ... + an = nb.

Output

Output one integer, the minimal possible value of the maximum of xi.

Examples

Input

5 1 1
1 0 0 0 4


Output

1


Input

6 1 2
3 8 0 1 0 0


Output

2

Note

In the first sample the first three rooms are processed by the first instructor, and the last two are processed by the second instructor. One of the optimal strategies is the following: firstly three students run from room 5 to room 4, on the next stage two of them run to room 3, and one of those two hides under a bed. This way, the first instructor writes down room 2, and the second writes down nothing.

In the second sample one of the optimal strategies is the following: firstly all students in room 1 hide, all students from room 2 run to room 3. On the next stage one student runs from room 3 to room 4, and 5 students hide. This way, the first instructor writes down rooms 1 and 2, the second instructor writes down rooms 5 and 6.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
func = lambda: map(int,raw_input().split())
n, d, b = func()
d += 1
t, a = 0, [0] * (n + 1)
for i, x in enumerate(func()):
    t += x
    a[i + 1] = t
print(max(i - min(a[min(n, i * d)], (a[n] - a[max(0, n - i * d)])) // b for i in range(n + 3 >> 1)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dcurfewbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_d=9, max_b=10):
        self.max_n = max_n
        self.max_d = max_d
        self.max_b = max_b
    
    def case_generator(self):
        n = random.randint(2, self.max_n)
        max_possible_d = min(n-1, self.max_d)
        d = random.randint(1, max_possible_d)
        b = random.randint(1, self.max_b)
        total = n * b
        
        dividers = [0]
        dividers += sorted(random.sample(range(1, total), n-1))  # 确保生成n-1个分割点
        dividers.append(total)
        a = [dividers[i+1] - dividers[i] for i in range(n)]
        
        return {
            "n": n,
            "d": d,
            "b": b,
            "a": a
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        d = question_case['d']
        b = question_case['b']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        
        problem = f"""The house has {n} rooms arranged in a row, each supposed to have exactly {b} students. Initially, the rooms have the following number of students: {a_str}. Two instructors enforce curfew, starting from opposite ends and moving towards each other. Students can move up to {d} rooms before each processing step and choose to hide. Locked rooms prevent further movement.

**Process Details:**
1. **Initial Move:** All students move up to {d} rooms from their initial position, then optionally hide.
2. **First Processing:** Instructors process rooms 1 and {n}, lock them, and record deviations.
3. **Subsequent Moves:** Remaining students move up to {d} rooms within unlocked rooms, then hide.
4. **Repeat Processing:** Instructors move to next rooms, repeating steps until all are processed. Middle room (if odd) is handled by the first instructor.

**Goal:** Minimize the maximum number of rooms either instructor records. Provide the minimal possible value as an integer within [answer] tags.

Example format: [answer]2[/answer]"""
        return problem
    
    @staticmethod
    def extract_output(output):
        # 严格匹配最后一个答案标签，并提取数字
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        # 提取所有数字，取最后一个有效数字
        numbers = re.findall(r'\d+', last_answer)
        return numbers[-1] if numbers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        def compute_correct(identity):
            n = identity['n']
            d_input = identity['d']
            b = identity['b']
            a_list = identity['a']
            d = d_input + 1  # 按参考代码逻辑处理d
            prefix = [0] * (n + 1)
            current_sum = 0
            for i in range(n):
                current_sum += a_list[i]
                prefix[i+1] = current_sum
            max_val = 0
            max_i = (n + 3) // 2  # 原题中的循环次数计算
            for i in range(max_i):
                pos1 = min(n, i * d)
                term1 = prefix[pos1]
                pos2 = max(0, n - i * d)
                term2 = prefix[n] - prefix[pos2]
                min_t = min(term1, term2)
                count = min_t // b
                current = i - count
                max_val = max(max_val, current)
            return max_val
        
        try:
            # 处理可能的多余字符并转换
            user_answer = int(solution.strip())
            correct_answer = compute_correct(identity)
            return user_answer == correct_answer
        except (ValueError, TypeError, KeyError):
            return False

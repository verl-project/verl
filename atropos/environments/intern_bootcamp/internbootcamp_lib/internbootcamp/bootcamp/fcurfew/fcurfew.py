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
#include <bits/stdc++.h>
using namespace std;
const int MOD = 1000000007;
const int INF = 0x3f3f3f3f;
const long long LL_INF = 0x3f3f3f3f3f3f3f3f;
const double PI = acos(-1);
const double EPS = 1e-10;
const int N = 1000010;
long long n, d, b;
long long s[N];
long long pref[N];
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  cin >> n >> d >> b;
  for (int i = 1; i <= n; ++i) {
    cin >> s[i];
    pref[i] = pref[i - 1] + s[i];
  }
  long long low = 0, high = n, res = 0;
  while (low <= high) {
    long long mid = (low + high) / 2;
    long long cnt = 0;
    bool fail = 0;
    for (int i = 1; i <= n; ++i) {
      if (2 * i <= n + 1) {
        if (i > mid) {
          ++cnt;
          if (pref[min(n, i * (d + 1))] < cnt * b) {
            fail = true;
          }
        }
      }
    }
    cnt = 0;
    for (int i = n; i >= 1; --i) {
      if (2 * i <= n + 1) {
        break;
      }
      if (i + mid <= n) {
        ++cnt;
        if (pref[n] - pref[max(0LL, n - (long long)(n - i + 1) * (d + 1))] <
            cnt * b) {
          fail = true;
        }
      }
    }
    if (fail) {
      low = mid + 1;
    } else {
      res = mid;
      high = mid - 1;
    }
  }
  cout << res;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Fcurfewbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_b=1000):
        self.max_n = max_n
        self.max_b = max_b

    def case_generator(self):
        n = random.randint(2, self.max_n)
        d = random.randint(1, min(n-1, 10))  # 限制d避免极端情况
        b = random.randint(1, self.max_b)
        a = self._generate_valid_a(n, b)
        correct_answer = self.solve(n, d, b, a)
        return {
            'n': n,
            'd': d,
            'b': b,
            'a': a,
            'correct_answer': correct_answer
        }

    def _generate_valid_a(self, n, b):
        """生成合法学生分布，确保数值稳定性"""
        a = [b] * n
        # 有限次数的合理扰动
        for _ in range(min(100, n)):  # 扰动次数与房间数相关
            src, dst = random.sample(range(n), 2)
            if a[src] > 0:
                a[src] -= 1
                a[dst] += 1
        return a

    @staticmethod
    def solve(n, d, b, a_list):
        # 输入校验
        assert sum(a_list) == n * b, "Invalid student distribution"
        
        a = [0] + a_list  # 1-based索引
        pref = [0] * (n + 2)
        for i in range(1, n+1):
            pref[i] = pref[i-1] + a[i]

        low, high, res = 0, n, 0
        while low <= high:
            mid = (low + high) // 2
            fail = False

            # 前向检查
            cnt_front = 0
            for i in range(1, n//2 + 2):
                if i > mid:
                    cnt_front += 1
                    window = min(n, i * (d + 1))
                    if pref[window] < cnt_front * b:
                        fail = True
                        break

            # 后向检查
            if not fail:
                cnt_rear = 0
                for i in range(n, n//2, -1):
                    if (n - i + 1) > mid:
                        cnt_rear += 1
                        window = max(0, n - (n - i + 1) * (d + 1))
                        if (pref[n] - pref[window]) < cnt_rear * b:
                            fail = True
                            break

            if fail:
                low = mid + 1
            else:
                res = mid
                high = mid - 1
        return res

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        d = question_case['d']
        b = question_case['b']
        a = question_case['a']
        
        room_list = '\n'.join([f"房间{i+1}: {count}人" for i, count in enumerate(a)])
        return f"""## 宿舍查房优化问题

**背景**  
某宿舍有{n}个房间排成一行，每个房间应有{b}名学生。当前分布：  
{room_list}

**查房规则**  
1. 两位教官分别从两端开始逐间检查
2. 学生每次可移动≤{d}个房间或选择隐藏
3. 目标：使两位教官记录的违规房间数最大值最小

**输出格式**  
将答案放在[answer]标签内，如：[answer]2[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

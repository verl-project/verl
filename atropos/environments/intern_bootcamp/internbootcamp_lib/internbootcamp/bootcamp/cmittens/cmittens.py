"""# 

### 谜题描述
A Christmas party in city S. had n children. All children came in mittens. The mittens can be of different colors, but each child had the left and the right mitten of the same color. Let's say that the colors of the mittens are numbered with integers from 1 to m, and the children are numbered from 1 to n. Then the i-th child has both mittens of color ci.

The Party had Santa Claus ('Father Frost' in Russian), his granddaughter Snow Girl, the children danced around the richly decorated Christmas tree. In fact, everything was so bright and diverse that the children wanted to wear mittens of distinct colors. The children decided to swap the mittens so that each of them got one left and one right mitten in the end, and these two mittens were of distinct colors. All mittens are of the same size and fit all the children.

The children started exchanging the mittens haphazardly, but they couldn't reach the situation when each child has a pair of mittens of distinct colors. Vasily Petrov, the dad of one of the children, noted that in the general case the children's idea may turn out impossible. Besides, he is a mathematician and he came up with such scheme of distributing mittens that the number of children that have distinct-colored mittens was maximum. You task is to repeat his discovery. Note that the left and right mittens are different: each child must end up with one left and one right mitten.

Input

The first line contains two integers n and m — the number of the children and the number of possible mitten colors (1 ≤ n ≤ 5000, 1 ≤ m ≤ 100). The second line contains n integers c1, c2, ... cn, where ci is the color of the mittens of the i-th child (1 ≤ ci ≤ m).

Output

In the first line, print the maximum number of children who can end up with a distinct-colored pair of mittens. In the next n lines print the way the mittens can be distributed in this case. On the i-th of these lines print two space-separated integers: the color of the left and the color of the right mitten the i-th child will get. If there are multiple solutions, you can print any of them.

Examples

Input

6 3
1 3 2 2 1 1


Output

6
2 1
1 2
2 1
1 3
1 2
3 1


Input

4 2
1 2 1 1


Output

2
1 2
1 1
2 1
1 1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())
c = map(int, raw_input().split())

counts = [0] * m

for col in c:
    counts[col - 1] += 1

add = []
change = True

while change:
    change = False
    for i in xrange(m):
        if counts[i] > n / 2:
            counts[i] -= 1
            add.append(i + 1)
            n -= 1
            change = True
            break

print n

counts = [(counts[i], i + 1) for i in xrange(m)]

for d in add:
    print '{0} {0}'.format(d)

while n > 0:
    counts.sort()
    counts.reverse()
    while counts[-1][0] == 0:
        counts.pop()
    k = 0
    if n == 3:
        k = 3
    elif n >= 2:
        k = 2
    for i in xrange(k):
        counts[i] = (counts[i][0] - 1, counts[i][1])
        print '{0} {1}'.format(counts[i][1], counts[(i + 1) % k][1])
    n -= k
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Cmittensbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20, min_m=1, max_m=100):
        if not (1 <= min_n <= max_n <= 5000):
            raise ValueError("Invalid n range. Must be 1 <= min_n <= max_n <= 5000")
        if not (1 <= min_m <= max_m <= 100):
            raise ValueError("Invalid m range. Must be 1 <= min_m <= max_m <= 100")
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        m = random.randint(self.min_m, self.max_m)
        c = [random.randint(1, m) for _ in range(n)]
        return {'n': n, 'm': m, 'c': c}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        c_str = ' '.join(map(str, question_case['c']))
        prompt = f"""圣诞派对上，有{n}个孩子，他们每人戴着左右同一颜色的连指手套。手套颜色共有{m}种（编号1到{m}）。第i个孩子的原手套颜色为{c_str}。孩子们决定交换手套，使得每个孩子得到一只左手套和一只右手套，且两颜色不同。你的任务是求出能让最多孩子满足条件的方案，并输出该数目及分配方式。

输入格式：
第一行：{n} {m}
第二行：{c_str}

输出格式：
第一行输出最大满足条件的孩子数目k。
接下来的{n}行，每行两个整数，表示该孩子的左右手套颜色。若有多个解，输出任意一个即可。

注意：
- 交换后的左右手套必须来自原所有手套，即每种颜色的左右手套总数必须与原输入一致。
- 每个孩子必须获得左右各一只手套，颜色必须不同。
- 答案必须严格按输出格式，并包裹在[answer]和[/answer]标签内。例如：

[answer]
k
left_1 right_1
left_2 right_2
...
left_n right_n
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_block = matches[-1].strip()
        lines = [line.strip() for line in answer_block.split('\n') if line.strip()]
        if not lines:
            return None
        try:
            k = int(lines[0])
        except ValueError:
            return None
        assignments = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 2:
                return None
            try:
                left = int(parts[0])
                right = int(parts[1])
                assignments.append((left, right))
            except ValueError:
                return None
        return {'k': k, 'assignments': assignments}

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        try:
            k_user = solution['k']
            assignments = solution['assignments']
        except KeyError:
            return False

        n = identity['n']
        m = identity['m']
        c = identity['c']

        if len(assignments) != n:
            return False

        correct_k = cls.compute_max_k(n, m, c)
        if k_user != correct_k:
            return False

        # Validate color counts and distinct pairs
        original_counts = defaultdict(int)
        for color in c:
            original_counts[color] += 1

        left_counts = defaultdict(int)
        right_counts = defaultdict(int)
        valid_pairs = 0

        for left, right in assignments:
            if left == right:
                continue
            valid_pairs += 1
            left_counts[left] += 1
            right_counts[right] += 1

        if valid_pairs != k_user:
            return False

        # Check left/right color counts match original distribution
        for color, count in original_counts.items():
            if left_counts[color] != count or right_counts[color] != count:
                return False

        # Verify no extra colors introduced
        all_colors = set(original_counts.keys())
        if (set(left_counts.keys()) | set(right_counts.keys())) - all_colors:
            return False

        return True

    @classmethod
    def compute_max_k(cls, n, m, c):
        counts = [0] * m
        for col in c:
            counts[col - 1] += 1
        current_n = n
        change = True
        
        while change:
            change = False
            for i in range(m):
                if counts[i] > current_n // 2:
                    counts[i] -= 1
                    current_n -= 1
                    change = True
                    break
        return current_n

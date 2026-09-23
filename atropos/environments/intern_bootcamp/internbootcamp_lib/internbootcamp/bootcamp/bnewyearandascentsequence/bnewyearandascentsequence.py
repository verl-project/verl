"""# 

### 谜题描述
A sequence a = [a_1, a_2, …, a_l] of length l has an ascent if there exists a pair of indices (i, j) such that 1 ≤ i < j ≤ l and a_i < a_j. For example, the sequence [0, 2, 0, 2, 0] has an ascent because of the pair (1, 4), but the sequence [4, 3, 3, 3, 1] doesn't have an ascent.

Let's call a concatenation of sequences p and q the sequence that is obtained by writing down sequences p and q one right after another without changing the order. For example, the concatenation of the [0, 2, 0, 2, 0] and [4, 3, 3, 3, 1] is the sequence [0, 2, 0, 2, 0, 4, 3, 3, 3, 1]. The concatenation of sequences p and q is denoted as p+q.

Gyeonggeun thinks that sequences with ascents bring luck. Therefore, he wants to make many such sequences for the new year. Gyeonggeun has n sequences s_1, s_2, …, s_n which may have different lengths. 

Gyeonggeun will consider all n^2 pairs of sequences s_x and s_y (1 ≤ x, y ≤ n), and will check if its concatenation s_x + s_y has an ascent. Note that he may select the same sequence twice, and the order of selection matters.

Please count the number of pairs (x, y) of sequences s_1, s_2, …, s_n whose concatenation s_x + s_y contains an ascent.

Input

The first line contains the number n (1 ≤ n ≤ 100 000) denoting the number of sequences.

The next n lines contain the number l_i (1 ≤ l_i) denoting the length of s_i, followed by l_i integers s_{i, 1}, s_{i, 2}, …, s_{i, l_i} (0 ≤ s_{i, j} ≤ 10^6) denoting the sequence s_i. 

It is guaranteed that the sum of all l_i does not exceed 100 000.

Output

Print a single integer, the number of pairs of sequences whose concatenation has an ascent.

Examples

Input


5
1 1
1 1
1 2
1 4
1 3


Output


9


Input


3
4 2 0 2 0
6 9 9 8 8 7 7
1 6


Output


7


Input


10
3 62 24 39
1 17
1 99
1 60
1 64
1 30
2 79 29
2 20 73
2 85 37
1 100


Output


72

Note

For the first example, the following 9 arrays have an ascent: [1, 2], [1, 2], [1, 3], [1, 3], [1, 4], [1, 4], [2, 3], [2, 4], [3, 4]. Arrays with the same contents are counted as their occurences.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()

maximums = []
minimums = []


def add_max_and_min(array):
    minimum = array[0]
    maximum = array[0]
    flag = False

    for i in range(1, len(array)):
        if (array[i] < minimum):
            minimum = array[i]
        if (array[i] > maximum):
            maximum = array[i]
        if (array[i] > minimum):
            flag = True

    if (not flag):
        maximums.append(maximum)
        minimums.append(minimum)


for i in range(n):
    lines = map(int, raw_input().split())
    integers = lines[1:]
    add_max_and_min(integers)

maximums.sort()
minimums.sort()
length = len(maximums)
total = ((n - length) ** 2) + ((n - length) * length * 2)
count = 0
index = 0

while (count < length and index < length):
    if (minimums[index] < maximums[count] and index < length):
        index += 1
    else:
        total += index
        count += 1

total += (length - count) * index

print(total)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_answer(input_lines):
    n = int(input_lines[0])
    maximums = []
    minima = []
    for line in input_lines[1:]:
        parts = line.split()
        l_i = int(parts[0])
        array = list(map(int, parts[1:]))
        min_val = array[0]
        max_val = array[0]
        has_internal_ascent = False
        for num in array[1:]:
            if num < min_val:
                min_val = num
            if num > max_val:
                max_val = num
            if num > min_val:
                has_internal_ascent = True
        if not has_internal_ascent:
            maximums.append(max_val)
            minima.append(min_val)
    
    k = n - len(maximums)
    total = k * n * 2 - k * k  # (x in K, any y) + (any x, y in K) - duplicate
    
    maxima_sorted = sorted(maximums)
    minima_sorted = sorted(minima)
    m = len(maxima_sorted)
    
    ptr = 0
    for max_val in maxima_sorted:
        while ptr < m and minima_sorted[ptr] < max_val:
            ptr += 1
        total += ptr
    
    return total

class Bnewyearandascentsequencebootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = params
    
    def case_generator(self):
        # 参数处理
        max_n = self.params.get('max_n', 20)
        max_len = self.params.get('max_len', 10)
        value_range = self.params.get('value_range', (0, 100))
        
        n = random.randint(1, max_n)
        sequences = []
        has_ascent_count = 0
        
        # 生成多样化序列
        for _ in range(n):
            l_i = random.randint(1, max_len)
            seq = []
            has_ascent = False
            
            # 随机生成基础序列
            for _ in range(l_i):
                seq.append(random.randint(*value_range))
            
            # 确保至少30%的序列有内部上升
            if random.random() < 0.3:
                # 在随机位置插入上升对
                if l_i >= 2:
                    pos = random.randint(0, l_i-2)
                    seq[pos+1] = seq[pos] + random.randint(1, 5)
                    has_ascent = True
                # 添加全局上升特征
                if random.choice([True, False]):
                    seq.sort()
                    has_ascent = True
                else:
                    # 添加局部上升
                    for i in range(1, len(seq)):
                        if seq[i] > seq[i-1]:
                            has_ascent = True
                            break
            
            # 验证实际ascent情况
            actual_ascent = False
            min_so_far = seq[0]
            for num in seq[1:]:
                if num > min_so_far:
                    actual_ascent = True
                    break
                min_so_far = min(min_so_far, num)
            
            sequences.append(seq)
            if actual_ascent:
                has_ascent_count += 1
        
        # 构造输入数据
        input_lines = [str(n)]
        for seq in sequences:
            input_lines.append(f"{len(seq)} {' '.join(map(str, seq))}")
        
        return {
            "n": n,
            "sequences": [[len(seq)] + seq for seq in sequences],
            "expected_answer": compute_answer(input_lines)
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case["n"]
        sequences = question_case["sequences"]
        problem_input = f"{n}\n" + "\n".join(" ".join(map(str, seq)) for seq in sequences)
        
        return f"""You need to solve a sequence pairs counting problem. Analyze the input and output the final answer within [answer] tags.

Problem Rules:
1. A sequence has an ascent if any later element is greater than an earlier element
2. Count all ordered pairs (x,y) where concatenating x and y creates an ascent

Input Format:
- First line: n (number of sequences)
- Next n lines: l_i followed by l_i integers

Output Format:
- Single integer: number of valid pairs

Example Input:
3
4 2 0 2 0
6 9 9 8 8 7 7
1 6

Example Output:
7

Now solve this input:
{problem_input}

Put your final answer within [answer][/answer] tags."""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity["expected_answer"]

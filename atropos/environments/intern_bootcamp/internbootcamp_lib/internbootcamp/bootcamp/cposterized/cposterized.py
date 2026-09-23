"""# 

### 谜题描述
Professor Ibrahim has prepared the final homework for his algorithm’s class. He asked his students to implement the Posterization Image Filter.

Their algorithm will be tested on an array of integers, where the i-th integer represents the color of the i-th pixel in the image. The image is in black and white, therefore the color of each pixel will be an integer between 0 and 255 (inclusive).

To implement the filter, students are required to divide the black and white color range [0, 255] into groups of consecutive colors, and select one color in each group to be the group’s key. In order to preserve image details, the size of a group must not be greater than k, and each color should belong to exactly one group.

Finally, the students will replace the color of each pixel in the array with that color’s assigned group key.

To better understand the effect, here is an image of a basking turtle where the Posterization Filter was applied with increasing k to the right. 

<image>

To make the process of checking the final answer easier, Professor Ibrahim wants students to divide the groups and assign the keys in a way that produces the lexicographically smallest possible array.

Input

The first line of input contains two integers n and k (1 ≤ n ≤ 10^5, 1 ≤ k ≤ 256), the number of pixels in the image, and the maximum size of a group, respectively.

The second line contains n integers p_1, p_2, ..., p_n (0 ≤ p_i ≤ 255), where p_i is the color of the i-th pixel.

Output

Print n space-separated integers; the lexicographically smallest possible array that represents the image after applying the Posterization filter.

Examples

Input

4 3
2 14 3 4


Output

0 12 3 3


Input

5 2
0 2 1 255 254


Output

0 1 1 254 254

Note

One possible way to group colors and assign keys for the first sample:

Color 2 belongs to the group [0,2], with group key 0.

Color 14 belongs to the group [12,14], with group key 12.

Colors 3 and 4 belong to group [3, 5], with group key 3.

Other groups won't affect the result so they are not listed here.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k = map(int, raw_input().strip().split())
p = map(int, raw_input().strip().split())

ans = [0] * n

assigned = [0 for i in xrange(256)]
key = [i for i in xrange(256)]

for i in xrange(n):
    cur = p[i]
    if assigned[cur]:
        ans[i] = str(key[cur])
        continue

    foundkey = 0
    no1 = 1
    rep = -1
    repb = -1
    for it in xrange(cur, max(-1, cur - k), -1):
        if assigned[it] == 1: no1 = 0
        if assigned[it] == 1 and key[it] == it:
            foundkey = 1
            rep = it
            break
        elif assigned[it] == 0 and no1 == 1:
            repb = it

    if not foundkey:
        for it in xrange(repb, cur + 1):
            assigned[it] = 1
            key[it] = repb

        ans[i] = str(repb)
    else:
        for it in xrange(rep, cur + 1):
            assigned[it] = 1
            key[it] = rep

        ans[i] = str(rep)

print ' '.join(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def generate_answer(n, k, p):
    assigned = [0] * 256
    key = list(range(256))
    ans = [0] * n

    for i in range(n):
        cur = p[i]
        if assigned[cur]:
            ans[i] = key[cur]
            continue

        foundkey = False
        no1 = True
        rep = -1
        repb = -1
        start_search = max(0, cur - k + 1)
        
        # 扫描可能的区间
        for it in range(cur, start_search - 1, -1):
            if assigned[it]:
                no1 = False
                if key[it] == it:  # 有效锚点
                    foundkey = True
                    rep = it
                    break
            elif no1:
                repb = it

        if not foundkey:
            # 处理255边界
            group_start = max(0, repb)
            group_end = min(255, group_start + k - 1)
            for it in range(group_start, group_end + 1):
                assigned[it] = 1
                key[it] = group_start
            ans[i] = group_start
        else:
            group_end = min(255, rep + k - 1)
            for it in range(rep, group_end + 1):
                assigned[it] = 1
                key[it] = rep
            ans[i] = rep

    return ans

class Cposterizedbootcamp(Basebootcamp):
    def __init__(self, max_n=10**5, min_n=1, max_k=256, min_k=1):
        self.max_n = max_n
        self.min_n = min_n
        self.max_k = max_k
        self.min_k = min_k
    
    def case_generator(self):
        # 生成关键测试模式
        if random.random() < 0.2:
            # 模式1：全范围测试 (k=256)
            n = random.randint(100, 1000)
            k = 256
            p = [random.randint(0, 255) for _ in range(n)]
        elif random.random() < 0.3:
            # 模式2：最小k测试 (k=1)
            n = random.randint(1, 1000)
            k = 1
            p = [random.randint(0, 255) for _ in range(n)]
        else:
            # 常规随机测试
            n = random.randint(self.min_n, self.max_n)
            k = random.randint(self.min_k, min(self.max_k, 256))
            p = [random.randint(0, 255) for _ in range(n)]
        
        # 强制包含边界值
        if n >= 2:
            p[0] = 0
            p[-1] = 255
        
        ans = generate_answer(n, k, p)
        return {
            'n': n,
            'k': k,
            'p': p,
            'ans': ans
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = f"{question_case['n']} {question_case['k']}\n{' '.join(map(str, question_case['p']))}"
        prompt = f"""Implement the posterization filter to produce the lexicographically smallest array.

**Key Rules:**
1. Groups must be consecutive colors with size ≤ {question_case['k']}
2. Each group selects ONE key color within its range
3. ALL occurrences of colors in a group are replaced by the key
4. The result array must be the lex smallest possible when comparing element-wise from left to right

**Input:**
{input_lines}

**Output Format:**
Exactly {question_case['n']} integers between [answer] and [/answer], e.g.:
[answer]0 1 2 3[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\][\n\s]*((?:-?\d+[\s\n]*)+)[\n\s]*\[/answer\]', output)
        if not matches:
            return None
        try:
            last_match = matches[-1].replace('\n', ' ').strip()
            return list(map(int, last_match.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 严格验证数组长度
            if len(solution) != identity['n']:
                return False
            # 类型一致性检查
            if not all(isinstance(x, int) for x in solution):
                return False
            return solution == identity['ans']
        except:
            return False

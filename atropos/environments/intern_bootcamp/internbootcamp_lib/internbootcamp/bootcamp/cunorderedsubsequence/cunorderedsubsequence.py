"""# 

### 谜题描述
The sequence is called ordered if it is non-decreasing or non-increasing. For example, sequnces [3, 1, 1, 0] and [1, 2, 3, 100] are ordered, but the sequence [1, 3, 3, 1] is not. You are given a sequence of numbers. You are to find it's shortest subsequence which is not ordered.

A subsequence is a sequence that can be derived from the given sequence by deleting zero or more elements without changing the order of the remaining elements.

Input

The first line of the input contains one integer n (1 ≤ n ≤ 105). The second line contains n space-separated integers — the given sequence. All numbers in this sequence do not exceed 106 by absolute value.

Output

If the given sequence does not contain any unordered subsequences, output 0. Otherwise, output the length k of the shortest such subsequence. Then output k integers from the range [1..n] — indexes of the elements of this subsequence. If there are several solutions, output any of them.

Examples

Input

5
67 499 600 42 23


Output

3
1 3 5


Input

3
1 2 3


Output

0


Input

3
2 3 1


Output

3
1 2 3

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
	n = input()
	a = map(int, raw_input().split())
	s = [(1, a[0])]
	for i in xrange(1, n):
		if a[i] != s[-1][1]: s.append((i + 1, a[i]))
	for i in xrange(1, len(s) - 1):
		if s[i - 1][1] < s[i][1] > s[i + 1][1] or s[i - 1][1] > s[i][1] < s[i + 1][1]:
			print 3
			print s[i - 1][0], s[i][0], s[i + 1][0]
			return 
	print 0
 
if __name__ == \"__main__\": main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cunorderedsubsequencebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)  # 默认n为5
        self.min_n = params.get('min_n', 3)  # 最小n为3
        self.max_n = params.get('max_n', 1000)  # 最大n为1000

    def case_generator(self):
        while True:
            n = random.randint(self.min_n, self.max_n)
            # 确保n至少为3
            n = max(n, self.min_n)
            # 生成初始有序序列，递增或递减的概率各占一半
            if random.choice([True, False]):
                sequence = list(range(1, n + 1))
            else:
                sequence = list(range(n, 0, -1))
            
            # 随机选择一个位置i，确保在有效范围内
            i = random.randint(0, n - 3)
            # 判断当前序列是递增还是递减
            if sequence[i] < sequence[i + 1]:
                # 递增，形成一个峰
                new_val = sequence[i + 1] + random.randint(1, 100)
                sequence[i + 1] = new_val
            else:
                # 递减，形成一个谷
                new_val = sequence[i + 1] - random.randint(1, 100)
                sequence[i + 1] = new_val
            
            # 压缩序列，处理相同元素
            s = []
            for idx, num in enumerate(sequence):
                if not s or s[-1][1] != num:
                    s.append((idx + 1, num))  # 使用1-based索引
            
            # 检查是否有三元组
            correct_length = 0
            correct_indices = []
            for j in range(1, len(s) - 1):
                a, b, c = s[j - 1][1], s[j][1], s[j + 1][1]
                if (a < b and b > c) or (a > b and b < c):
                    correct_length = 3
                    correct_indices = [s[j - 1][0], s[j][0], s[j + 1][0]]
                    break
            if correct_length == 3:
                # 记录正确答案
                case = {
                    "sequence": sequence,
                    "n": n,
                    "correct_answer_length": correct_length,
                    "correct_answer_indices": correct_indices
                }
                return case
            else:
                # 继续生成，直到找到一个有解的序列
                continue

    @staticmethod
    def prompt_func(question_case):
        sequence = question_case['sequence']
        n = question_case['n']
        prompt = f"""你是一个聪明的解谜者，请解决以下问题：

给定一个长度为{n}的数字序列：
{sequence}

你的任务是找到其中最短的无序子序列。无序子序列指的是既不是非递增也不是非递减的子序列。子序列可以通过删除一些元素得到，但顺序必须保持不变。

规则说明：
1. 如果整个序列是有序的（非递增或非递减），则输出0。
2. 否则，输出最短无序子序列的长度k，接着输出k个元素的索引（1-based）。
3. 如果有多个解，输出任意一个即可。

例如：
输入：
5
67 499 600 42 23

输出：
3
1 3 5

请将答案放在[answer]标签中，格式如下：
如果无解，输出：
[answer]
0
[/answer]

否则，输出：
[answer]
3
1 3 5
[/answer]

现在，请解决以下问题：

给定序列：{sequence}

输出格式：
[answer]
...
[/answer]
"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        answer_str = matches[-1].strip()
        lines = answer_str.split('\n')
        if not lines:
            return None
        # 第一行是k
        k_str = lines[0].strip()
        if not k_str:
            return None
        try:
            k = int(k_str)
        except:
            return None
        if k == 0:
            return 0
        else:
            if len(lines) < 2:
                return None
            indices_str = lines[1].strip()
            indices = list(map(int, indices_str.split()))
            if len(indices) != k:
                return None
            return {'k': k, 'indices': indices}

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        correct_length = identity['correct_answer_length']
        correct_indices = identity['correct_answer_indices']
        if correct_length == 0:
            return solution == 0
        else:
            if isinstance(solution, dict) and solution['k'] == correct_length:
                sequence = identity['sequence']
                indices = solution['indices']
                # 转换为0-based
                subseq = [sequence[i - 1] for i in indices]
                # 检查是否无序
                is_ordered = False
                # 检查是否非递增
                non_increasing = True
                for i in range(len(subseq) - 1):
                    if subseq[i] < subseq[i + 1]:
                        non_increasing = False
                        break
                if non_increasing:
                    is_ordered = True
                else:
                    # 检查是否非递减
                    non_decreasing = True
                    for i in range(len(subseq) - 1):
                        if subseq[i] > subseq[i + 1]:
                            non_decreasing = False
                            break
                    if non_decreasing:
                        is_ordered = True
                # 如果子序列是无序的，则正确
                return not is_ordered
            else:
                return False

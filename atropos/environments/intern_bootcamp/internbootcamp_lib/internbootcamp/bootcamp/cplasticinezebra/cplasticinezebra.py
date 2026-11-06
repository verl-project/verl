"""# 

### 谜题描述
Is there anything better than going to the zoo after a tiresome week at work? No wonder Grisha feels the same while spending the entire weekend accompanied by pretty striped zebras. 

Inspired by this adventure and an accidentally found plasticine pack (represented as a sequence of black and white stripes), Grisha now wants to select several consequent (contiguous) pieces of alternating colors to create a zebra. Let's call the number of selected pieces the length of the zebra.

Before assembling the zebra Grisha can make the following operation 0 or more times. He splits the sequence in some place into two parts, then reverses each of them and sticks them together again. For example, if Grisha has pieces in the order \"bwbbw\" (here 'b' denotes a black strip, and 'w' denotes a white strip), then he can split the sequence as bw|bbw (here the vertical bar represents the cut), reverse both parts and obtain \"wbwbb\".

Determine the maximum possible length of the zebra that Grisha can produce.

Input

The only line contains a string s (1 ≤ |s| ≤ 10^5, where |s| denotes the length of the string s) comprised of lowercase English letters 'b' and 'w' only, where 'w' denotes a white piece and 'b' denotes a black piece.

Output

Print a single integer — the maximum possible zebra length.

Examples

Input

bwwwbwwbw


Output

5


Input

bwwbwwb


Output

3

Note

In the first example one of the possible sequence of operations is bwwwbww|bw → w|wbwwwbwb → wbwbwwwbw, that gives the answer equal to 5.

In the second example no operation can increase the answer.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# input_str = raw_input()
# input_str_double = input_str+input_str
# len_str_double = len(input_str_double)
# ans =0
# j=0
# for i in range(1, len_str_double+1):
#     i = j
#     for j in range(i+1, len_str_double+1):
#         if input_str_double[j] == input_str_double[j-1]:
#             break
#     ans= max(ans, j-i)
# print min(ans, len_str_double/2)

input_str = raw_input()
len_input_str = len(input_str)
input_str_double = [0]*(2*len_input_str)
# print  input_str, len_input_str
for i in range(0, len_input_str):
    input_str_double[i] = input_str[i]
    input_str_double[i + len_input_str] = input_str[i]

# print  input_str_double[0]
ans1 = 0
ans2 = 0
id = -1
if input_str_double[0] == 'w':
    id = -1
else:
    id = 1
for i in range(1, len_input_str * 2):
    if input_str_double[i] == 'w':
        if id == 1:
            ans1 += 1
            id = -1
        else:
            if input_str_double[i] == 'b':
                ans1 += 1
                id = -1
            else:
                ans1 = 1
        ans2 = max(ans1, ans2)
    else:
        if id == -1:
            ans1 += 1
            id = 1
        else:
            if input_str_double[i] == 'w':
                ans1 += 1
                id = 1
            else:
                ans1 = 1
        ans2 = max(ans1, ans2)

if ans2 > len_input_str:
    ans2 = len_input_str
print ans2
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cplasticinezebrabootcamp(Basebootcamp):
    def __init__(self, min_length=1, max_length=20):
        """
        初始化训练场参数，支持生成长度1到20的测试案例
        """
        self.min_length = min_length
        self.max_length = max_length
    
    def case_generator(self):
        """
        生成包含全交替、全同、混合模式的三类测试案例
        """
        pattern_type = random.choices(
            ['alternating', 'uniform', 'segmented'],
            weights=[0.3, 0.2, 0.5],  # 增加混合模式概率
            k=1
        )[0]

        length = random.randint(self.min_length, self.max_length)
        
        if pattern_type == 'alternating':
            start = random.choice(['b', 'w'])
            s = [start if i%2 == 0 else 'w' if start == 'b' else 'b' for i in range(length)]
        elif pattern_type == 'uniform':
            s = [random.choice(['b', 'w'])] * length
        else:
            # 生成分段交替模式（如 bbwwwbbww）
            s = []
            current = random.choice(['b', 'w'])
            while len(s) < length:
                # 随机生成1-3个相同字符
                seg_len = random.randint(1, 3)
                s.extend([current] * seg_len)
                current = 'w' if current == 'b' else 'b'
        
        s = ''.join(s[:length])
        return {'s': s}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        return f"""请根据以下规则解决斑马条纹问题：
1. 允许执行任意次数的分割反转操作（分割点任意选）
2. 每次操作将字符串分成左右两部分，分别反转后拼接
3. 目标是通过操作获得最长的连续交替颜色序列（相邻字符不同）

输入字符串：{s}

思考步骤：
1. 分析原始字符串的交替模式
2. 考虑如何通过分割反转合并不同的交替段
3. 特别注意字符串的循环特性可能带来的最长序列

将最终答案放在[answer]标签内，如：[answer]5[/answer]。"""

    @staticmethod
    def extract_output(output):
        # 严格匹配整数答案，过滤非数字内容
        answers = re.findall(r'\[answer\s*\](.*?)\[/answer\s*\]', output, re.DOTALL)
        if not answers:
            return None
        try:
            return int(answers[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        s = identity['s']
        return solution == cls._calculate_max_zebra(s)
    
    @staticmethod
    def _calculate_max_zebra(s):
        """
        优化后的验证算法，处理边界情况
        """
        if not s:
            return 0
        
        max_len = current = 1
        doubled = s * 2
        
        # 遍历双倍字符串寻找最大交替序列
        for i in range(1, len(doubled)):
            if doubled[i] != doubled[i-1]:
                current += 1
                max_len = max(max_len, current)
            else:
                current = 1
        
        # 最终结果不能超过原字符串长度
        return min(max_len, len(s))

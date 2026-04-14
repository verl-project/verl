"""# 

### 谜题描述
A boy named Ayrat lives on planet AMI-1511. Each inhabitant of this planet has a talent. Specifically, Ayrat loves running, moreover, just running is not enough for him. He is dreaming of making running a real art.

First, he wants to construct the running track with coating t. On planet AMI-1511 the coating of the track is the sequence of colored blocks, where each block is denoted as the small English letter. Therefore, every coating can be treated as a string.

Unfortunately, blocks aren't freely sold to non-business customers, but Ayrat found an infinite number of coatings s. Also, he has scissors and glue. Ayrat is going to buy some coatings s, then cut out from each of them exactly one continuous piece (substring) and glue it to the end of his track coating. Moreover, he may choose to flip this block before glueing it. Ayrat want's to know the minimum number of coating s he needs to buy in order to get the coating t for his running track. Of course, he also want's to know some way to achieve the answer.

Input

First line of the input contains the string s — the coating that is present in the shop. Second line contains the string t — the coating Ayrat wants to obtain. Both strings are non-empty, consist of only small English letters and their length doesn't exceed 2100.

Output

The first line should contain the minimum needed number of coatings n or -1 if it's impossible to create the desired coating.

If the answer is not -1, then the following n lines should contain two integers xi and yi — numbers of ending blocks in the corresponding piece. If xi ≤ yi then this piece is used in the regular order, and if xi > yi piece is used in the reversed order. Print the pieces in the order they should be glued to get the string t.

Examples

Input

abc
cbaabc


Output

2
3 1
1 3


Input

aaabrytaaa
ayrat


Output

3
1 1
6 5
8 7


Input

ami
no


Output

-1

Note

In the first sample string \"cbaabc\" = \"cba\" + \"abc\".

In the second sample: \"ayrat\" = \"a\" + \"yr\" + \"at\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s = raw_input()
t = raw_input()
rev = s[::-1]
t=t+\"#\"
ans=[]
temp=t[0]
n=len(s)
for i in xrange(1,len(t)):
    cur = temp+t[i]
    if cur in s or cur in rev:
        temp=cur
    else:
        if temp in s:
            start = s.find(temp)
            ans.append((start+1,start+len(temp)))
        elif temp in rev:
            start = rev.find(temp)
            ans.append((n-start,n-start-len(temp)+1))
        else :
            print -1
            exit(0)
        temp=t[i]
print len(ans)
for i in xrange(len(ans)):
    print ans[i][0],ans[i][1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from string import ascii_lowercase
from bootcamp import Basebootcamp

class Crunningtrackbootcamp(Basebootcamp):
    def __init__(self, s_min_length=3, s_max_length=10, allow_unsolvable=False, unsolvable_prob=0.2):
        self.s_min_length = s_min_length
        self.s_max_length = s_max_length
        self.allow_unsolvable = allow_unsolvable
        self.unsolvable_prob = unsolvable_prob
    
    def case_generator(self):
        # 生成满足条件的s
        while True:
            s = ''.join(random.choices(ascii_lowercase, 
                     k=random.randint(self.s_min_length, self.s_max_length)))
            if len(set(s)) >= 2:  # 确保s至少有2个不同字符
                break
        
        # 生成可解的t案例
        if not self.allow_unsolvable or random.random() > self.unsolvable_prob:
            for _ in range(100):  # 最多尝试100次生成合法案例
                # 基于s生成保证可解的t
                t_parts = []
                num_parts = random.randint(1, 5)
                
                for _ in range(num_parts):
                    # 确保每次切割都有效
                    max_len = len(s) - 1 if len(s) > 1 else 1
                    part_len = random.randint(1, max_len)
                    start = random.randint(0, len(s)-part_len)
                    reverse = random.random() < 0.5
                    t_parts.append(s[start:start+part_len][::-1] if reverse else s[start:start+part_len])
                
                t = ''.join(t_parts)
                ans_ref = self._solve_reference(s, t)
                if ans_ref != -1 and len(ans_ref) == num_parts:
                    return {'s': s, 't': t, 'ans_ref': ans_ref}
            
            # 生成保证可解的默认案例
            default_s = 'abc'
            default_t = default_s[::-1] + default_s  # 'cbaabc'
            ans_ref = self._solve_reference(default_s, default_t)
            return {'s': default_s, 't': default_t, 'ans_ref': ans_ref}
        
        # 生成不可解案例
        else:
            # 策略1：插入非法字符
            if random.random() < 0.5:
                invalid_char = random.choice([c for c in ascii_lowercase if c not in s])
                return {'s': s, 't': invalid_char*(len(s)+1), 'ans_ref': -1}
            
            # 策略2：构造无法分割的合法t
            for _ in range(100):
                # 生成交替模式如abababc (s长度超过3时)
                t = (s[:2] * 3)[:random.randint(5, 15)]
                if self._solve_reference(s, t) == -1:
                    return {'s': s, 't': t, 'ans_ref': -1}
            
            # 策略3：强制构造不可解案例
            critical_char = s[0]
            t = critical_char * (len(s)*2)  # 超过s中连续出现的最大长度
            return {'s': s, 't': t, 'ans_ref': -1}
    
    @staticmethod
    def _solve_reference(s, t):
        rev_s = s[::-1]
        result = []
        current = t[0]
        
        for c in t[1:]:
            test_str = current + c
            if test_str in s or test_str in rev_s:
                current = test_str
            else:
                # 寻找最长匹配
                found = False
                for l in range(len(current), 0, -1):
                    substr = current[:l]
                    if substr in s:
                        start = s.index(substr)
                        result.append((start+1, start+l))
                        current = current[l:] + c
                        found = True
                        break
                    elif substr in rev_s:
                        start = rev_s.index(substr)
                        result.append((len(s)-start, len(s)-start-l+1))
                        current = current[l:] + c
                        found = True
                        break
                if not found:
                    return -1
        
        # 处理最后剩余部分
        if current:
            if current in s:
                start = s.index(current)
                result.append((start+1, start+len(current)))
            elif current in rev_s:
                start = rev_s.index(current)
                result.append((len(s)-start, len(s)-start-len(current)+1))
            else:
                return -1
        
        return result if result else -1
    
    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        t = question_case['t']
        return f"""你需要帮助Ayrat用最少段商店涂层s（可反转）拼接出目标涂层t。
输入：
s = {s}
t = {t}

输出要求：
首行为最小段数n（无法完成输出-1）
随后每行给出xi yi（1-based索引，xi≤yi正向使用，xi>yi反向使用）
将最终答案置于[answer]和[/answer]之间。示例：
[answer]
2
3 1
1 3
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强匹配模式，允许各种空白字符
        matches = re.findall(r'\[answer\][\s]*(.*?)[\s]*\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
        
        try:
            # 处理首行可能的多余字符（如-1后面带说明）
            first_line = lines[0].split()[0]
            if first_line == '-1':
                return -1
            
            n = int(first_line)
            if len(lines) < n+1 or n <= 0:
                return None
            
            parts = []
            for line in lines[1:n+1]:
                # 允许各种分隔符（空格、逗号等）
                nums = re.findall(r'-?\d+', line)
                if len(nums) != 2:
                    return None
                x, y = map(int, nums)
                parts.append((x, y))
            return parts
        except Exception as e:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证不可解情况
        if identity['ans_ref'] == -1:
            return solution == -1
        
        # 验证可解情况
        if solution == -1:
            return False
        
        # 类型检查
        if not isinstance(solution, list) or len(solution) != len(identity['ans_ref']):
            return False
        
        s = identity['s']
        t = identity['t']
        reconstructed = []
        
        for x, y in solution:
            # 坐标有效性检查
            if not (1 <= x <= len(s)) or not (1 <= y <= len(s)):
                return False
            if x <= y:
                # 正向子串检查
                if s[x-1:y] not in s:
                    return False
                reconstructed.append(s[x-1:y])
            else:
                # 反向子串检查
                rev_sub = s[y-1:x][::-1]
                if rev_sub not in s:
                    return False
                reconstructed.append(rev_sub)
        
        return ''.join(reconstructed) == t

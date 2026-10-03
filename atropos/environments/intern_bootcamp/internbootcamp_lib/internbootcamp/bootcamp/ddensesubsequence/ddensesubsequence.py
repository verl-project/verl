"""# 

### 谜题描述
You are given a string s, consisting of lowercase English letters, and the integer m.

One should choose some symbols from the given string so that any contiguous subsegment of length m has at least one selected symbol. Note that here we choose positions of symbols, not the symbols themselves.

Then one uses the chosen symbols to form a new string. All symbols from the chosen position should be used, but we are allowed to rearrange them in any order.

Formally, we choose a subsequence of indices 1 ≤ i1 < i2 < ... < it ≤ |s|. The selected sequence must meet the following condition: for every j such that 1 ≤ j ≤ |s| - m + 1, there must be at least one selected index that belongs to the segment [j, j + m - 1], i.e. there should exist a k from 1 to t, such that j ≤ ik ≤ j + m - 1.

Then we take any permutation p of the selected indices and form a new string sip1sip2... sipt.

Find the lexicographically smallest string, that can be obtained using this procedure.

Input

The first line of the input contains a single integer m (1 ≤ m ≤ 100 000).

The second line contains the string s consisting of lowercase English letters. It is guaranteed that this string is non-empty and its length doesn't exceed 100 000. It is also guaranteed that the number m doesn't exceed the length of the string s.

Output

Print the single line containing the lexicographically smallest string, that can be obtained using the procedure described above.

Examples

Input

3
cbabc


Output

a


Input

2
abcab


Output

aab


Input

3
bcabcbaccba


Output

aaabb

Note

In the first sample, one can choose the subsequence {3} and form a string \"a\".

In the second sample, one can choose the subsequence {1, 2, 4} (symbols on this positions are 'a', 'b' and 'a') and rearrange the chosen symbols to form a string \"aab\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
# coding: utf-8

if __name__ == '__main__':
    import sys
    f = sys.stdin
    #f = open('/home/ilya/opt/programming/tasks/724D.txt')

    if False:
        import StringIO
        f = StringIO.StringIO(\"\"\"3
bcabcbaccba\"\"\")

    if False:
        import StringIO
        f = StringIO.StringIO(\"\"\"2
abcab\"\"\")

    if False:
        import StringIO
        f = StringIO.StringIO(\"\"\"5
wjjdqawypvtgrncmqvcsergermprauyevcegjtcrrblkwiugrcjfpjyxngyryxntauxlouvwgjzpsuxyxvhavgezwtuzknetdibv\"\"\")

    if False:
        import StringIO
        f = StringIO.StringIO(\"\"\"5
addcdbddddbddddcc\"\"\")

    if False:
        import StringIO
        f = StringIO.StringIO(\"\"\"1
addcdbddddbddddcc\"\"\")

    def read_int_line():
        return map(int, next(f).split())

    def getline():
        return next(f).rstrip('\n\r')

    m = read_int_line()[0]
    s = getline()

    n = len(s)

    freq_lst = []
    def append_freq(cur_char, cnt):
        freq_lst.append((cur_char, cnt))

    cur_char = None
    cnt = 0
    for c in sorted(s):
        if c != cur_char:
            if cur_char:
                append_freq(cur_char, cnt)
            cnt = 1
            cur_char = c
        else:
            cnt += 1
    if cnt:
        append_freq(cur_char, cnt)

    def print_subrange(c, cnt):
        sys.stdout.write(\"\".join(c for i in xrange(cnt)))

    for char, freq in freq_lst:
        cnt = 0
        last, last2 = -1, -1
        s_ok = True
        for i, c in enumerate(s):
            if c < char:
                last = last2 = i
            else:
                if c == char:
                    last2 = i

                if i - last == m:
                    if last2 > last:
                        cnt += 1
                        last = last2
                    else:
                        s_ok = False
                        break

        if s_ok:
            print_subrange(char, cnt)
            break
        else:
            print_subrange(char, freq)

    print
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

def solve(m, s):
    n = len(s)
    if n == 0 or m == 0:
        return ""
    
    # Frequency list generation
    sorted_chars = sorted(s)
    freq = []
    current_char = sorted_chars[0]
    count = 1
    
    for c in sorted_chars[1:]:
        if c == current_char:
            count += 1
        else:
            freq.append((current_char, count))
            current_char = c
            count = 1
    freq.append((current_char, count))
    
    # Find minimal solution
    for idx, (char, total) in enumerate(freq):
        required = 0
        last_covered = -1
        last_candidate = -1
        valid = True
        
        for i in range(n):
            if s[i] < char:
                last_covered = i
                last_candidate = i
            elif s[i] == char:
                last_candidate = i
            
            # Check window violation
            if i - last_covered >= m:
                if last_candidate > last_covered:
                    required += 1
                    last_covered = last_candidate
                else:
                    valid = False
                    break
        
        # Final check for the last window
        if valid and (n - last_covered) > m:
            valid = False
        
        if valid:
            # Calculate required count
            min_chars = []
            for c, _ in freq[:idx+1]:
                if c < char:
                    min_chars.append(c)
            return char * required + ''.join(sorted(min_chars))
        else:
            continue
    
    # Fallback to all smallest characters
    return ''.join(sorted(s))

class Ddensesubsequencebootcamp(Basebootcamp):
    def __init__(self, max_s_length=20, test_edge_cases=True):
        super().__init__()
        self.max_s_length = max_s_length
        self.test_edge_cases = test_edge_cases
        self.edge_cases = [
            {'m': 1, 's': 'abcde'},        # m=1必须全选
            {'m': 5, 's': 'aaaaa'},        # 全相同字符
            {'m': 3, 's': 'abababa'},      # 交错模式
            {'m': 4, 's': 'aabbaacc'},     # 重复模式
            {'m': 2, 's': 'zxyx'},         # 包含局部最小值
            {'m': 5, 's': 'edcba'},        # 递减序列
            {'m': 3, 's': 'abb'}
        ]
    
    def case_generator(self):
        if self.test_edge_cases and random.random() < 0.7:
            case = random.choice(self.edge_cases)
            m = case['m']
            s = case['s']
        else:
            len_s = random.randint(m_min := 1, self.max_s_length)
            m = random.randint(1, len_s)
            s = ''.join(random.choices(string.ascii_lowercase, k=len_s))
        
        # 确保m不超过s长度
        m = min(m, len(s))
        return {
            'm': m,
            's': s,
            'expected': solve(m, s)
        }
    
    @staticmethod
    def prompt_func(question_case):
        return f"""Given string s = "{question_case['s']}" and m = {question_case['m']}, find the lex smallest string by selecting positions that cover all m-length windows. Put your final answer between [answer] and [/answer]. Example: [answer]abc[/answer]"""
    
    @staticmethod
    def extract_output(output):
        # 多模式匹配：支持<!-- answer -->格式和不同大小写
        patterns = [
            r'\[answer\](.*?)\[/answer\]',    # 标准格式
            r'<!-- answer:?(.*?)-->',          # HTML注释格式
            r'answer:\s*(\S+)'                 # 简单前缀格式
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
            if matches:
                clean = ''.join(filter(str.isalpha, matches[-1])).lower()
                if clean:
                    return clean
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 生成标准答案
        expected = identity['expected'].lower().strip()
        # 处理空答案情况
        if not expected:
            return solution == ''
        # 允许字符顺序不同但排序后相同
        return sorted(solution.lower()) == sorted(expected)

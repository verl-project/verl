"""# 

### 谜题描述
One popular website developed an unusual username editing procedure. One can change the username only by deleting some characters from it: to change the current name s, a user can pick number p and character c and delete the p-th occurrence of character c from the name. After the user changed his name, he can't undo the change.

For example, one can change name \"arca\" by removing the second occurrence of character \"a\" to get \"arc\". 

Polycarpus learned that some user initially registered under nickname t, where t is a concatenation of k copies of string s. Also, Polycarpus knows the sequence of this user's name changes. Help Polycarpus figure out the user's final name.

Input

The first line contains an integer k (1 ≤ k ≤ 2000). The second line contains a non-empty string s, consisting of lowercase Latin letters, at most 100 characters long. The third line contains an integer n (0 ≤ n ≤ 20000) — the number of username changes. Each of the next n lines contains the actual changes, one per line. The changes are written as \"pi ci\" (without the quotes), where pi (1 ≤ pi ≤ 200000) is the number of occurrences of letter ci, ci is a lowercase Latin letter. It is guaranteed that the operations are correct, that is, the letter to be deleted always exists, and after all operations not all letters are deleted from the name. The letters' occurrences are numbered starting from 1.

Output

Print a single string — the user's final name after all changes are applied to it.

Examples

Input

2
bac
3
2 a
1 b
2 c


Output

acb


Input

1
abacaba
4
1 a
1 a
1 c
2 b


Output

baa

Note

Let's consider the first sample. Initially we have name \"bacbac\"; the first operation transforms it into \"bacbc\", the second one — to \"acbc\", and finally, the third one transforms it into \"acb\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
k = int(raw_input())
s = list(raw_input()) * k
h = [[] for _ in xrange(27)]
for i in xrange(len(s)):
    h[ord(s[i]) - ord('a')].append(i)
for _ in xrange(int(raw_input())):
    i, c = raw_input().split()
    s[h[ord(c) - ord('a')].pop(int(i) - 1)] = ''
print(''.join(s))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import string
from collections import defaultdict
from bootcamp import Basebootcamp

class Cstringmanipulation10bootcamp(Basebootcamp):
    def __init__(self, **params):
        self.k_min = params.get('k_min', 1)
        self.k_max = params.get('k_max', 100)  # 降低默认最大值便于测试
        self.s_min_len = params.get('s_min_len', 3)  # 增加最小长度确保操作多样性
        self.s_max_len = params.get('s_max_len', 20)
        self.n_max = params.get('n_max', 1000)
    
    def case_generator(self):
        # 生成保证包含多个不同字符的初始字符串
        while True:
            s = ''.join(random.choices(
                string.ascii_lowercase, 
                k=random.randint(self.s_min_len, self.s_max_len)
            ))
            if len(set(s)) >= 2:  # 确保至少包含两个不同字符
                break
        
        k = random.randint(self.k_min, self.k_max)
        total_len = len(s) * k
        max_ops = total_len - 1  # 保留至少一个字符
        n = random.randint(0, min(self.n_max, max_ops))
        
        # 使用双向链表结构跟踪字符位置
        char_map = defaultdict(list)
        full_str = list(s * k)
        for idx, c in enumerate(full_str):
            char_map[c].append(idx)
        
        operations = []
        remaining = total_len
        for _ in range(n):
            # 动态筛选可操作的字符
            valid_chars = [c for c in char_map if len(char_map[c]) > 0]
            if not valid_chars:
                break
            
            # 根据剩余操作次数调整权重
            c = random.choice(valid_chars)
            available = len(char_map[c])
            max_p = available
            
            # 生成不超过当前可用次数的p值
            p = random.randint(1, max_p)
            
            # 记录操作并更新数据结构
            operations.append({'p': p, 'c': c})
            del_idx = char_map[c].pop(p-1)
            full_str[del_idx] = None
            remaining -= 1
            
            # 提前终止条件
            if remaining == 1:
                break
        
        return {
            'k': k,
            's': s,
            'n': len(operations),  # 实际生成的操作数
            'operations': operations
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = f"""## 用户名变更记录解析

初始用户名由字符串 "{question_case['s']}" 重复 {question_case['k']} 次组成（总长度：{len(question_case['s'])*question_case['k']}），随后进行了 {question_case['n']} 次删除操作。

**操作规则**：
1. 每次操作格式："p c" — 删除第p次出现的字符c
2. 字符位置从1开始计数
3. 操作不可逆，后续操作基于当前用户名状态

**操作记录**：
{question_case['n']}
""" + '\n'.join(f"{op['p']} {op['c']}" for op in question_case['operations']) + """

**输出要求**：
请严格按照以下格式返回最终用户名：
[answer]完整结果字符串[/answer]

示例：[answer]abc[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 匹配最后出现的完整答案块
        matches = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 重建原始字符串
        original = list(identity['s'] * identity['k'])
        removed = set()
        
        # 预处理操作索引
        char_positions = defaultdict(list)
        for idx, c in enumerate(original):
            char_positions[c].append(idx)
        
        # 按顺序执行操作
        for op in identity['operations']:
            c = op['c']
            p = op['p'] - 1  # 转换为0-based索引
            
            if p < len(char_positions[c]):
                target = char_positions[c].pop(p)
                removed.add(target)
        
        # 生成最终结果
        expected = ''.join(c for idx, c in enumerate(original) if idx not in removed)
        return solution == expected

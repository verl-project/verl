"""# 

### 谜题描述
Andrew and Eugene are playing a game. Initially, Andrew has string s, consisting of digits. Eugene sends Andrew multiple queries of type \"di → ti\", that means \"replace all digits di in string s with substrings equal to ti\". For example, if s = 123123, then query \"2 → 00\" transforms s to 10031003, and query \"3 → \" (\"replace 3 by an empty string\") transforms it to s = 1212. After all the queries Eugene asks Andrew to find the remainder after division of number with decimal representation equal to s by 1000000007 (109 + 7). When you represent s as a decimal number, please ignore the leading zeroes; also if s is an empty string, then it's assumed that the number equals to zero.

Andrew got tired of processing Eugene's requests manually and he asked you to write a program for that. Help him!

Input

The first line contains string s (1 ≤ |s| ≤ 105), consisting of digits — the string before processing all the requests.

The second line contains a single integer n (0 ≤ n ≤ 105) — the number of queries.

The next n lines contain the descriptions of the queries. The i-th query is described by string \"di->ti\", where di is exactly one digit (from 0 to 9), ti is a string consisting of digits (ti can be an empty string). The sum of lengths of ti for all queries doesn't exceed 105. The queries are written in the order in which they need to be performed.

Output

Print a single integer — remainder of division of the resulting number by 1000000007 (109 + 7).

Examples

Input

123123
1
2-&gt;00


Output

10031003


Input

123123
1
3-&gt;


Output

1212


Input

222
2
2-&gt;0
0-&gt;7


Output

777


Input

1000000008
0


Output

1

Note

Note that the leading zeroes are not removed from string s after the replacement (you can see it in the third sample).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
MOD = 10**9+7

s = raw_input()
n = int(raw_input())
query = [['',s]]+[raw_input().split('->') for i in range(n)]

value = {}
pow10 = {}

for i in range(10):
    value[str(i)] = i
    pow10[str(i)] = 10

for i in range(n,-1,-1):
    newValue = 0
    newPow10 = 1
    for d in query[i][1]:
        newValue = (newValue*pow10[d] + value[d])%MOD
        newPow10 = (newPow10 * pow10[d])%MOD
    value[query[i][0]], pow10[query[i][0]] = (newValue, newPow10)

print value['']
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Esubstitutesinnumberbootcamp(Basebootcamp):
    def __init__(self, s_min_length=1, s_max_length=100, max_queries=10, max_total_ti_length=1000):
        super().__init__()
        self.s_min_length = s_min_length
        self.s_max_length = s_max_length
        self.max_queries = max_queries
        self.max_total_ti_length = max_total_ti_length
    
    def case_generator(self):
        # 确保生成有效初始字符串（长度>=1）
        s_length = random.randint(max(1, self.s_min_length), self.s_max_length)
        s = ''.join(random.choices('0123456789', k=s_length))
        
        # 动态调整最大查询数避免溢出
        effective_max_queries = min(self.max_queries, 10**5)
        n = random.randint(0, effective_max_queries)
        
        queries = []
        total_ti_length = 0
        
        # 生成合法query序列
        available_digits = list('0123456789')
        for _ in range(n):
            if total_ti_length >= self.max_total_ti_length:
                ti = ''
            else:
                remaining = self.max_total_ti_length - total_ti_length
                ti_length = random.randint(0, min(remaining, 10**5))  # 遵守题目约束
                ti = ''.join(random.choices(available_digits, k=ti_length)) if ti_length > 0 else ''
            
            di = random.choice(available_digits)
            queries.append((di, ti))
            total_ti_length += len(ti)
        
        return {
            's': s,
            'queries': queries
        }
    
    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        queries = question_case['queries']
        n = len(queries)
        
        # 格式化query显示
        query_display = []
        for di, ti in queries:
            replacement = "空字符串" if ti == '' else ti
            query_display.append(f"{di} -> {replacement}")
        
        prompt = f"""## 数字替换游戏

### 游戏规则
1. 初始字符串：{s}
2. 需要按顺序执行以下替换操作（共{n}个）：
{chr(10).join(query_display) if query_display else "无替换操作"}
3. 最终结果计算要求：
   - 保留所有前导零（例如替换后得到0023仍视为0023）
   - 空字符串视为0
   - 计算结果对1,000,000,007取模

### 输出格式
请将最终结果放在[answer]标签内，例如：[answer]31415926[/answer]

请逐步思考并给出最终答案。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 增强模式匹配鲁棒性
        pattern = r'\[answer\][\s\n]*(-?\d+)[\s\n]*\[/answer\]'
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1].strip())
            except:
                return None
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 添加类型检查
        if not isinstance(solution, int):
            return False
        return solution == cls.compute_answer(identity['s'], identity['queries'])
    
    @classmethod
    def compute_answer(cls, s, queries):
        # 加强边界条件处理
        if not s and not queries:
            return 0 % MOD
        
        value = {str(d): d % MOD for d in range(10)}
        pow10 = {str(d): 10 % MOD for d in range(10)}
        
        # 初始化特殊键位
        value[''] = 0
        pow10[''] = 1
        
        # 构建完整操作序列（包含初始字符串）
        operation_stack = [('', s)] + queries
        
        # 逆向处理操作序列
        for i in reversed(range(len(operation_stack))):
            current_d, replacement = operation_stack[i]
            
            current_value = 0
            current_pow = 1
            
            for char in replacement:
                current_value = (current_value * pow10.get(char, 1) + value.get(char, 0)) % MOD
                current_pow = (current_pow * pow10.get(char, 1)) % MOD
            
            # 更新当前操作的映射关系
            value[current_d] = current_value
            pow10[current_d] = current_pow
        
        return value.get('', 0) % MOD

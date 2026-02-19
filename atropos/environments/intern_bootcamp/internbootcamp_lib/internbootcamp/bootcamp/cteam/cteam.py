"""# 

### 谜题描述
Now it's time of Olympiads. Vanya and Egor decided to make his own team to take part in a programming Olympiad. They've been best friends ever since primary school and hopefully, that can somehow help them in teamwork.

For each team Olympiad, Vanya takes his play cards with numbers. He takes only the cards containing numbers 1 and 0. The boys are very superstitious. They think that they can do well at the Olympiad if they begin with laying all the cards in a row so that:

  * there wouldn't be a pair of any side-adjacent cards with zeroes in a row; 
  * there wouldn't be a group of three consecutive cards containing numbers one. 



Today Vanya brought n cards with zeroes and m cards with numbers one. The number of cards was so much that the friends do not know how to put all those cards in the described way. Help them find the required arrangement of the cards or else tell the guys that it is impossible to arrange cards in such a way.

Input

The first line contains two integers: n (1 ≤ n ≤ 106) — the number of cards containing number 0; m (1 ≤ m ≤ 106) — the number of cards containing number 1.

Output

In a single line print the required sequence of zeroes and ones without any spaces. If such sequence is impossible to obtain, print -1.

Examples

Input

1 2


Output

101


Input

4 8


Output

110110110101


Input

4 10


Output

11011011011011


Input

1 5


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m = map(int, raw_input().split())

resp = ''

num_1 = 0
num_0 = 0

if n - 1 == m:
    resp = '0'
    temp = '10'
    resp += m * temp
    n = 0
    m = 0

while n != 0 or m != 0:
    if n == m:
        temp = '01'
        resp += n * temp
        n = 0
        m = 0
        break
    else:
        if m != 0 and num_1 != 2:
            resp += '1'
            m -= 1
            num_1 += 1
            num_0 = 0
            
        elif n != 0 and num_0 == 0:
            resp += '0'
            num_1 = 0
            num_0 += 1
            n -= 1
            
        else:
            break

if n == 0 and m == 0:
    print resp
else:
    print -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cteambootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_m=20):
        """
        初始化训练场参数，确保生成的案例参数合法性
        """
        # 保证max_m至少为10以支持各种案例类型
        self.max_n = max(1, max_n)
        self.max_m = max(self.max_n*2+5, max_m)  # 动态调整保证案例生成可能性
    
    def case_generator(self):
        """
        严格遵循数学约束生成案例：
        1. 有解案例必须满足 n <= m+1 且 m <= 2(n+1)
        2. 无解案例必须违反上述任一条件
        """
        for _ in range(100):
            generate_solvable = random.choice([True, False])
            
            if generate_solvable:
                # 生成合法解的参数空间
                n = random.randint(1, self.max_n)
                m_lower = max(n-1, 1)
                m_upper = min(2*(n+1), self.max_m)
                
                if m_lower <= m_upper:
                    m = random.randint(m_lower, m_upper)
                    if n <= m+1 and m <= 2*(n+1):
                        return {'n': n, 'm': m}
                
            else:
                # 确保生成明确的非法参数组合
                violation_type = random.choice([1, 2])
                n, m = 0, 0
                
                if violation_type == 1:  # 违反条件1: n > m+1
                    while True:
                        m = random.randint(1, self.max_m)
                        min_n = m + 2
                        if min_n <= self.max_n:
                            n = random.randint(min_n, self.max_n)
                            break
                else:  # 违反条件2: m > 2(n+1)
                    while True:
                        n = random.randint(1, self.max_n)
                        min_m = 2*(n+1) + 1
                        if min_m <= self.max_m:
                            m = random.randint(min_m, self.max_m)
                            break
                
                # 最终验证参数组合的非法性
                if not (n <= m+1 and m <= 2*(n+1)):
                    return {'n': n, 'm': m}
        
        # 保底返回经典有解案例
        return {'n': 1, 'm': 2}

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        生成包含完整规则和格式化要求的问题描述
        """
        n = question_case['n']
        m = question_case['m']
        return f"""作为编程奥林匹克选手，你需要解决以下卡牌排列问题：

给定{n}张0卡和{m}张1卡，要求排列满足：
1. 不能有相邻的两个0（如00非法）
2. 不能有超过两个连续1（如111非法）

请输出合法排列（如101）或-1表示无解。将最终答案包裹在[answer]标签内，如：[answer]1101[/answer]。"""

    @staticmethod
    def extract_output(output):
        """
        强化答案提取逻辑，处理多种可能输出格式
        """
        # 优先匹配标准格式
        answer_tag_match = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if answer_tag_match:
            candidate = answer_tag_match[-1].strip().replace(' ', '').replace('\n', '')
            if candidate == '-1':
                return '-1'
            if all(c in {'0', '1'} for c in candidate):
                return candidate
        
        # 处理无标签但符合格式的输出
        clean_output = output.strip().replace(' ', '').replace('\n', '')
        if clean_output == '-1':
            return '-1'
        if len(clean_output) == (question_case.get('n',0) + question_case.get('m',0)):
            if '00' not in clean_output and '111' not in clean_output:
                return clean_output
        
        # 提取最长有效序列
        valid_sequences = re.findall(r'[01]+', output)
        if valid_sequences:
            return max(valid_sequences, key=len)
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        实现数学验证算法，确保与参考解法逻辑一致
        """
        n, m = identity['n'], identity['m']
        
        # 处理无解情况
        if solution == '-1':
            # 计算理论是否可解
            valid = not ((n <= m + 1) and (m <= 2 * (n + 1)))
            return valid
        
        # 验证基本参数
        try:
            if (solution.count('0') != n or 
                solution.count('1') != m or
                len(solution) != n + m):
                return False
        except:
            return False
        
        # 实现参考解法验证逻辑
        prev_zero = False
        consecutive_ones = 0
        for c in solution:
            if c == '0':
                if prev_zero:
                    return False
                prev_zero = True
                consecutive_ones = 0
            else:
                consecutive_ones += 1
                if consecutive_ones > 2:
                    return False
                prev_zero = False
        return True

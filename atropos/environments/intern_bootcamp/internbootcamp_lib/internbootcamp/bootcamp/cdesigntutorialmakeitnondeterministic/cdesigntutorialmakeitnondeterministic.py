"""# 

### 谜题描述
A way to make a new task is to make it nondeterministic or probabilistic. For example, the hard task of Topcoder SRM 595, Constellation, is the probabilistic version of a convex hull.

Let's try to make a new task. Firstly we will use the following task. There are n people, sort them by their name. It is just an ordinary sorting problem, but we can make it more interesting by adding nondeterministic element. There are n people, each person will use either his/her first name or last name as a handle. Can the lexicographical order of the handles be exactly equal to the given permutation p?

More formally, if we denote the handle of the i-th person as hi, then the following condition must hold: <image>.

Input

The first line contains an integer n (1 ≤ n ≤ 105) — the number of people.

The next n lines each contains two strings. The i-th line contains strings fi and si (1 ≤ |fi|, |si| ≤ 50) — the first name and last name of the i-th person. Each string consists only of lowercase English letters. All of the given 2n strings will be distinct.

The next line contains n distinct integers: p1, p2, ..., pn (1 ≤ pi ≤ n).

Output

If it is possible, output \"YES\", otherwise output \"NO\".

Examples

Input

3
gennady korotkevich
petr mitrichev
gaoyuan chen
1 2 3


Output

NO


Input

3
gennady korotkevich
petr mitrichev
gaoyuan chen
3 1 2


Output

YES


Input

2
galileo galilei
nicolaus copernicus
2 1


Output

YES


Input

10
rean schwarzer
fei claussell
alisa reinford
eliot craig
laura arseid
jusis albarea
machias regnitz
sara valestin
emma millstein
gaius worzel
1 2 3 4 5 6 7 8 9 10


Output

NO


Input

10
rean schwarzer
fei claussell
alisa reinford
eliot craig
laura arseid
jusis albarea
machias regnitz
sara valestin
emma millstein
gaius worzel
2 4 9 6 5 7 1 3 8 10


Output

YES

Note

In example 1 and 2, we have 3 people: tourist, Petr and me (cgy4ever). You can see that whatever handle is chosen, I must be the first, then tourist and Petr must be the last.

In example 3, if Copernicus uses \"copernicus\" as his handle, everything will be alright.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def MIN(t):
    if t[0] < t[1]:
        return t[0]
    else:
        return t[1]
def MAX(t):
    if t[0] > t[1]:
        return t[0]
    else:
        return t[1] 
n = input()

name = []
while n:
    n -= 1
    a,b = raw_input().split()
    name.append([a,b])
p = map(int,raw_input().split())

Flag = True
Tmp = MIN(name[p[0]-1])
for i in xrange(1,len(p)):
    if Tmp >= MIN(name[p[i]-1]):
        if Tmp >= MAX(name[p[i]-1]):
            Flag = False
            break
        else:
            Tmp = MAX(name[p[i]-1])
    else:
        Tmp = MIN(name[p[i]-1])

if Flag == True:
    print \"YES\"
else:
    print \"NO\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def solve_handle_order(names, p_list):
    if not names or len(p_list) != len(names):
        return "NO"
    n = len(names)
    p = p_list
    Flag = True
    current_user = names[p[0]-1]
    a, b = current_user
    Tmp = a if a < b else b
    for i in range(1, len(p)):
        current_user = names[p[i]-1]
        a, b = current_user
        current_min = a if a < b else b
        current_max = b if a < b else a
        if Tmp >= current_min:
            if Tmp >= current_max:
                Flag = False
                break
            else:
                Tmp = current_max
        else:
            Tmp = current_min
    return "YES" if Flag else "NO"

def generate_random_string(min_length=1, max_length=50):
    length = random.randint(min_length, max_length)
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))

class Cdesigntutorialmakeitnondeterministicbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=10, solvable=None, max_attempts=1000):
        super().__init__()  # 显式调用父类初始化
        self.min_n = min_n
        self.max_n = max_n
        self.solvable = solvable
        self.max_attempts = max_attempts
    
    def case_generator(self):
        for _ in range(self.max_attempts):
            # 保证n至少为1
            n = random.randint(max(1, self.min_n), self.max_n)
            
            names = []
            all_names = set()
            valid = True
            
            # 为每个用户生成唯一的名字对
            for _ in range(n):
                attempt = 0
                while True:
                    first = generate_random_string(1, 50)
                    last = generate_random_string(1, 50)
                    if first != last and first not in all_names and last not in all_names:
                        all_names.update([first, last])
                        names.append((first, last))
                        break
                    attempt += 1
                    if attempt > 100:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # 生成有效排列p
            p = list(range(1, n+1))
            random.shuffle(p)
            answer = solve_handle_order(names, p)
            
            # 根据solvable参数筛选案例
            if self.solvable is None:
                return {'n': n, 'names': names, 'p': p}
            elif (self.solvable and answer == "YES") or (not self.solvable and answer == "NO"):
                return {'n': n, 'names': names, 'p': p}
        
        # 备用方案：生成确定性的可解/不可解案例
        if self.solvable:
            return {'n': 3, 'names': [('a','z'), ('b','y'), ('c','x')], 'p': [3,2,1]}
        else:
            return {'n': 3, 'names': [('z','a'), ('y','b'), ('x','c')], 'p': [1,2,3]}
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])]
        for first, last in question_case['names']:
            input_lines.append(f"{first} {last}")
        input_lines.append(' '.join(map(str, question_case['p'])))
        example_input = '\n'.join(input_lines)
        
        prompt = (
            "给定n个人，每个人可以选择使用名字或姓氏作为handle。\n"
            "判断是否存在一种选择方式，使得按handle的字典序结果恰好等于给定的排列p。\n\n"
            "输入格式：\n"
            f"- 首行：n（人数）\n"
            f"- 接下来n行：每行两个小写字母组成的字符串\n"
            f"- 最后一行：排列p（1-based索引）\n\n"
            "示例：\n"
            "输入：\n3\na b\nc d\ne f\n3 1 2\n输出：YES\n\n"
            "当前问题输入：\n"
            f"{example_input}\n\n"
            "请将最终答案放在[answer]和[/answer]标签之间。"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 支持多格式匹配（包含换行和大小写）
        matches = re.findall(r'\[answer\s*\]\s*(.*?)\s*\[\s*/answer\s*\]', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip().upper()
        return last_match if last_match in {"YES", "NO"} else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证前先进行基本清洗
        cleaned_solution = str(solution).strip().upper()
        return cleaned_solution == solve_handle_order(identity['names'], identity['p'])

# 测试用例示例
if __name__ == "__main__":
    bootcamp = Cdesigntutorialmakeitnondeterministicbootcamp(solvable=True)
    case = bootcamp.case_generator()
    print("生成案例:", case)
    print("问题描述:", bootcamp.prompt_func(case)[:200] + "...")
    print("验证结果:", bootcamp._verify_correction("YES", case))

"""# 

### 谜题描述
This is a harder version of the problem. In this version, n ≤ 300 000.

Vasya is an experienced developer of programming competitions' problems. As all great minds at some time, Vasya faced a creative crisis. To improve the situation, Petya gifted him a string consisting of opening and closing brackets only. Petya believes, that the beauty of the bracket string is a number of its cyclical shifts, which form a correct bracket sequence.

To digress from his problems, Vasya decided to select two positions of the string (not necessarily distinct) and swap characters located at this positions with each other. Vasya will apply this operation exactly once. He is curious what is the maximum possible beauty he can achieve this way. Please help him.

We remind that bracket sequence s is called correct if: 

  * s is empty; 
  * s is equal to \"(t)\", where t is correct bracket sequence; 
  * s is equal to t_1 t_2, i.e. concatenation of t_1 and t_2, where t_1 and t_2 are correct bracket sequences. 



For example, \"(()())\", \"()\" are correct, while \")(\" and \"())\" are not.

The cyclical shift of the string s of length n by k (0 ≤ k < n) is a string formed by a concatenation of the last k symbols of the string s with the first n - k symbols of string s. For example, the cyclical shift of string \"(())()\" by 2 equals \"()(())\".

Cyclical shifts i and j are considered different, if i ≠ j.

Input

The first line contains an integer n (1 ≤ n ≤ 300 000), the length of the string.

The second line contains a string, consisting of exactly n characters, where each of the characters is either \"(\" or \")\".

Output

The first line should contain a single integer — the largest beauty of the string, which can be achieved by swapping some two characters.

The second line should contain integers l and r (1 ≤ l, r ≤ n) — the indices of two characters, which should be swapped in order to maximize the string's beauty.

In case there are several possible swaps, print any of them.

Examples

Input


10
()()())(()


Output


5
8 7


Input


12
)(()(()())()


Output


4
5 10


Input


6
)))(()


Output


0
1 1

Note

In the first example, we can swap 7-th and 8-th character, obtaining a string \"()()()()()\". The cyclical shifts by 0, 2, 4, 6, 8 of this string form a correct bracket sequence.

In the second example, after swapping 5-th and 10-th character, we obtain a string \")(())()()(()\". The cyclical shifts by 11, 7, 5, 3 of this string form a correct bracket sequence.

In the third example, swap of any two brackets results in 0 cyclical shifts being correct bracket sequences. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
s=raw_input()
a=[]
maxx=0
for i in s:
    a.append(i)
    if i=='(':
        maxx+=1
    else:
        maxx-=1
if maxx!=0:
    print 0
    print 1,1
else:
    x=1
    y=1
    maxx=0
    dp=[0]*n
    val=0
    for i in range(n):
        if a[i]=='(':
            val+=1
        else:
            val-=1
        dp[i]=val
    minn=min(dp)
    for i in dp:
        if i==minn:
            maxx+=1
    for i in range(n):
        for j in range(i,n):
            if(a[i]==a[j]):
                continue
            a[i],a[j]=a[j],a[i]
            dp=[0]*n
            val=0
            for i1 in range(n):
                if a[i1]=='(':
                    val+=1
                else:
                    val-=1
                dp[i1]=val
            minn=min(dp)
            for i1 in dp:
                if i1==minn:
                    val+=1
            if val>maxx:
                maxx=val
                x=i+1
                y=j+1
            a[i],a[j]=a[j],a[i]
    print maxx
    print x,y
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_min_balance_and_count(s):
    balance = 0
    min_balance = 0
    count = 0
    prefix = []
    for c in s:
        balance += 1 if c == '(' else -1
        prefix.append(balance)
        if balance < min_balance:
            min_balance = balance
            count = 1
        elif balance == min_balance:
            count += 1
    return min_balance, count, prefix

def calculate_real_beauty(s):
    total = sum(1 if c == '(' else -1 for c in s)
    if total != 0:
        return 0
    min_balance, count, prefix = compute_min_balance_and_count(s)
    overall_min = min(prefix)
    if overall_min < 0:
        return 0
    return count

def optimal_solution(n, s):
    max_beauty = 0
    best_pair = (1, 1)
    original_beauty = calculate_real_beauty(s)
    max_beauty = original_beauty
    
    s_list = list(s)
    for i in range(n):
        for j in range(i, n):
            if s_list[i] == s_list[j]:
                continue
            
            # Perform swap
            s_list[i], s_list[j] = s_list[j], s_list[i]
            new_s = ''.join(s_list)
            current_beauty = calculate_real_beauty(new_s)
            
            if current_beauty > max_beauty:
                max_beauty = current_beauty
                best_pair = (i+1, j+1)
            
            # Revert swap
            s_list[i], s_list[j] = s_list[j], s_list[i]
    
    return (max_beauty, best_pair[0], best_pair[1])

class Btheworldisjustaprogrammingtaskhardversionbootcamp(Basebootcamp):
    def __init__(self, max_n=12):
        self.max_n = max_n  # 控制案例规模保证验证效率
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        # 生成有效测试案例（包含平衡和非平衡情况）
        if random.random() < 0.5 and n % 2 == 0:
            # 生成平衡括号字符串
            s = ['(']*(n//2) + [')']*(n//2)
            random.shuffle(s)
            s = ''.join(s)
        else:
            # 随机生成可能不平衡的字符串
            s = ''.join(random.choices(['(', ')'], k=n))
        
        max_beauty, l, r = optimal_solution(n, s)
        return {
            'n': n,
            's': s,
            'expected_max': max_beauty,
            'swap_l': l,
            'swap_r': r
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['s']
        return f"""给定一个长度为{n}的括号字符串："{s}"
请通过交换两个字符（允许相同位置），最大化循环移位构成有效括号序列的数量。输出最大数量及交换位置（1-based）。

有效括号序列定义：
1. 空字符串
2. (A) 其中A是有效序列
3. AB 其中A和B都是有效序列

答案格式：
[answer]
{{最大数量}}
{{位置1}} {{位置2}}
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        try:
            if len(lines) >= 2:
                max_val = int(lines[0])
                l, r = map(int, lines[1].split())
                return (max_val, l, r)
        except:
            pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != 3:
            return False
        max_val, l, r = solution
        
        # 验证基础参数
        if max_val != identity['expected_max']:
            return False
        if not (1 <= l <= identity['n'] and 1 <= r <= identity['n']):
            return False
        
        # 执行交换操作
        s_list = list(identity['s'])
        l_idx, r_idx = l-1, r-1
        s_list[l_idx], s_list[r_idx] = s_list[r_idx], s_list[l_idx]
        new_s = ''.join(s_list)
        
        # 计算实际美丽值
        actual_beauty = calculate_real_beauty(new_s)
        
        # 允许误差处理（应对计算误差）
        return actual_beauty == max_val

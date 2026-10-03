"""# 

### 谜题描述
Little X has met the following problem recently. 

Let's define f(x) as the sum of digits in decimal representation of number x (for example, f(1234) = 1 + 2 + 3 + 4). You are to calculate <image>

Of course Little X has solved this problem quickly, has locked it, and then has tried to hack others. He has seen the following C++ code: 
    
    
      
        ans = solve(l, r) % a;  
        if (ans <= 0)  
          ans += a;  
      
    

This code will fail only on the test with <image>. You are given number a, help Little X to find a proper test for hack.

Input

The first line contains a single integer a (1 ≤ a ≤ 1018).

Output

Print two integers: l, r (1 ≤ l ≤ r < 10200) — the required test data. Leading zeros aren't allowed. It's guaranteed that the solution exists.

Examples

Input

46


Output

1 10


Input

126444381000032


Output

2333333 2333333333333

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def f(s):
    N = 0
    p = 0
    L = len(s)
    for i in xrange(len(s)):
        k = L - i - 1
        for j in xrange(int(s[i])):
            N += 9 * k * (10 ** k) / 2
            N += (p + j) * (10 ** k)
        p += int(s[i])
    return N

def g(N):
    if N == 0:
        return '0'
    s = ''
    L = 20
    for i in xrange(L):
        d = 0
        for j in xrange(10):
            if f(s + str(j) + ('0' * (L - i - 1))) >= N:
                s = s + str(j - 1)
                d = 1
                break
        if not d:
            s = s + str(9)
        #print s
    return s

A = input()
s = []
p = []
i = 1
while 1:
    m = g(i * A)
    q = f(m) % A
    d = 0
#    print m, q
    for i in xrange(len(p)):
        if q == p[i] and int(m) > int(s[i]):
            print int(s[i]), int(m) - 1
            d = 1
            break
    if d:
        break
    s.append(m)
    p.append(f(m) % A)
    i += 1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def f(s):
    N = 0
    p = 0
    L = len(s)
    for i in range(len(s)):
        k = L - i - 1
        for j in range(int(s[i])):
            term1 = 9 * k * (10 ** k) // 2
            term2 = (p + j) * (10 ** k)
            N += term1 + term2
        p += int(s[i])
    return N

def g(N):
    if N == 0:
        return '0'
    s = ''
    L = 200  # 调整为200位
    for i in range(L):
        d = 0
        for j in range(10):
            test_s = s + str(j) + '0' * (L - i - 1)
            current_f = f(test_s)
            if current_f >= N:
                if j > 0:
                    s += str(j-1)
                else:
                    s += '0'  # 处理j=0的情况
                d = 1
                break
        if not d:
            s += '9'
    s = s.lstrip('0') or '0'
    return s

def find_test_case(a):
    s_list = []
    p_list = []
    i = 1
    while True:
        target = i * a
        m = g(target)
        q = f(m) % a
        for idx in range(len(p_list)):
            if q == p_list[idx] and int(m) > int(s_list[idx]):
                l = int(s_list[idx])
                r = int(m) - 1
                return (l, r)
        s_list.append(m)
        p_list.append(q)
        i += 1

class Chackitbootcamp(Basebootcamp):
    def __init__(self, max_a=10**18):  # 支持更大的a值
        self.max_a = max_a

    def case_generator(self):
        a = random.randint(1, self.max_a)
        l, r = find_test_case(a)
        # 确保数值有效性（示例代码逻辑保证）
        return {'a': a, 'l': l, 'r': r}

    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        prompt = f"""Little X需要构造一个hack测试用例。请找到两个整数l和r，使得所有在区间[l, r]内数字的digit sum之和模{a}等于0。

输入要求：
- 第一行为整数a（此处a={a}）
- 输出l和r满足：1 ≤ l ≤ r < 10^200

关键规则：
1. digit sum定义：例如f(123)=1+2+3=6
2. 目标程序使用错误取模逻辑：ans = solve(l, r) % a; if (ans <=0) ans += a;
3. 你构造的(l, r)必须使正确结果为0，但目标程序会错误输出a

输出格式：
将答案放在[answer]标签内，例如：[answer]1 10[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        # 优先提取answer标签内容
        answer_tag = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if answer_tag:
            pair = answer_tag[-1].strip().split()
            if len(pair) == 2:
                try:
                    return (int(pair[0]), int(pair[1]))
                except:
                    pass
        
        # 兜底提取最后的数字对
        digit_pairs = re.findall(r'\b(\d+)\s+(\d+)\b', output)
        if digit_pairs:
            last_pair = digit_pairs[-1]
            try:
                return (int(last_pair[0]), int(last_pair[1]))
            except:
                return None
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != 2:
            return False
        l, r = solution
        a = identity['a']
        
        # 基本约束检查
        if not (1 <= l <= r < 10**200):
            return False
        if any(len(str(x)) != len(str(int(x))) for x in (l, r)):  # 检查前导零
            return False

        # 优化计算模值
        def compute_mod(n_str, mod):
            total = 0
            pos = 0
            digit_sum = 0
            for ch in reversed(n_str):
                digit = int(ch)
                cnt = digit
                # 0-9的模式重复次数
                full_cycles = cnt * (cnt-1) // 2
                total += (full_cycles * (10**pos)) % mod
                # 当前位贡献
                total += digit_sum * cnt * (10**pos) % mod
                # 更新digit_sum
                digit_sum += digit
                pos += 1
            return total % mod

        try:
            mod_total = (compute_mod(str(r), a) - compute_mod(str(l-1), a)) % a
            return mod_total == 0
        except:
            return False

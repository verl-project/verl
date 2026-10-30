"""# 

### 谜题描述
Anna is a girl so brave that she is loved by everyone in the city and citizens love her cookies. She is planning to hold a party with cookies. Now she has a vanilla cookies and b chocolate cookies for the party.

She invited n guests of the first type and m guests of the second type to the party. They will come to the party in some order. After coming to the party, each guest will choose the type of cookie (vanilla or chocolate) to eat. There is a difference in the way how they choose that type:

If there are v vanilla cookies and c chocolate cookies at the moment, when the guest comes, then

  * if the guest of the first type: if v>c the guest selects a vanilla cookie. Otherwise, the guest selects a chocolate cookie. 
  * if the guest of the second type: if v>c the guest selects a chocolate cookie. Otherwise, the guest selects a vanilla cookie. 



After that:

  * If there is at least one cookie of the selected type, the guest eats one. 
  * Otherwise (there are no cookies of the selected type), the guest gets angry and returns to home. 



Anna wants to know if there exists some order of guests, such that no one guest gets angry. Your task is to answer her question.

Input

The input consists of multiple test cases. The first line contains a single integer t (1 ≤ t ≤ 1000) — the number of test cases. Next t lines contain descriptions of test cases.

For each test case, the only line contains four integers a, b, n, m (0 ≤ a,b,n,m ≤ 10^{18}, n+m ≠ 0).

Output

For each test case, print the answer in one line. If there exists at least one valid order, print \"Yes\". Otherwise, print \"No\".

You can print each letter in any case (upper or lower).

Example

Input


6
2 2 1 2
0 100 0 1
12 13 25 1
27 83 14 25
0 0 1 0
1000000000000000000 1000000000000000000 1000000000000000000 1000000000000000000


Output


Yes
No
No
Yes
No
Yes

Note

In the first test case, let's consider the order \{1, 2, 2\} of types of guests. Then:

  * The first guest eats a chocolate cookie. After that, there are 2 vanilla cookies and 1 chocolate cookie. 
  * The second guest eats a chocolate cookie. After that, there are 2 vanilla cookies and 0 chocolate cookies. 
  * The last guest selects a chocolate cookie, but there are no chocolate cookies. So, the guest gets angry. 



So, this order can't be chosen by Anna.

Let's consider the order \{2, 2, 1\} of types of guests. Then:

  * The first guest eats a vanilla cookie. After that, there is 1 vanilla cookie and 2 chocolate cookies. 
  * The second guest eats a vanilla cookie. After that, there are 0 vanilla cookies and 2 chocolate cookies. 
  * The last guest eats a chocolate cookie. After that, there are 0 vanilla cookies and 1 chocolate cookie. 



So, the answer to this test case is \"Yes\".

In the fifth test case, it is illustrated, that the number of cookies (a + b) can be equal to zero, but the number of guests (n + m) can't be equal to zero.

In the sixth test case, be careful about the overflow of 32-bit integer type.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
if sys.subversion[0] == \"PyPy\":
    import io, atexit
    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
    
    sys.stdin = io.BytesIO(sys.stdin.read())
    input = lambda: sys.stdin.readline().rstrip()

RS = raw_input
RI = lambda x=int: map(x,RS().split())
RN = lambda x=int: x(RS())
''' ...................................................................... '''

for _ in xrange(RN()):
    a,b,n,m = RI()
    ans = 0

    if a<b: a,b = b,a

    if n+m <= (a+b) and m<=b:
        ans = 1


    print ('Yes' if ans else 'No')
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random

class Cacookieforyoubootcamp(Basebootcamp):
    def __init__(self, max_value=10**18):
        self.max_value = max_value
        self.max_retries = 1000  # 新增重试次数限制防止死循环
    
    def case_generator(self):
        rand = random.random()
        if rand < 0.25:
            return self._generate_zero_cookie_case()
        elif rand < 0.5:
            return self._generate_single_type_guest_case()
        elif rand < 0.75:
            return self._generate_yes_case()
        else:
            return self._generate_no_case()
    
    def _generate_zero_cookie_case(self):
        """确保至少有一个客人存在"""
        a, b = 0, 0
        while True:
            n = random.randint(0, self.max_value)
            m = random.randint(0, self.max_value)
            if n + m > 0:
                return {'a': a, 'b': b, 'n': n, 'm': m}
    
    def _generate_single_type_guest_case(self):
        """确保至少有一类客人存在"""
        while True:
            if random.random() < 0.5:
                case = {
                    'a': random.randint(0, self.max_value),
                    'b': random.randint(0, self.max_value),
                    'n': random.randint(0, self.max_value),
                    'm': 0
                }
            else:
                case = {
                    'a': random.randint(0, self.max_value),
                    'b': random.randint(0, self.max_value),
                    'n': 0,
                    'm': random.randint(0, self.max_value)
                }
            if case['n'] + case['m'] > 0:
                return case
    
    def _generate_yes_case(self):
        """添加重试机制防止死循环"""
        for _ in range(self.max_retries):
            a = random.randint(0, self.max_value)
            b = random.randint(0, self.max_value)
            if a + b == 0:
                continue
            
            a_new, b_new = max(a, b), min(a, b)
            max_m = b_new
            m = random.randint(0, max_m)
            remaining = (a_new + b_new) - m
            n = random.randint(0, remaining)
            
            if n + m > 0:
                return {'a': a, 'b': b, 'n': n, 'm': m}
        # 保底生成合法案例
        return {'a': 2, 'b': 1, 'n': 1, 'm': 1}
    
    def _generate_no_case(self):
        strategies = [
            self._generate_case_total_exceed,
            self._generate_case_m_exceed,
            self._generate_zero_cookie_angry_case
        ]
        return random.choice(strategies)()
    
    def _generate_case_total_exceed(self):
        for _ in range(self.max_retries):
            a = random.randint(1, self.max_value)
            b = random.randint(1, self.max_value)
            total = a + b
            min_guest = total + 1
            n = random.randint(0, min_guest)
            m = min_guest - n
            if m < 0:
                m = 0
                n = min_guest
            if a + b < n + m:
                return {'a': a, 'b': b, 'n': n, 'm': m}
        return {'a': 1, 'b': 1, 'n': 2, 'm': 1}
    
    def _generate_case_m_exceed(self):
        for _ in range(self.max_retries):
            a = random.randint(0, self.max_value)
            b = random.randint(0, self.max_value)
            a_new, b_new = max(a, b), min(a, b)
            if b_new == 0:
                continue
            m = random.randint(b_new + 1, self.max_value)
            remaining = (a_new + b_new) - m
            n = random.randint(0, max(remaining, 0))
            if (n + m) <= (a_new + b_new):
                return {'a': a, 'b': b, 'n': n, 'm': m}
        return {'a': 3, 'b': 1, 'n': 1, 'm': 3}
    
    def _generate_zero_cookie_angry_case(self):
        return {'a': 0, 'b': 0, 
                'n': random.randint(1, self.max_value),
                'm': random.randint(0, self.max_value)}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        a = question_case['a']
        b = question_case['b']
        n = question_case['n']
        m = question_case['m']
        problem = f"""Anna has {a} vanilla and {b} chocolate cookies. She invited:
- {n} Type 1 guests (choose majority type)
- {m} Type 2 guests (choose minority type)

Determine if any guest order exists where none leave angry. 

Rules:
1. Guests arrive in any order
2. Each chooses based on current cookie counts
3. Leaves if chosen type unavailable

Answer format: [answer]Yes[/answer] or [answer]No[/answer]"""
        return problem
    
    @staticmethod
    def extract_output(output):
        # 强化提取逻辑处理换行和空格
        matches = re.findall(r'\[answer\s*](.*?)\[/answer\s*]', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip().lower()
        return 'Yes' if last_answer == 'yes' else 'No' if last_answer == 'no' else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a, b, n, m = identity['a'], identity['b'], identity['n'], identity['m']
        
        # 处理饼干为零的特殊情况
        if a + b == 0:
            return solution == 'No'
        
        # 主验证逻辑
        a_new, b_new = max(a, b), min(a, b)
        total_guest = n + m
        total_cookie = a_new + b_new
        
        valid = (total_guest <= total_cookie) and (m <= b_new)
        expected = 'Yes' if valid else 'No'
        return solution.lower() == expected.lower()

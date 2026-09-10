"""# 

### 谜题描述
Nearly each project of the F company has a whole team of developers working on it. They often are in different rooms of the office in different cities and even countries. To keep in touch and track the results of the project, the F company conducts shared online meetings in a Spyke chat.

One day the director of the F company got hold of the records of a part of an online meeting of one successful team. The director watched the record and wanted to talk to the team leader. But how can he tell who the leader is? The director logically supposed that the leader is the person who is present at any conversation during a chat meeting. In other words, if at some moment of time at least one person is present on the meeting, then the leader is present on the meeting.

You are the assistant director. Given the 'user logged on'/'user logged off' messages of the meeting in the chronological order, help the director determine who can be the leader. Note that the director has the record of only a continuous part of the meeting (probably, it's not the whole meeting).

Input

The first line contains integers n and m (1 ≤ n, m ≤ 105) — the number of team participants and the number of messages. Each of the next m lines contains a message in the format:

  * '+ id': the record means that the person with number id (1 ≤ id ≤ n) has logged on to the meeting. 
  * '- id': the record means that the person with number id (1 ≤ id ≤ n) has logged off from the meeting. 



Assume that all the people of the team are numbered from 1 to n and the messages are given in the chronological order. It is guaranteed that the given sequence is the correct record of a continuous part of the meeting. It is guaranteed that no two log on/log off events occurred simultaneously.

Output

In the first line print integer k (0 ≤ k ≤ n) — how many people can be leaders. In the next line, print k integers in the increasing order — the numbers of the people who can be leaders.

If the data is such that no member of the team can be a leader, print a single number 0.

Examples

Input

5 4
+ 1
+ 2
- 2
- 1


Output

4
1 3 4 5 

Input

3 2
+ 1
- 2


Output

1
3 

Input

2 4
+ 1
- 1
+ 2
- 2


Output

0


Input

5 6
+ 1
- 1
- 3
+ 3
+ 4
- 4


Output

3
2 3 5 

Input

2 4
+ 1
- 2
+ 2
- 1


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#import random

n, m = map(lambda x: int(x), raw_input().split())
#n, m = 100000, 100000

messages = list()

for i in range(m):
    sign, num = raw_input().split()
    #sign, num = random.choice(['+', '-']), random.randint(1, n)
    num = int(num)
    messages.append((sign, num))

entered = set()
prefix = list()
for m in messages:
    sign, num = m
    if sign == '+':
        entered.add(num)
    else:
        if not num in entered:
            prefix.append(('+', num))

prefix.reverse()
messages = prefix + messages

online = set()
leaders = set(range(1, n + 1))

prev_sign = None
prev_num = 0

for m in messages:
    sign, num = m
    if prev_sign and prev_sign != sign and prev_num != num:
        if num in leaders:
            leaders.remove(num)
        if prev_num in leaders:
            leaders.remove(prev_num)
    if sign == '+':
        if len(online) > 0 and num in leaders:
            leaders.remove(num)
        online.add(num)
    else:
        if num in online:
            online.remove(num)
        if len(online) > 0 and num in leaders:
            leaders.remove(num)
    prev_sign, prev_num = sign, num

print len(leaders)

if len(leaders) > 0:
    print ' '.join([str(x) for x in sorted(list(leaders))])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def calculate_solution(n, messages):
    entered = set()
    prefix = []
    for sign, num in messages:
        if sign == '+':
            entered.add(num)
        else:
            if num not in entered:
                prefix.append(('+', num))
    prefix.reverse()
    full_messages = prefix + messages
    online = set()
    leaders = set(range(1, n+1))
    prev_sign = None
    prev_num = 0
    
    for m in full_messages:
        sign, num = m
        if prev_sign is not None and prev_sign != sign and prev_num != num:
            if num in leaders:
                leaders.remove(num)
            if prev_num in leaders:
                leaders.remove(prev_num)
        if sign == '+':
            if len(online) > 0 and num in leaders:
                leaders.remove(num)
            online.add(num)
        else:
            if num in online:
                online.remove(num)
            if len(online) > 0 and num in leaders:
                leaders.remove(num)
        prev_sign, prev_num = sign, num
    return sorted(leaders) if leaders else []

class Conlinemeetingbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=20):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        messages = []
        online = set()
        
        for _ in range(m):
            available_ops = []
            if online:
                available_ops.append('-')
            if len(online) < n:
                available_ops.append('+')
            if not available_ops:
                break
            
            sign = random.choice(available_ops)
            if sign == '+':
                available_users = list(set(range(1, n+1)) - online)
                user_id = random.choice(available_users)
                online.add(user_id)
            else:
                user_id = random.choice(list(online))
                online.remove(user_id)
            messages.append((sign, user_id))
        
        while len(messages) < m:
            available_ops = []
            if online:
                available_ops.append('-')
            if len(online) < n:
                available_ops.append('+')
            if not available_ops:
                break
            
            sign = random.choice(available_ops)
            if sign == '+':
                available_users = list(set(range(1, n+1)) - online)
                user_id = random.choice(available_users)
                online.add(user_id)
            else:
                user_id = random.choice(list(online))
                online.remove(user_id)
            messages.append((sign, user_id))
        
        expected_leaders = calculate_solution(n, messages)
        return {
            'n': n,
            'm': len(messages),
            'messages': messages,
            'expected_leaders': expected_leaders
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        messages = question_case['messages']
        input_lines = [f"{n} {m}"] + [f"{sign} {user_id}" for sign, user_id in messages]
        input_str = '\n'.join(input_lines)
        prompt = f"""You are the assistant director of company F. Given the login/logout records of an online meeting, determine all possible team leaders. The leader must be present whenever at least one person is online during the recorded period.

Input format:
- First line: two integers n (number of team members) and m (number of messages)
- Following m lines: Each line is '+ id' (login) or '- id' (logout)

Output requirements:
- If there are possible leaders: 
  First line: k (number of leaders)
  Second line: k IDs in increasing order
- If no possible leaders: 
  Single line: 0

Put your final answer within [answer] and [/answer]. For example:
[answer]
3
2 4 5
[/answer]

Input data:
{input_str}"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
        
        try:
            k = int(lines[0])
        except:
            return None
        
        if k == 0:
            return 0 if len(lines) == 1 else None
        else:
            if len(lines) < 2:
                return None
            try:
                leaders = list(map(int, lines[1].split()))
            except:
                return None
            if len(leaders) != k or sorted(leaders) != leaders:
                return None
            return leaders
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_leaders']
        if solution == 0:
            return len(expected) == 0
        elif isinstance(solution, list):
            return solution == expected
        else:
            return False

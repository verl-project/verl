"""# 

### 谜题描述
Vasya has several phone books, in which he recorded the telephone numbers of his friends. Each of his friends can have one or several phone numbers.

Vasya decided to organize information about the phone numbers of friends. You will be given n strings — all entries from Vasya's phone books. Each entry starts with a friend's name. Then follows the number of phone numbers in the current entry, and then the phone numbers themselves. It is possible that several identical phones are recorded in the same record.

Vasya also believes that if the phone number a is a suffix of the phone number b (that is, the number b ends up with a), and both numbers are written by Vasya as the phone numbers of the same person, then a is recorded without the city code and it should not be taken into account.

The task is to print organized information about the phone numbers of Vasya's friends. It is possible that two different people have the same number. If one person has two numbers x and y, and x is a suffix of y (that is, y ends in x), then you shouldn't print number x. If the number of a friend in the Vasya's phone books is recorded several times in the same format, it is necessary to take it into account exactly once.

Read the examples to understand statement and format of the output better.

Input

First line contains the integer n (1 ≤ n ≤ 20) — number of entries in Vasya's phone books. 

The following n lines are followed by descriptions of the records in the format described in statement. Names of Vasya's friends are non-empty strings whose length does not exceed 10. They consists only of lowercase English letters. Number of phone numbers in one entry is not less than 1 is not more than 10. The telephone numbers consist of digits only. If you represent a phone number as a string, then its length will be in range from 1 to 10. Phone numbers can contain leading zeros.

Output

Print out the ordered information about the phone numbers of Vasya's friends. First output m — number of friends that are found in Vasya's phone books.

The following m lines must contain entries in the following format \"name number_of_phone_numbers phone_numbers\". Phone numbers should be separated by a space. Each record must contain all the phone numbers of current friend.

Entries can be displayed in arbitrary order, phone numbers for one record can also be printed in arbitrary order.

Examples

Input

2
ivan 1 00123
masha 1 00123


Output

2
masha 1 00123 
ivan 1 00123 


Input

3
karl 2 612 12
petr 1 12
katya 1 612


Output

3
katya 1 612 
petr 1 12 
karl 1 612 


Input

4
ivan 3 123 123 456
ivan 2 456 456
ivan 8 789 3 23 6 56 9 89 2
dasha 2 23 789


Output

2
dasha 2 23 789 
ivan 4 789 123 2 456 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
di={}
for i in range(0,n):
  s=raw_input().split()
  s1=str(s[0])
  k=int(s[1])
  t=[]
  for i in range(2,len(s)):
    t.append(s[i])
  if s1 in di:
    di[s1]=di[s1]+t 
  else:
    di[s1]=t 
print len(di)
for k in di:
  val=di[k]
  ans=[]
  for i in range(0,len(val)):
    val[i]=(len(val[i]),val[i])  
    
  val.sort(reverse=True)
  for i in range(0,len(val)):
    val[i]=(val[i][1]) 
  for i in range(0,len(val)):
    z=0
    for j in range(0,len(ans)):
      vl=ans[j] 
      l=len(ans[j])
      e=val[i] 
      l2=len(e)
      if vl[l-l2:]==e:
        z=-1 
        break 
    if z==0:
      ans.append(val[i])
  print k ,len(ans),\" \".join(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

class Cphonenumbersbootcamp(Basebootcamp):
    def __init__(self, max_entries=20, min_entries=1, max_phones_per_entry=10, min_phones_per_entry=1):
        self.max_entries = max_entries
        self.min_entries = min_entries
        self.max_phones = max_phones_per_entry
        self.min_phones = min_phones_per_entry
    
    def case_generator(self):
        n = random.randint(self.min_entries, self.max_entries)
        m = random.randint(1, n)
        friends = []
        for _ in range(m):
            name_length = random.randint(1, 10)
            name = ''.join(random.choices(string.ascii_lowercase, k=name_length))
            friends.append(name)
        
        expected = {}
        for name in friends:
            valid_numbers = self._generate_valid_numbers()
            expected[name] = valid_numbers
        
        entries = []
        remaining = n
        for name in friends:
            if remaining <= 0:
                break
            entries.append({'name': name, 'numbers': []})
            remaining -= 1
        
        while remaining > 0:
            name = random.choice(friends)
            entries.append({'name': name, 'numbers': []})
            remaining -= 1
        
        for entry in entries:
            name = entry['name']
            valid = expected[name]
            redundant = []
            for vn in valid:
                for l in range(1, len(vn)):
                    redundant.append(vn[-l:])
            possible_numbers = valid + redundant
            k = random.randint(self.min_phones, self.max_phones)
            selected = random.choices(possible_numbers, k=k)
            entry['numbers'] = selected
        
        return {
            'n': n,
            'entries': entries,
            'expected': expected
        }
    
    def _generate_valid_numbers(self, max_tries=10):
        valid = []
        candidates = []
        for _ in range(max_tries):
            length = random.randint(1, 10)
            number = ''.join(random.choices('0123456789', k=length))
            candidates.append(number)
        candidates.sort(key=lambda x: len(x), reverse=True)
        for num in candidates:
            if not any(num.endswith(vn) for vn in valid):
                valid.append(num)
        if not valid:
            number = ''.join(random.choices('0123456789', k=random.randint(1, 10)))
            valid.append(number)
        num_valid = random.randint(1, 5)
        while len(valid) > num_valid:
            valid.pop()
        return valid
    
    @staticmethod
    def prompt_func(question_case):
        entries = question_case['entries']
        input_lines = [f"{entry['name']} {len(entry['numbers'])} {' '.join(entry['numbers'])}" for entry in entries]
        input_str = f"{question_case['n']}\n" + '\n'.join(input_lines)
        prompt = f"""Vasya整理电话簿的任务：

你需要帮助Vasya整理他的电话簿记录。每个记录包含朋友的姓名及其电话号码。处理规则如下：

1. 合并同一朋友的所有电话号码，去除重复的号码。
2. 如果同一朋友的电话号码a是另一个号码b的后缀（即b以a结尾），则只保留较长的号码b。
3. 输出时，朋友顺序不限，每个朋友的电话号码顺序也不限，但每个号码必须唯一。

输入格式：
第一行是记录的个数n。
接下来的n行，每行格式为：姓名 k 电话号码1 电话号码2 ... 电话号码k。

输出格式：
第一行是朋友的数量m。
接下来m行，每行格式为：姓名 k 电话号码1 ... 电话号码k，其中k是处理后该朋友的唯一电话号码数量。

请处理以下输入数据，并将最终答案放在[answer]标签之间。

输入数据：
{input_str}

确保输出格式正确，并将答案放在[answer]和[/answer]之间。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        def parse_solution(solution_str):
            lines = solution_str.strip().split('\n')
            if not lines:
                return None
            try:
                m = int(lines[0].strip())
            except:
                return None
            if len(lines) < m + 1:
                return None
            result = {}
            for line in lines[1:m+1]:
                parts = line.strip().split()
                if len(parts) < 2:
                    return None
                name = parts[0]
                try:
                    count = int(parts[1])
                except:
                    return None
                numbers = parts[2:]
                if len(numbers) != count:
                    return None
                result[name] = set(numbers)
            return result
        
        parsed = parse_solution(solution)
        if parsed is None:
            return False
        expected = identity['expected']
        if set(parsed.keys()) != set(expected.keys()):
            return False
        for name in expected:
            if name not in parsed:
                return False
            expected_set = set(expected[name])
            actual_set = parsed[name]
            if expected_set != actual_set:
                return False
        return True

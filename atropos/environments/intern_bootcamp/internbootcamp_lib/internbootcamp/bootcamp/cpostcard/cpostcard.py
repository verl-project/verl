"""# 

### 谜题描述
Andrey received a postcard from Irina. It contained only the words \"Hello, Andrey!\", and a strange string consisting of lowercase Latin letters, snowflakes and candy canes. Andrey thought that this string is an encrypted message, and decided to decrypt it.

Andrey noticed that snowflakes and candy canes always stand after the letters, so he supposed that the message was encrypted as follows. Candy cane means that the letter before it can be removed, or can be left. A snowflake means that the letter before it can be removed, left, or repeated several times.

For example, consider the following string: 

<image>

This string can encode the message «happynewyear». For this, candy canes and snowflakes should be used as follows: 

  * candy cane 1: remove the letter w, 
  * snowflake 1: repeat the letter p twice, 
  * candy cane 2: leave the letter n, 
  * snowflake 2: remove the letter w, 
  * snowflake 3: leave the letter e. 

<image>

Please note that the same string can encode different messages. For example, the string above can encode «hayewyar», «happpppynewwwwwyear», and other messages.

Andrey knows that messages from Irina usually have a length of k letters. Help him to find out if a given string can encode a message of k letters, and if so, give an example of such a message.

Input

The first line contains the string received in the postcard. The string consists only of lowercase Latin letters, as well as the characters «*» and «?», meaning snowflake and candy cone, respectively. These characters can only appear immediately after the letter. The length of the string does not exceed 200.

The second line contains an integer number k (1 ≤ k ≤ 200), the required message length.

Output

Print any message of length k that the given string can encode, or «Impossible» if such a message does not exist.

Examples

Input


hw?ap*yn?eww*ye*ar
12


Output


happynewyear


Input


ab?a
2


Output


aa

Input


ab?a
3


Output


aba

Input


ababb
5


Output


ababb

Input


ab?a
1


Output


Impossible

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
inp=raw_input()
inpk=input()
# inpk=12
ns=0
ny=0
nss=[]
nyy=[]
for i,k in enumerate(inp):
    if k==\"?\":
        ns+=1
        nss.append(i)
    if k==\"*\":
        ny+=1
        nyy.append(i)
# print nss,nyy
# print ns,ny
s=\"\"
flag=1
if not (inpk>=len(inp)-2*ns-2*ny):
    print \"Impossible\"
    flag=0
elif ny==0:
    if inpk>len(inp)-ns:
        print \"Impossible\"
        flag=0
def stt(astr):
    s=\"\"
    for i in astr:
        if i==\"?\" or i==\"*\":
            continue
        s+=i
    return s
# print \"lol\",stt(inp)
if flag:
    if len(stt(inp))==inpk:print stt(inp)
    else:
        change=inpk-len(stt(inp))
        # print change
        nsss=[]
        nyyy=[]
        # print \"nyy\",nyyy
        # for i in range(abs(change)):
        if change<0:
            nss.extend(nyy)
            for i in range(abs(change)):
                nsss.append(nss[i])
        if change>0:
            # print \"nyy\",nyyy
            for i in range(abs(change)+1):
                nyyy.append(nyy[0])
        news=\"\"
        # print nyyy
        if change<0:
            for i,k in enumerate(inp):
                if i+1 in nsss:
                    continue
                else:news+=k
            print stt(news)
        if change>0:
            for i,k in enumerate(inp):
                if i+1 in nyyy:
                    news+=inp[i]*len(nyyy)
                else:news+=k
            print stt(news)

        # for i in inp:
        #     if i+1 in nss:
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cpostcardbootcamp(Basebootcamp):
    def __init__(self, max_length=200, symbol_prob=0.4):
        self.max_length = max_length
        self.symbol_prob = symbol_prob
    
    def case_generator(self):
        # 生成符合字符后接符号规则的加密字符串
        encrypted = []
        length = random.randint(1, self.max_length)
        
        # 首字符必须是字母
        encrypted.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
        for _ in range(1, length):
            # 前一个字符是字母时才可添加符号
            if encrypted[-1] not in ['?', '*'] and random.random() < self.symbol_prob:
                encrypted.append(random.choice(['?', '*']))
                if len(encrypted) >= length:
                    break  # 长度控制
            encrypted.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
        
        encrypted_str = ''.join(encrypted[:length])
        
        # 动态计算k的范围
        min_k = self._calculate_min_length(encrypted_str)
        max_k = self._calculate_max_length(encrypted_str)
        
        # 生成k值：40%概率有效，60%概率可能越界
        if random.random() < 0.4 and max_k >= min_k:
            k = random.randint(min_k, max_k)
        else:
            k = random.randint(1, 200)
        
        return {
            'encrypted_str': encrypted_str,
            'k': k
        }
    
    @classmethod
    def _calculate_min_length(cls, s):
        """计算最小可能长度"""
        return len(s) - 2*(s.count('?') + s.count('*'))
    
    @classmethod
    def _calculate_max_length(cls, s):
        """计算最大可能长度"""
        base = len(s) - s.count('?') - s.count('*')
        stars = s.count('*')
        return base + 100*stars if stars > 0 else base
    
    @staticmethod
    def prompt_func(question_case):
        encrypted_str = question_case['encrypted_str']
        k = question_case['k']
        prompt = f'''The encrypted string rules:
1. Letters followed by '?' can be removed or kept
2. Letters followed by '*' can be removed, kept, or repeated multiple times

Given string: {encrypted_str}
Target length: {k}

Provide a valid {k}-length message or "Impossible". Place your answer within [answer][/answer].'''
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        encrypted_str = identity['encrypted_str']
        k = identity['k']
        
        if solution == 'Impossible':
            return not cls._is_case_possible(encrypted_str, k)
        
        return (
            len(solution) == k and 
            cls._is_valid_solution(encrypted_str, solution)
        )
    
    @classmethod
    def _is_case_possible(cls, s, k):
        min_len = cls._calculate_min_length(s)
        max_len = cls._calculate_max_length(s)
        return min_len <= k <= max_len
    
    @classmethod
    def _is_valid_solution(cls, encrypted, candidate):
        ptr = 0
        i = 0
        while i < len(encrypted):
            if i+1 < len(encrypted) and encrypted[i+1] in ['?', '*']:
                # 处理带符号的字符
                char = encrypted[i]
                symbol = encrypted[i+1]
                i += 2
                
                # 查找候选字符串中的匹配情况
                count = 0
                while ptr < len(candidate) and candidate[ptr] == char:
                    ptr += 1
                    count += 1
                
                if symbol == '?':  # 0或1次
                    if count not in [0, 1]:
                        return False
                elif symbol == '*':  # 任意次数（含0）
                    if count < 0:
                        return False
            else:
                # 处理普通字符
                if ptr >= len(candidate) or candidate[ptr] != encrypted[i]:
                    return False
                ptr += 1
                i += 1
        
        return ptr == len(candidate)

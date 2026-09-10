"""# 

### 谜题描述
Sherlock Holmes found a mysterious correspondence of two VIPs and made up his mind to read it. But there is a problem! The correspondence turned out to be encrypted. The detective tried really hard to decipher the correspondence, but he couldn't understand anything. 

At last, after some thought, he thought of something. Let's say there is a word s, consisting of |s| lowercase Latin letters. Then for one operation you can choose a certain position p (1 ≤ p < |s|) and perform one of the following actions: 

  * either replace letter sp with the one that alphabetically follows it and replace letter sp + 1 with the one that alphabetically precedes it; 
  * or replace letter sp with the one that alphabetically precedes it and replace letter sp + 1 with the one that alphabetically follows it. 



Let us note that letter \"z\" doesn't have a defined following letter and letter \"a\" doesn't have a defined preceding letter. That's why the corresponding changes are not acceptable. If the operation requires performing at least one unacceptable change, then such operation cannot be performed.

Two words coincide in their meaning iff one of them can be transformed into the other one as a result of zero or more operations.

Sherlock Holmes needs to learn to quickly determine the following for each word: how many words can exist that coincide in their meaning with the given word, but differs from the given word in at least one character? Count this number for him modulo 1000000007 (109 + 7).

Input

The input data contains several tests. The first line contains the only integer t (1 ≤ t ≤ 104) — the number of tests.

Next t lines contain the words, one per line. Each word consists of lowercase Latin letters and has length from 1 to 100, inclusive. Lengths of words can differ.

Output

For each word you should print the number of different other words that coincide with it in their meaning — not from the words listed in the input data, but from all possible words. As the sought number can be very large, print its value modulo 1000000007 (109 + 7).

Examples

Input

1
ab


Output

1


Input

1
aaaaaaaaaaa


Output

0


Input

2
ya
klmbfxzb


Output

24
320092793

Note

Some explanations about the operation:

  * Note that for each letter, we can clearly define the letter that follows it. Letter \"b\" alphabetically follows letter \"a\", letter \"c\" follows letter \"b\", ..., \"z\" follows letter \"y\". 
  * Preceding letters are defined in the similar manner: letter \"y\" precedes letter \"z\", ..., \"a\" precedes letter \"b\". 
  * Note that the operation never changes a word's length. 



In the first sample you can obtain the only other word \"ba\". In the second sample you cannot obtain any other word, so the correct answer is 0.

Consider the third sample. One operation can transform word \"klmbfxzb\" into word \"klmcexzb\": we should choose p = 4, and replace the fourth letter with the following one (\"b\"  →  \"c\"), and the fifth one — with the preceding one (\"f\"  →  \"e\"). Also, we can obtain many other words from this one. An operation can transform word \"ya\" only into one other word \"xb\". 

Word \"ya\" coincides in its meaning with words \"xb\", \"wc\", \"vd\", ..., \"ay\" (overall there are 24 other words). The word \"klmbfxzb has many more variants — there are 3320092814 other words that coincide with in the meaning. So the answer for the first word equals 24 and for the second one equals 320092793 — the number 3320092814 modulo 109 + 7

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from math import  *
mod=1000000007
dp=[]
maxlen=101
maxV=101*25
dp.append([0]*maxV)
dp[0][0]=1
for i in range(1,maxlen):
    sum=0
    dp.append([0]*maxV)
    for j in range(0,maxV):
        sum+=dp[i-1][j]
        if(sum>=mod):
            sum-=mod
        if(j>=26):
            sum-=dp[i-1][j-26]
            if(sum<0):
                sum+=mod
        dp[i][j]=sum

testCase=int(raw_input())

def get(s):
    sum=0
    for ch in s:
        sum+=ord(ch)-ord('a')
    return sum
set={}
res=[]
ar=[]
for test in range(1,testCase+1):
    s=raw_input()
    v=get(s)
    res.append(dp[len(s)][v])
    num=maxV*101+len(s)
    if ~(num  in set):
        set[num]=0
    set[num]=set[num]+1
    ar.append(num)
for test in range(1,testCase+1):
    res[test-1]-=set[ar[test-1]]
    print res[test-1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import string
import re

class Ccipherbootcamp(Basebootcamp):
    _dp = None
    mod = 10**9 + 7

    def __init__(self, min_length=1, max_length=100, **kwargs):
        if min_length < 1 or max_length > 100:
            raise ValueError("Word length must be between 1 and 100")
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self._ensure_dp_initialized()

    @classmethod
    def _ensure_dp_initialized(cls):
        if cls._dp is None:
            cls._init_dp()

    @classmethod
    def _init_dp(cls):
        max_len = 101  # 支持长度0到100的单词
        maxV = 100*25 + 1  # 最大可能和为2500
        cls._dp = [[0] * maxV for _ in range(max_len)]
        cls._dp[0][0] = 1

        for length in range(1, max_len):
            current_sum = 0
            for s in range(maxV):
                # 累加前一行当前列的贡献
                current_sum += cls._dp[length-1][s]
                
                # 移除超过26窗口的部分
                if s >= 26:
                    current_sum -= cls._dp[length-1][s-26]
                
                # 处理模运算和负数
                current_sum %= cls.mod
                if current_sum < 0:
                    current_sum += cls.mod
                
                cls._dp[length][s] = current_sum

    def case_generator(self):
        length = random.randint(self.min_length, self.max_length)
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        char_sum = sum(ord(c) - ord('a') for c in word)
        
        if length >= len(self._dp) or char_sum >= len(self._dp[length]):
            return {'word': word, 'correct_answer': 0}
        
        total = self._dp[length][char_sum]
        return {
            'word': word,
            'correct_answer': (total - 1) % self.mod
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        word = question_case['word']
        return f"""根据以下规则计算等价单词数量：

**操作规则**
1. 选择位置p（1 ≤ p < 单词长度）
2. 执行以下任一操作：
   a) p位字母+1，p+1位字母-1
   b) p位字母-1，p+1位字母+1
注：'a'不可减，'z'不可加

**输入单词**
{word}

请输出满足条件的其他单词数量（模1e9+7），格式示例：[answer]0[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = identity['correct_answer']
            return (int(solution) % cls.mod) == expected
        except:
            return False

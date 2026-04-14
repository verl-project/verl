"""# 

### 谜题描述
Oleg the client and Igor the analyst are good friends. However, sometimes they argue over little things. Recently, they started a new company, but they are having trouble finding a name for the company.

To settle this problem, they've decided to play a game. The company name will consist of n letters. Oleg and Igor each have a set of n letters (which might contain multiple copies of the same letter, the sets can be different). Initially, the company name is denoted by n question marks. Oleg and Igor takes turns to play the game, Oleg moves first. In each turn, a player can choose one of the letters c in his set and replace any of the question marks with c. Then, a copy of the letter c is removed from his set. The game ends when all the question marks has been replaced by some letter.

For example, suppose Oleg has the set of letters {i, o, i} and Igor has the set of letters {i, m, o}. One possible game is as follows :

Initially, the company name is ???.

Oleg replaces the second question mark with 'i'. The company name becomes ?i?. The set of letters Oleg have now is {i, o}.

Igor replaces the third question mark with 'o'. The company name becomes ?io. The set of letters Igor have now is {i, m}.

Finally, Oleg replaces the first question mark with 'o'. The company name becomes oio. The set of letters Oleg have now is {i}.

In the end, the company name is oio.

Oleg wants the company name to be as lexicographically small as possible while Igor wants the company name to be as lexicographically large as possible. What will be the company name if Oleg and Igor always play optimally?

A string s = s1s2...sm is called lexicographically smaller than a string t = t1t2...tm (where s ≠ t) if si < ti where i is the smallest index such that si ≠ ti. (so sj = tj for all j < i)

Input

The first line of input contains a string s of length n (1 ≤ n ≤ 3·105). All characters of the string are lowercase English letters. This string denotes the set of letters Oleg has initially.

The second line of input contains a string t of length n. All characters of the string are lowercase English letters. This string denotes the set of letters Igor has initially.

Output

The output should contain a string of n lowercase English letters, denoting the company name if Oleg and Igor plays optimally.

Examples

Input

tinkoff
zscoder


Output

fzfsirk


Input

xxxxxx
xxxxxx


Output

xxxxxx


Input

ioi
imo


Output

ioi

Note

One way to play optimally in the first sample is as follows :

  * Initially, the company name is ???????.
  * Oleg replaces the first question mark with 'f'. The company name becomes f??????.
  * Igor replaces the second question mark with 'z'. The company name becomes fz?????.
  * Oleg replaces the third question mark with 'f'. The company name becomes fzf????.
  * Igor replaces the fourth question mark with 's'. The company name becomes fzfs???.
  * Oleg replaces the fifth question mark with 'i'. The company name becomes fzfsi??.
  * Igor replaces the sixth question mark with 'r'. The company name becomes fzfsir?.
  * Oleg replaces the seventh question mark with 'k'. The company name becomes fzfsirk.



For the second sample, no matter how they play, the company name will always be xxxxxx.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
first = raw_input()
second = raw_input()

first = sorted([i for i in first])
second = sorted([i for i in second],reverse = True)

n = len(first)

ans = ['']*n
l = 0
r = n-1

f = first[:n/2]
s = second[:n/2]

if n % 2 != 0:
    f += first[n/2]


fl = 0
fr = len(f)-1

sl = 0
sr = len(s)-1

index = 0

while True:
    if index % 2 == 0: # First player

        if index == n-1:
            ans[l] = f[fl]
            break
    
        if f[fl] >= s[sl]:
            ans[r] = f[fr]
            r -= 1
            fr -= 1
        else:
            ans[l] = f[fl]
            l += 1
            fl += 1
        


    else: # second
        if index == n-1:
            ans[l] = s[sl]
            break
        
        if s[sl] <= f[fl]:
            ans[r] = s[sr]
            
            r -= 1
            sr -= 1
        else:
            ans[l] = s[sl]

            l += 1
            sl += 1

        
    index += 1

    
print ''.join(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

class Cnamingcompanybootcamp(Basebootcamp):
    def __init__(self, **params):
        """支持动态参数配置"""
        self.n = params.get('n', 7)
        self.seed = params.get('seed', None)
        random.seed(self.seed) if self.seed else None

    def case_generator(self):
        """生成多样化测试案例"""
        # 生成原始输入（保持输入顺序）
        s = ''.join(random.choices(string.ascii_lowercase, k=self.n))
        t = ''.join(random.choices(string.ascii_lowercase, k=self.n))
        
        # 确保至少20%概率生成特殊案例
        if random.random() < 0.2:  
            # case1: 全等字符
            if random.choice([True, False]):
                char = random.choice(string.ascii_lowercase)
                return {'s': char*self.n, 't': char*self.n}
            # case2: 极值情况（一方全a一方全z）
            else:
                return {'s': 'a'*self.n, 't': 'z'*self.n}
        return {'s': s, 't': t}

    @staticmethod
    def prompt_func(case):
        """生成符合题设要求的完整问题描述"""
        return f"""Oleg和Igor的字母集分别为：
Oleg: {case['s']}
Igor: {case['t']}

根据博弈规则确定最终公司名称，将答案放在[answer]标签内。例如：[answer]abc[/answer]"""

    @staticmethod
    def extract_output(output):
        """增强答案提取鲁棒性"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, case):
        """严格验证答案正确性"""
        expected = cls._calculate_answer(case['s'], case['t'])
        return solution == expected

    @staticmethod
    def _calculate_answer(s, t):
        """与参考算法保持完全一致的实现"""
        first = sorted(s)
        second = sorted(t, reverse=True)
        n = len(first)
        ans = [''] * n
        
        split = n // 2
        f = first[:split]
        s_part = second[:split]
        
        if n % 2:
            f.append(first[split])
        
        l, r = 0, n-1
        fl, fr = 0, len(f)-1
        sl, sr = 0, len(s_part)-1
        
        for idx in range(n):
            if idx % 2 == 0:  # Oleg's turn
                if idx == n-1:
                    ans[l] = f[fl]
                    break
                if f[fl] >= s_part[sl]:
                    ans[r] = f[fr]
                    r -= 1
                    fr -= 1
                else:
                    ans[l] = f[fl]
                    l += 1
                    fl += 1
            else:  # Igor's turn
                if idx == n-1:
                    ans[l] = s_part[sl]
                    break
                if s_part[sl] <= f[fl]:
                    ans[r] = s_part[sr]
                    r -= 1
                    sr -= 1
                else:
                    ans[l] = s_part[sl]
                    l += 1
                    sl += 1
        return ''.join(ans)

"""# 

### 谜题描述
Berland scientists know that the Old Berland language had exactly n words. Those words had lengths of l1, l2, ..., ln letters. Every word consisted of two letters, 0 and 1. Ancient Berland people spoke quickly and didn’t make pauses between the words, but at the same time they could always understand each other perfectly. It was possible because no word was a prefix of another one. The prefix of a string is considered to be one of its substrings that starts from the initial symbol.

Help the scientists determine whether all the words of the Old Berland language can be reconstructed and if they can, output the words themselves.

Input

The first line contains one integer N (1 ≤ N ≤ 1000) — the number of words in Old Berland language. The second line contains N space-separated integers — the lengths of these words. All the lengths are natural numbers not exceeding 1000.

Output

If there’s no such set of words, in the single line output NO. Otherwise, in the first line output YES, and in the next N lines output the words themselves in the order their lengths were given in the input file. If the answer is not unique, output any.

Examples

Input

3
1 2 3


Output

YES
0
10
110


Input

3
1 1 1


Output

NO

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, a = input (), sorted (enumerate (map (int, raw_input ().split ())), key=lambda x: x[1])
pref, ans = 2, [None] * n
for i, x in a:
    pref <<= x - len (bin (pref)[3:]) 
    s = bin (pref)[3:]
    ans[i] = s 
    if None in ans and '0' not in s:
        print 'NO' 
        exit (0)
    pref += 1
print 'YES\n' + '\n'.join (ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Coldberlandlanguagebootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_length=1000):
        self.max_n = max(max_n, 1)
        self.max_length = max(max_length, 1)
    
    def case_generator(self):
        # 尝试生成有效案例的多种组合
        for _ in range(5):  # 有限次尝试避免死循环
            n = random.randint(1, min(20, self.max_n))  # 控制生成规模
            lengths = [random.randint(1, min(20, self.max_length)) for _ in range(n)]
            
            # 按照参考算法进行验证
            try:
                sorted_indices = sorted(enumerate(lengths), key=lambda x: x[1])
                pref = 2
                ans = [None] * n
                valid_case = True
                
                for idx, (original_idx, length) in enumerate(sorted_indices):
                    current_len = len(bin(pref)) - 3  # 当前前缀长度
                    shift_needed = length - current_len
                    
                    if shift_needed < 0:
                        valid_case = False
                        break
                    
                    pref <<= shift_needed
                    word = bin(pref)[3:]  # 去除'0b1'
                    
                    if len(word) != length or (None in ans and '0' not in word):
                        valid_case = False
                        break
                    
                    ans[original_idx] = word
                    pref += 1
                
                if valid_case and None not in ans:
                    # 最终验证前缀条件
                    prefix_valid = True
                    for i in range(n):
                        for j in range(n):
                            if i != j and (ans[i].startswith(ans[j]) or ans[j].startswith(ans[i])):
                                prefix_valid = False
                                break
                        if not prefix_valid:
                            break
                    if prefix_valid:
                        return {
                            'n': n,
                            'lengths': lengths,
                            'possible': True
                        }
            except:
                continue
        
        # 生成无效案例（如多个长度1或随机冲突）
        return {
            'n': random.randint(2, 10),
            'lengths': [1] * random.randint(2, 10),
            'possible': False
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        lengths = question_case['lengths']
        lengths_str = ' '.join(map(str, lengths))
        return (
            f"Determine if {n} binary words with lengths {lengths_str} can exist such that no word is a prefix of another. "
            f"If possible, output YES followed by the words in input order. Otherwise, output NO. "
            f"Format your answer within [answer]...[/answer] tags.\n\n"
            f"Example format:\n[answer]YES\n0\n10\n110[/answer] or [answer]NO[/answer]"
        )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if identity['possible']:
            if not solution.startswith("YES"):
                return False
            parts = solution.split('\n')
            if len(parts)-1 != identity['n']:
                return False
            words = parts[1:]
            # 验证长度和前缀条件
            return (
                all(len(w) == l for w, l in zip(words, identity['lengths'])) and
                not any(w1.startswith(w2) or w2.startswith(w1) 
                        for i, w1 in enumerate(words) 
                        for j, w2 in enumerate(words) if i != j)
            )
        else:
            return solution.strip().upper() == 'NO'

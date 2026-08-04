"""# 

### 谜题描述
Polycarp has n different binary words. A word called binary if it contains only characters '0' and '1'. For example, these words are binary: \"0001\", \"11\", \"0\" and \"0011100\".

Polycarp wants to offer his set of n binary words to play a game \"words\". In this game, players name words and each next word (starting from the second) must start with the last character of the previous word. The first word can be any. For example, these sequence of words can be named during the game: \"0101\", \"1\", \"10\", \"00\", \"00001\".

Word reversal is the operation of reversing the order of the characters. For example, the word \"0111\" after the reversal becomes \"1110\", the word \"11010\" after the reversal becomes \"01011\".

Probably, Polycarp has such a set of words that there is no way to put them in the order correspondent to the game rules. In this situation, he wants to reverse some words from his set so that:

  * the final set of n words still contains different words (i.e. all words are unique); 
  * there is a way to put all words of the final set of words in the order so that the final sequence of n words is consistent with the game rules. 



Polycarp wants to reverse minimal number of words. Please, help him.

Input

The first line of the input contains one integer t (1 ≤ t ≤ 10^4) — the number of test cases in the input. Then t test cases follow.

The first line of a test case contains one integer n (1 ≤ n ≤ 2⋅10^5) — the number of words in the Polycarp's set. Next n lines contain these words. All of n words aren't empty and contains only characters '0' and '1'. The sum of word lengths doesn't exceed 4⋅10^6. All words are different.

Guaranteed, that the sum of n for all test cases in the input doesn't exceed 2⋅10^5. Also, guaranteed that the sum of word lengths for all test cases in the input doesn't exceed 4⋅10^6.

Output

Print answer for all of t test cases in the order they appear.

If there is no answer for the test case, print -1. Otherwise, the first line of the output should contain k (0 ≤ k ≤ n) — the minimal number of words in the set which should be reversed. The second line of the output should contain k distinct integers — the indexes of the words in the set which should be reversed. Words are numerated from 1 to n in the order they appear. If k=0 you can skip this line (or you can print an empty line). If there are many answers you can print any of them.

Example

Input


4
4
0001
1000
0011
0111
3
010
101
0
2
00000
00001
4
01
001
0001
00001


Output


1
3 
-1
0

2
1 2 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sets import Set
T = int(raw_input())
for _ in xrange(T):
    n = int(raw_input())
    count = [0,0,0,0]
    swap = [[],[],[],[]]
    lines = [raw_input() for _ in xrange(n)]
    items = set(lines)
    for i, s in enumerate(lines):
        pos = (ord(s[0]) - ord('0')) * 2 + ord(s[-1]) - ord('0')
        count[pos] += 1
        if s[::-1] not in items:
            swap[pos].append(i+1)
    if count[1] > count[2]:
        count[1], count[2] = count[2], count[1]
        swap[1], swap[2] = swap[2], swap[1]
    if count[1]+count[2] is 0:
        print '-1' if count[0] > 0 and count[-1] > 0 else '0\n'
    else:
        diff = 0
        i = 1 if len(swap[1]) > len(swap[2]) else 2
        while count[2]-count[1] > 1:
            diff += 1
            count[2] -= 1
            count[1] += 1
        print diff
        print ' '.join(map(str, swap[i][:diff])) if len(swap[i]) >= diff else '-1'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

def solve_case(words):
    count = [0] * 4  # 00, 01, 10, 11
    swap = [[] for _ in range(4)]
    items = set(words)
    
    for i, s in enumerate(words):
        a = s[0]
        b = s[-1]
        pos = int(a) * 2 + int(b)
        count[pos] += 1
        reversed_s = s[::-1]
        if reversed_s not in items:
            swap[pos].append(i + 1)  # Using 1-based index
    
    if count[1] > count[2]:
        count[1], count[2] = count[2], count[1]
        swap[1], swap[2] = swap[2], swap[1]
    
    if count[1] + count[2] == 0:
        if count[0] > 0 and count[3] > 0:
            return (-1, None)
        else:
            return (0, [])
    else:
        diff = 0
        original_count_01 = count[1]
        original_count_10 = count[2]
        while count[2] - count[1] > 1:
            diff += 1
            count[2] -= 1
            count[1] += 1
        i = 1 if len(swap[1]) > len(swap[2]) else 2
        if len(swap[i]) >= diff:
            indexes = swap[i][:diff]
            return (diff, indexes)
        else:
            return (-1, None)

class Dletsplaythewordsbootcamp(Basebootcamp):
    def __init__(self, min_words=1, max_words=5, min_length=1, max_length=5, **kwargs):
        super().__init__(**kwargs)
        self.min_words = min_words
        self.max_words = max_words
        self.min_length = min_length
        self.max_length = max_length
    
    def case_generator(self):
        n = random.randint(self.min_words, self.max_words)
        existing = set()
        words = []
        for _ in range(n):
            while True:
                length = random.randint(self.min_length, self.max_length)
                word = ''.join(random.choice('01') for _ in range(length))
                if word not in existing:
                    break
            existing.add(word)
            words.append(word)
        correct_k, correct_indexes = solve_case(words)
        return {
            'n': n,
            'words': words,
            'correct_k': correct_k,
            'correct_indexes': correct_indexes if correct_k != -1 else None
        }
    
    @staticmethod
    def prompt_func(question_case):
        words = question_case['words']
        n = question_case['n']
        example_input = f"{n}\n" + '\n'.join(words)
        prompt = f"""Polycarp has a set of {n} distinct binary words. Your task is to reverse the minimal number of words so that all words remain unique and can be arranged in a sequence where each subsequent word starts with the last character of the previous word.

Input:
{example_input}

Output format:
- If impossible, output -1.
- If possible, output k (the minimal number of reversals). If k > 0, the next line should list the 1-based indices of the words to reverse.

Put your final answer within [answer] and [/answer] tags. Example:

[answer]
1
3 
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        if not lines:
            return None
        first_line = lines[0]
        if first_line == '-1':
            return -1
        try:
            k = int(first_line)
        except:
            return None
        if k < 0:
            return None
        if k == 0:
            return []
        if len(lines) < 2:
            return None
        indexes = []
        if not lines[1]:
            return None
        for s in lines[1].split():
            try:
                idx = int(s)
                indexes.append(idx)
            except:
                return None
        return indexes
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        words = identity['words']
        n = identity['n']
        if solution == -1:
            correct_k = identity['correct_k']
            if correct_k != -1:
                return False
            return True  # Assume case_generator's solve_case is correct
        if not isinstance(solution, list):
            return False
        k = len(solution)
        valid_indices = set(range(1, n+1))
        seen_indices = set()
        for idx in solution:
            if idx not in valid_indices or idx in seen_indices:
                return False
            seen_indices.add(idx)
        modified_words = []
        seen_words = set()
        for i, word in enumerate(words):
            idx = i + 1
            if idx in solution:
                modified = word[::-1]
            else:
                modified = word
            if modified in seen_words:
                return False
            seen_words.add(modified)
            modified_words.append(modified)
        count = defaultdict(int)
        for word in modified_words:
            start = word[0]
            end = word[-1]
            type_ = f"{start}{end}"
            if type_ == "00":
                count['00'] += 1
            elif type_ == "01":
                count['01'] += 1
            elif type_ == "10":
                count['10'] += 1
            elif type_ == "11":
                count['11'] += 1
        c01 = count['01']
        c10 = count['10']
        if abs(c01 - c10) > 1:
            return False
        if (c01 + c10) == 0:
            if count['00'] > 0 and count['11'] > 0:
                return False
        return True

"""# 

### 谜题描述
Everyone knows that DNA strands consist of nucleotides. There are four types of nucleotides: \"A\", \"T\", \"G\", \"C\". A DNA strand is a sequence of nucleotides. Scientists decided to track evolution of a rare species, which DNA strand was string s initially. 

Evolution of the species is described as a sequence of changes in the DNA. Every change is a change of some nucleotide, for example, the following change can happen in DNA strand \"AAGC\": the second nucleotide can change to \"T\" so that the resulting DNA strand is \"ATGC\".

Scientists know that some segments of the DNA strand can be affected by some unknown infections. They can represent an infection as a sequence of nucleotides. Scientists are interested if there are any changes caused by some infections. Thus they sometimes want to know the value of impact of some infection to some segment of the DNA. This value is computed as follows:

  * Let the infection be represented as a string e, and let scientists be interested in DNA strand segment starting from position l to position r, inclusive. 
  * Prefix of the string eee... (i.e. the string that consists of infinitely many repeats of string e) is written under the string s from position l to position r, inclusive. 
  * The value of impact is the number of positions where letter of string s coincided with the letter written under it. 



Being a developer, Innokenty is interested in bioinformatics also, so the scientists asked him for help. Innokenty is busy preparing VK Cup, so he decided to delegate the problem to the competitors. Help the scientists!

Input

The first line contains the string s (1 ≤ |s| ≤ 105) that describes the initial DNA strand. It consists only of capital English letters \"A\", \"T\", \"G\" and \"C\".

The next line contains single integer q (1 ≤ q ≤ 105) — the number of events.

After that, q lines follow, each describes one event. Each of the lines has one of two formats: 

  * 1 x c, where x is an integer (1 ≤ x ≤ |s|), and c is a letter \"A\", \"T\", \"G\" or \"C\", which means that there is a change in the DNA: the nucleotide at position x is now c. 
  * 2 l r e, where l, r are integers (1 ≤ l ≤ r ≤ |s|), and e is a string of letters \"A\", \"T\", \"G\" and \"C\" (1 ≤ |e| ≤ 10), which means that scientists are interested in the value of impact of infection e to the segment of DNA strand from position l to position r, inclusive. 

Output

For each scientists' query (second type query) print a single integer in a new line — the value of impact of the infection on the DNA.

Examples

Input

ATGCATGC
4
2 1 8 ATGC
2 2 6 TTT
1 4 T
2 2 6 TA


Output

8
2
4


Input

GAGTTGTTAA
6
2 3 4 TATGGTG
1 1 T
1 6 G
2 5 9 AGTAATA
1 10 G
2 2 6 TTGT


Output

0
3
1

Note

Consider the first example. In the first query of second type all characters coincide, so the answer is 8. In the second query we compare string \"TTTTT...\" and the substring \"TGCAT\". There are two matches. In the third query, after the DNA change, we compare string \"TATAT...\"' with substring \"TGTAT\". There are 4 matches.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

class segtree:
    def __init__(s, data):
        s.n = len(data)
        s.m = 1
        while s.m < s.n: s.m *= 2
        s.data = [0]*s.m
        s.data += data
        for i in reversed(range(2, s.n + s.m)):
            s.data[i >> 1] += s.data[i]

    def add(s, i, x):
        i += s.m
        while i:
            s.data[i] += x
            i >>= 1

    def summa(s, l, r):
        l += s.m
        r += s.m

        tot = 0
        while l<r:
            if l & 1:
                tot += s.data[l]
                l += 1
            if r & 1:
                tot += s.data[r - 1]

            l >>= 1
            r >>= 1
        return tot

def trans(x):
    if x == 'A':
        return 0
    elif x == 'C':
        return 1
    elif x == 'G':
        return 2
    return 3

inp = sys.stdin.read().split()
ii = 0

S = [trans(x) for x in inp[ii]]
ii += 1

segs =  [
            [
                [
                    segtree([+(s==c) for s in S[j::i]]) for c in range(4)
                ] 
                for j in range(i)
            ] 
            for i in range(11)
        ]

q = int(inp[ii])
ii += 1
out = []
for _ in range(q):
    c = inp[ii]
    ii += 1
    if c == '1':
        ind = int(inp[ii]) - 1
        ii += 1
        char = trans(inp[ii])
        ii += 1

        old_char = S[ind]
        for i in range(1,11):
            seg = segs[i][ind % i]
            seg[old_char].add(ind // i, -1)
            seg[char].add(ind // i, 1)

        S[ind] = char
    else:
        l = int(inp[ii]) - 1
        ii += 1
        r = int(inp[ii])
        ii += 1

        e = inp[ii]
        ii += 1

        tot = 0

        i = len(e)
        for k in range(i):
            L = l + k
            start = L//i
            length = (r - L + i - 1)//i
            tot += segs[i][L % i][trans(e[k])].summa(start, start + length)
        out.append(tot)

print '\n'.join(str(x) for x in out)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class segtree:
    def __init__(self, data):
        self.n = len(data)
        self.m = 1
        while self.m < self.n:
            self.m *= 2
        self.data = [0] * (2 * self.m)
        for i in range(self.n):
            self.data[self.m + i] = data[i]
        for i in range(self.m - 1, 0, -1):
            self.data[i] = self.data[2 * i] + self.data[2 * i + 1]
    
    def add(self, i, delta):
        pos = self.m + i
        self.data[pos] += delta
        while pos > 1:
            pos >>= 1
            self.data[pos] += delta
    
    def summa(self, l, r):
        res = 0
        l += self.m
        r += self.m
        while l < r:
            if l % 2 == 1:
                res += self.data[l]
                l += 1
            if r % 2 == 1:
                r -= 1
                res += self.data[r]
            l >>= 1
            r >>= 1
        return res

trans_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def compute_answers(s, events):
    S = [trans_dict[c] for c in s]
    n = len(S)
    
    segs = [[] for _ in range(11)]  # segs[0] unused
    for i in range(1, 11):
        segs_i = []
        for j in range(i):
            segs_ij = []
            for c in range(4):
                data = []
                for k in range(j, n, i):
                    data.append(1 if S[k] == c else 0)
                segs_ij.append(segtree(data))
            segs_i.append(segs_ij)
        segs[i] = segs_i
    
    outputs = []
    for event in events:
        if event['type'] == 1:
            x = event['x'] - 1
            c = trans_dict[event['c']]
            old_char = S[x]
            for i in range(1, 11):
                j = x % i
                pos = x // i
                segs[i][j][old_char].add(pos, -1)
                segs[i][j][c].add(pos, 1)
            S[x] = c
        else:
            l = event['l'] - 1
            r = event['r']
            e = event['e']
            i = len(e)
            e_trans = [trans_dict[c] for c in e]
            tot = 0
            for k in range(i):
                c = e_trans[k]
                L = l + k
                if L >= r:
                    continue
                j = L % i
                start = L // i
                end = (r - 1 - j) // i + 1
                tot += segs[i][j][c].summa(start, end)
            outputs.append(tot)
    return outputs

class Cdnaevolutionbootcamp(Basebootcamp):
    def __init__(self, max_s_length=1000, min_events=3, max_events=10, query_ratio=0.7, **kwargs):
        super().__init__(**kwargs)
        self.max_s_length = max_s_length
        self.min_events = min_events
        self.max_events = max_events
        self.query_ratio = query_ratio
    
    def case_generator(self):
        chars = ['A', 'T', 'G', 'C']
        n = random.randint(10, self.max_s_length)
        s = ''.join(random.choices(chars, k=n))
        
        q = random.randint(self.min_events, self.max_events)
        events = []
        query_count = 0
        for _ in range(q):
            # Ensure at least one query and enforce query ratio
            if query_count == 0 or random.random() < self.query_ratio:
                event_type = 2
                query_count += 1
            else:
                event_type = 1
            
            if event_type == 1:
                x = random.randint(1, n)
                c = random.choice(chars)
                events.append({'type': 1, 'x': x, 'c': c})
            else:
                l = random.randint(1, n)
                r = random.randint(l, n)
                e_len = random.randint(1, 10)
                e = ''.join(random.choices(chars, k=e_len))
                events.append({'type': 2, 'l': l, 'r': r, 'e': e})
        
        # Ensure at least one query exists
        if query_count == 0:
            events[-1] = {'type': 2, 'l': 1, 'r': n, 'e': 'A'}
        
        try:
            expected_outputs = compute_answers(s, events)
        except Exception as e:
            return self.case_generator()
        
        case = {
            's': s,
            'events': events,
            'expected_outputs': expected_outputs
        }
        return case
    
    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        events = question_case['events']
        event_lines = []
        for evt in events:
            if evt['type'] == 1:
                event_lines.append(f"1 {evt['x']} {evt['c']}")
            else:
                event_lines.append(f"2 {evt['l']} {evt['r']} {evt['e']}")
        events_str = '\n'.join(event_lines)
        prompt = f"""You are analyzing DNA evolution. Given initial DNA and events, compute impact values for each query.

Initial DNA: {s}

Events (each line is 1=change or 2=query):
{events_str}

For EACH TYPE 2 EVENT, calculate how many positions match between:
1. DNA segment from l to r (inclusive)
2. Infinite repetition of infection string e

Output ONLY the numerical answers in order, each on a new line, enclosed in [answer] tags.

Example:
[answer]
5
0
[/answer]

Your answers:"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        last_block = answer_blocks[-1].strip()
        numbers = []
        for line in last_block.splitlines():
            clean_line = line.strip()
            if clean_line:
                if clean_line.isdigit() or (clean_line.startswith('-') and clean_line[1:].isdigit()):
                    numbers.append(int(clean_line))
        return numbers if numbers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('expected_outputs', [])
        return isinstance(solution, list) and solution == expected

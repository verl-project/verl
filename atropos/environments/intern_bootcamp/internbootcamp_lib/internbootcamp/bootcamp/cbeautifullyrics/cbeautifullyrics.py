"""# 

### 谜题描述
You are given n words, each of which consists of lowercase alphabet letters. Each word contains at least one vowel. You are going to choose some of the given words and make as many beautiful lyrics as possible.

Each lyric consists of two lines. Each line consists of two words separated by whitespace. 

A lyric is beautiful if and only if it satisfies all conditions below. 

  * The number of vowels in the first word of the first line is the same as the number of vowels in the first word of the second line. 
  * The number of vowels in the second word of the first line is the same as the number of vowels in the second word of the second line. 
  * The last vowel of the first line is the same as the last vowel of the second line. Note that there may be consonants after the vowel. 



Also, letters \"a\", \"e\", \"o\", \"i\", and \"u\" are vowels. Note that \"y\" is never vowel.

For example of a beautiful lyric, 

\"hello hellooowww\" 

\"whatsup yowowowow\" 

is a beautiful lyric because there are two vowels each in \"hello\" and \"whatsup\", four vowels each in \"hellooowww\" and \"yowowowow\" (keep in mind that \"y\" is not a vowel), and the last vowel of each line is \"o\".

For example of a not beautiful lyric, 

\"hey man\"

\"iam mcdic\" 

is not a beautiful lyric because \"hey\" and \"iam\" don't have same number of vowels and the last vowels of two lines are different (\"a\" in the first and \"i\" in the second).

How many beautiful lyrics can you write from given words? Note that you cannot use a word more times than it is given to you. For example, if a word is given three times, you can use it at most three times.

Input

The first line contains single integer n (1 ≤ n ≤ 10^{5}) — the number of words.

The i-th of the next n lines contains string s_{i} consisting lowercase alphabet letters — the i-th word. It is guaranteed that the sum of the total word length is equal or less than 10^{6}. Each word contains at least one vowel.

Output

In the first line, print m — the number of maximum possible beautiful lyrics.

In next 2m lines, print m beautiful lyrics (two lines per lyric).

If there are multiple answers, print any.

Examples

Input


14
wow
this
is
the
first
mcdics
codeforces
round
hooray
i
am
proud
about
that


Output


3
about proud
hooray round
wow first
this is
i that
mcdics am


Input


7
arsijo
suggested
the
idea
for
this
problem


Output


0


Input


4
same
same
same
differ


Output


1
same differ
same same

Note

In the first example, those beautiful lyrics are one of the possible answers. Let's look at the first lyric on the sample output of the first example. \"about proud hooray round\" forms a beautiful lyric because \"about\" and \"hooray\" have same number of vowels, \"proud\" and \"round\" have same number of vowels, and both lines have same last vowel. On the other hand, you cannot form any beautiful lyric with the word \"codeforces\".

In the second example, you cannot form any beautiful lyric from given words.

In the third example, you can use the word \"same\" up to three times.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout
MAX = 100010

n = input()
inp = stdin.readlines()

nvow = []
lvow = []
vowl = 'aeiou'

for line in inp:
    s = line.strip()
    cnt = 0
    vx = ''
    for c in s:
        if c in vowl:
            vx = c
            cnt += 1        

    while len(nvow) <= cnt:
        nvow.append([])
        lvow.append([])
        for i in xrange(6): lvow[-1].append([])

    nvow[cnt].append(s)

    idx = 0
    for i in xrange(5):
        idx += 1
        if vowl[i] == vx: break
    lvow[cnt][idx].append(s)

lyrics = []
for i in xrange(len(nvow)):
    for j in xrange(6):
        while len(lvow[i][j]) >= 2:
            x = lvow[i][j].pop()
            y = lvow[i][j].pop()
            lyrics.append(['', x, '', y])

ctr = 0
sz = len(lyrics)

for i in xrange(len(nvow)):
    if ctr >= sz: break
    
    same = []
    for j in xrange(6):
        while len(lvow[i][j]) > 0:
            x = lvow[i][j].pop()
            same.append(x)
    
    while ctr < sz:
        if len(same) < 2: break 
        while len(same) >= 2:
            u = same.pop()
            v = same.pop()
            lyrics[ctr][0] = u
            lyrics[ctr][2] = v
            ctr += 1
            if ctr >= sz: break
    
pp = []
i = sz - 1
while i >= ctr:
    zz = lyrics.pop()
    u, v = zz[1], zz[3]
    pp.append([u, v])
    i -= 1

while len(pp) >= 2:
    xx = pp.pop()
    yy = pp.pop()
    lyrics.append([xx[0], yy[0], xx[1], yy[1]])

m = len(lyrics)
print m
out = []
for l in lyrics:
    x, y, z, w = l
    out.append(' '.join([x, y]))
    out.append(' '.join([z, w]))

stdout.write(\"\n\".join(out))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from collections import defaultdict
from bootcamp import Basebootcamp

def count_vowels_and_last_vowel(word):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    last_v = None
    for c in word:
        if c in vowels:
            count += 1
            last_v = c
    return count, last_v

def extract_all_vowels(word):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    return [c for c in word if c in vowels]

class Cbeautifullyricsbootcamp(Basebootcamp):
    def __init__(self, max_words=10, min_words=4, mode='solvable'):
        self.max_words = max_words
        self.min_words = min_words
        self.mode = mode  # 'solvable' or 'unsolvable'

    def case_generator(self):
        if self.mode == 'solvable':
            words, expected_m = self._generate_solvable_case()
        else:
            words, expected_m = self._generate_unsolvable_case()
        return {
            'n': len(words),
            'words': words,
            'expected_m': expected_m,
            'word_counts': defaultdict(int, {word: words.count(word) for word in words})
        }

    def _generate_solvable_case(self):
        # Generate at least one valid lyric
        words = []
        # Generate one valid lyric with 4 unique words
        a = self._generate_word_with_vowel_count(2)
        c = self._generate_word_with_vowel_count(2)
        b = self._generate_word_with_vowel_count_and_last(1, 'a')
        d = self._generate_word_with_vowel_count_and_last(1, 'a')
        words.extend([a, b, c, d])
        # Ensure expected_m is 1 for this simple case
        expected_m = 1
        return words, expected_m

    def _generate_unsolvable_case(self):
        # Generate words that cannot form any lyric
        words = [
            self._generate_word_with_vowel_count_and_last(1, 'a'),
            self._generate_word_with_vowel_count_and_last(2, 'e'),
            self._generate_word_with_vowel_count_and_last(3, 'i'),
            self._generate_word_with_vowel_count_and_last(4, 'o')
        ]
        return words, 0

    def _generate_word_with_vowel_count(self, count):
        vowels = ['a', 'e', 'i', 'o', 'u']
        other = [chr(c) for c in range(ord('a'), ord('z')+1) if chr(c) not in vowels]
        parts = []
        for _ in range(count):
            parts.append(random.choice(vowels))
            if random.random() < 0.5 and len(parts) < count * 2:
                parts.append(random.choice(other))
        # Add trailing consonants
        for _ in range(random.randint(0, 3)):
            parts.append(random.choice(other))
        return ''.join(parts)

    def _generate_word_with_vowel_count_and_last(self, count, last_vowel):
        vowels = ['a', 'e', 'i', 'o', 'u']
        other = [chr(c) for c in range(ord('a'), ord('z')+1) if chr(c) not in vowels]
        parts = []
        # Generate count-1 vowels
        for _ in range(count-1):
            parts.append(random.choice(vowels))
            if random.random() < 0.5:
                parts.append(random.choice(other))
        # Add the last vowel
        parts.append(last_vowel)
        # Add trailing consonants
        for _ in range(random.randint(0, 3)):
            parts.append(random.choice(other))
        return ''.join(parts)

    @staticmethod
    def prompt_func(question_case):
        words = question_case['words']
        example_input = "\n".join(words)
        prompt = f"""You are given {question_case['n']} words, each consisting of lowercase letters. Each word contains at least one vowel. Your task is to form as many beautiful lyrics as possible.

A beautiful lyric consists of two lines. Each line has two words separated by a space. The conditions are:
1. The number of vowels in the first word of each line must be equal.
2. The number of vowels in the second word of each line must be equal.
3. The last vowel in the entire first line must be the same as the last vowel in the entire second line.

Vowels are 'a', 'e', 'i', 'o', 'u' (excluding 'y').

Input:
{question_case['n']}
{example_input}

Output format:
- The first line is the maximum number of lyrics, m.
- Followed by 2m lines, each pair forming a lyric.

Put your answer within [answer] tags. Example:
[answer]2[/answer]
word1 word2
word3 word4
word5 word6
word7 word8"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        return last_match

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        lines = [line.strip() for line in solution.split('\n') if line.strip()]
        if not lines:
            return False
        
        try:
            m = int(lines[0])
        except:
            return False
        
        if m != identity['expected_m']:
            return False
        
        if len(lines) != 1 + 2 * m:
            return False
        
        lyrics = []
        for i in range(m):
            line1 = lines[1 + 2*i].split()
            line2 = lines[2 + 2*i].split()
            if len(line1) != 2 or len(line2) != 2:
                return False
            lyrics.append((line1, line2))
        
        # Check each lyric's conditions
        for (first_line, second_line) in lyrics:
            a1, a2 = first_line
            b1, b2 = second_line

            # Condition 1
            a1_v, _ = count_vowels_and_last_vowel(a1)
            b1_v, _ = count_vowels_and_last_vowel(b1)
            if a1_v != b1_v:
                return False

            # Condition 2
            a2_v, _ = count_vowels_and_last_vowel(a2)
            b2_v, _ = count_vowels_and_last_vowel(b2)
            if a2_v != b2_v:
                return False

            # Condition 3
            line1_vowels = extract_all_vowels(a1) + extract_all_vowels(a2)
            line2_vowels = extract_all_vowels(b1) + extract_all_vowels(b2)
            if not line1_vowels or not line2_vowels:
                return False
            if line1_vowels[-1] != line2_vowels[-1]:
                return False

        # Check word usage counts
        word_counts = defaultdict(int)
        for word in identity['words']:
            word_counts[word] += 1
        used = defaultdict(int)
        for (line1, line2) in lyrics:
            for word in line1 + line2:
                used[word] += 1
                if used[word] > word_counts.get(word, 0):
                    return False

        return True

"""# 

### 谜题描述
After observing the results of Spy Syndrome, Yash realised the errors of his ways. He now believes that a super spy such as Siddhant can't use a cipher as basic and ancient as Caesar cipher. After many weeks of observation of Siddhant’s sentences, Yash determined a new cipher technique.

For a given sentence, the cipher is processed as: 

  1. Convert all letters of the sentence to lowercase. 
  2. Reverse each of the words of the sentence individually. 
  3. Remove all the spaces in the sentence. 



For example, when this cipher is applied to the sentence

Kira is childish and he hates losing

the resulting string is

ariksihsidlihcdnaehsetahgnisol

Now Yash is given some ciphered string and a list of words. Help him to find out any original sentence composed using only words from the list. Note, that any of the given words could be used in the sentence multiple times.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 10 000) — the length of the ciphered text. The second line consists of n lowercase English letters — the ciphered text t.

The third line contains a single integer m (1 ≤ m ≤ 100 000) — the number of words which will be considered while deciphering the text. Each of the next m lines contains a non-empty word wi (|wi| ≤ 1 000) consisting of uppercase and lowercase English letters only. It's guaranteed that the total length of all words doesn't exceed 1 000 000.

Output

Print one line — the original sentence. It is guaranteed that at least one solution exists. If there are multiple solutions, you may output any of those.

Examples

Input

30
ariksihsidlihcdnaehsetahgnisol
10
Kira
hates
is
he
losing
death
childish
L
and
Note


Output

Kira is childish and he hates losing 


Input

12
iherehtolleh
5
HI
Ho
there
HeLLo
hello


Output

HI there HeLLo 

Note

In sample case 2 there may be multiple accepted outputs, \"HI there HeLLo\" and \"HI there hello\" you may output any of them. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAX = 100010;
int lower(int character) {
  if ('a' <= character && character <= 'z') return character;
  return character - 'A' + 'a';
}
struct Trie {
  Trie *link[26];
  int ind;
  Trie(int ind) : ind(ind) {
    for (int i = 0; i < 26; ++i) link[i] = NULL;
  }
} * root;
string S, a[MAX];
int dp[MAX], pre[MAX], ans[MAX], N, M;
void add(string a, int ind) {
  int len = a.size();
  reverse(a.begin(), a.end());
  Trie *q, *p = root;
  for (int i = 0, j; i < len; ++i) {
    j = lower(a[i]) - 'a';
    q = p->link[j];
    if (q == NULL) p->link[j] = q = new Trie(-1);
    p = q;
  }
  p->ind = ind;
}
void openFile() { freopen(\"in.inp\", \"r\", stdin); }
int main() {
  ios::sync_with_stdio(false);
  cin >> N;
  cin >> S;
  cin >> M;
  root = new Trie(-1);
  for (int i = 0; i < M; ++i) {
    cin >> a[i];
    add(a[i], i);
  }
  reverse(S.begin(), S.end());
  S = \"#\" + S;
  dp[0] = 1;
  Trie *p;
  for (int i = 1, j; i <= N; ++i) {
    p = root;
    j = i;
    dp[i] = 0;
    while (j >= 1 && dp[i] == 0) {
      p = p->link[S[j] - 'a'];
      if (p == NULL) break;
      if (p->ind != -1 && dp[j - 1]) {
        dp[i] = 1;
        pre[i] = j - 1;
        ans[i] = p->ind;
        break;
      }
      --j;
    }
  }
  int i = N;
  while (i > 0) {
    cout << a[ans[i]];
    i = pre[i];
    if (i) cout << \" \";
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class Cspysyndrome2bootcamp(Basebootcamp):
    def __init__(self, min_words=3, max_words=5, word_pool=None, extra_words=2, max_word_length=8):
        self.min_words = min_words
        self.max_words = max_words
        self.word_pool = word_pool or [
            'Apple', 'Banana', 'Cherry', 'Hello', 'World',
            'Test', 'Example', 'Python', 'Code', 'Spy',
            'Cipher', 'Hacker', 'Secret', 'Message', 'Encode'
        ]
        self.extra_words = extra_words
        self.max_word_length = max_word_length

    def case_generator(self):
        # 1. Generate original sentence with varied cases
        word_count = random.randint(self.min_words, self.max_words)
        original_words = [
            self._random_case(random.choice(self.word_pool))
            for _ in range(word_count)
        ]
        
        # 2. Generate ciphered text
        cipher_parts = [word.lower()[::-1] for word in original_words]
        ciphered = ''.join(cipher_parts)
        
        # 3. Prepare valid words with guaranteed solutions
        valid_words = list(set(  # Ensure unique
            [self._random_case(w) for w in original_words] +
            [w.lower() for w in original_words]  # Guarantee lowercase variant
        ))
        
        # 4. Add decoy words (length filtered)
        decoy_pool = [w for w in self.word_pool if len(w) <= self.max_word_length]
        decoy_words = [
            self._random_case(random.choice(decoy_pool))
            for _ in range(self.extra_words)
        ]
        
        return {
            'ciphered': ciphered,
            'words': valid_words + decoy_words,
            '_solution': ' '.join(original_words)  # For validation reference
        }

    @staticmethod
    def _random_case(word):
        variants = [
            word.lower(),
            word.upper(),
            word.capitalize(),
            ''.join([c.upper() if random.random() > 0.5 else c.lower() for c in word])
        ]
        return random.choice(variants)

    @staticmethod
    def prompt_func(question_case):
        return f"""You are a codebreaker analyzing a ciphertext. Decrypt it using the following rules:
1. Convert all letters to lowercase
2. Reverse each word individually
3. Remove all spaces

Ciphertext (length {len(question_case['ciphered'])}):
{question_case['ciphered']}

Valid words ({len(question_case['words'])} options):
""" + '\n'.join(f"- {w}" for w in question_case['words']) + """

Requirements:
- Use EXACTLY the provided words (case-sensitive)
- No extra spaces

Put your answer between [answer] and [/answer]. Example:
[answer]Hello World[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, flags=re.DOTALL|re.IGNORECASE)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # Normalize solution
            candidate = solution.strip().replace('  ', ' ')
            words = candidate.split()
            
            # Check all words are valid
            if any(w not in identity['words'] for w in words):
                return False
                
            # Reconstruct cipher
            reconstructed = ''.join([w.lower()[::-1] for w in words])
            return reconstructed == identity['ciphered']
            
        except Exception:
            return False

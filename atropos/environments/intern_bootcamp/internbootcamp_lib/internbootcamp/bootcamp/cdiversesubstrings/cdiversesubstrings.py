"""# 

### 谜题描述
String diversity is the number of symbols that occur in the string at least once. Diversity of s will be denoted by d(s). For example , d(\"aaa\")=1, d(\"abacaba\")=3.

Given a string s, consisting of lowercase Latin letters. Consider all its substrings. Obviously, any substring diversity is a number from 1 to d(s). Find statistics about substrings diversity: for each k from 1 to d(s), find how many substrings of s has a diversity of exactly k.

Input

The input consists of a single line containing s. It contains only lowercase Latin letters, the length of s is from 1 to 3·105.

Output

Print to the first line the value d(s). Print sequence t1, t2, ..., td(s) to the following lines, where ti is the number of substrings of s having diversity of exactly i.

Examples

Input

abca


Output

3
4
3
3


Input

aabacaabbad


Output

4
14
19
28
5

Note

Consider the first example.

We denote by s(i, j) a substring of \"abca\" with the indices in the segment [i, j].

  * s(1, 1) =  \"a\", d(\"a\") = 1
  * s(2, 2) =  \"b\", d(\"b\") = 1
  * s(3, 3) =  \"c\", d(\"c\") = 1
  * s(4, 4) =  \"a\", d(\"a\") = 1
  * s(1, 2) =  \"ab\", d(\"ab\") = 2
  * s(2, 3) =  \"bc\", d(\"bc\") = 2
  * s(3, 4) =  \"ca\", d(\"ca\") = 2
  * s(1, 3) =  \"abc\", d(\"abc\") = 3
  * s(2, 4) =  \"bca\", d(\"bca\") = 3
  * s(1, 4) =  \"abca\", d(\"abca\") = 3



Total number of substring with diversity 1 is 4, with diversity 2 equals 3, 3 diversity is 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using ull = unsigned long long;
using vi = vector<int>;
using vl = vector<long>;
using vll = vector<ll>;
using vb = vector<bool>;
using vvb = vector<vb>;
using vvi = vector<vector<int> >;
using ii = pair<int, int>;
using vii = vector<ii>;
using vs = vector<string>;
using msi = map<string, int>;
using iss = istringstream;
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  string s;
  cin >> s;
  int n = s.length();
  for (int i = int(0); i <= int(n - 1); i++) s[i] -= 'a';
  vb seen(26, false);
  for (int i = int(0); i <= int(n - 1); i++) seen[s[i]] = true;
  int ds = accumulate(seen.begin(), seen.end(), 0);
  vll ans(ds + 1, 0);
  for (int k = int(1); k <= int(ds); k++) {
    vector<deque<int> > dq(26, deque<int>());
    int l = 0, r = -1;
    int uniq = 0;
    int upos = 0;
    ll cnt = 0;
    while (l < n) {
      while (true) {
        if (r == n - 1) break;
        if (uniq == k && !dq[s[r + 1]].size() > 0) break;
        r++;
        if (dq[s[r]].size() == 0) {
          uniq++;
          upos = r;
        }
        dq[s[r]].push_back(r);
        if (uniq == k) cnt++;
      }
      ans[k] += cnt;
      dq[s[l]].pop_front();
      if (dq[s[l]].size() == 0) {
        uniq--;
        cnt = 0;
      } else {
        int nupos = max(upos, dq[s[l]][0]);
        cnt = max(0ll, cnt - (nupos - upos));
        upos = nupos;
      }
      l++;
    }
  }
  cout << ds << '\n';
  for (int k = int(1); k <= int(ds); k++) cout << ans[k] << '\n';
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
from collections import defaultdict, deque
from bootcamp import Basebootcamp

class Cdiversesubstringsbootcamp(Basebootcamp):
    def __init__(self, min_length=4, max_length=100):
        self.min_length = max(min_length, 1)
        self.max_length = max(max_length, self.min_length)

    def case_generator(self):
        # Generate string with controlled diversity
        length = random.randint(self.min_length, self.max_length)
        max_possible_d = min(26, length)
        target_d = random.randint(1, max_possible_d)
        
        # Ensure exactly target_d distinct characters
        base_chars = random.sample(string.ascii_lowercase, target_d)
        s_list = base_chars.copy()
        
        # Fill remaining length with random choices from base chars
        for _ in range(length - target_d):
            s_list.append(random.choice(base_chars))
        
        random.shuffle(s_list)
        s = ''.join(s_list)
        
        # Calculate actual d and t_list using optimized algorithm
        d = len(set(s))
        t_list = self.compute_t_list(s)
        
        return {
            's': s,
            'd': d,
            't_list': t_list
        }

    def compute_t_list(self, s):
        n = len(s)
        s = [ord(c) - ord('a') for c in s]
        total_d = len(set(s))
        ans = [0] * (total_d + 1)  # ans[0] unused
        
        for k in range(1, total_d + 1):
            count = defaultdict(int)
            distinct = 0
            left = 0
            res = 0
            
            for right in range(n):
                c = s[right]
                if count[c] == 0:
                    distinct += 1
                count[c] += 1
                
                while distinct > k:
                    left_c = s[left]
                    count[left_c] -= 1
                    if count[left_c] == 0:
                        distinct -= 1
                    left += 1
                
                res += right - left + 1
            
            prev_res = 0
            if k > 1:
                prev_res = self.at_most_k(s, k-1)
            ans[k] = res - prev_res
        
        return [ans[k] for k in range(1, total_d+1)]

    def at_most_k(self, s, k):
        count = defaultdict(int)
        distinct = 0
        left = 0
        res = 0
        
        for right in range(len(s)):
            c = s[right]
            if count[c] == 0:
                distinct += 1
            count[c] += 1
            
            while distinct > k:
                left_c = s[left]
                count[left_c] -= 1
                if count[left_c] == 0:
                    distinct -= 1
                left += 1
            
            res += right - left + 1
        
        return res

    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        return f"""You are given a string consisting of lowercase Latin letters. The diversity of a string is defined as the number of distinct characters it contains. Your task is to calculate two things: 
1. The diversity value d(s) of the given string.
2. For each k from 1 to d(s), determine how many substrings have exactly k distinct characters.

Input string: {s}

Output Format:
- First line: d(s)
- Next d(s) lines: Each line contains the count of substrings with diversity exactly k (k from 1 to d(s))

Enclose your final answer within [answer] and [/answer] tags. For example:
[answer]
3
4
3
3
[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        numbers = []
        last_answer = matches[-1].strip()
        for line in last_answer.split('\n'):
            line = line.strip()
            if line:
                try:
                    numbers.append(int(line))
                except ValueError:
                    return None
        
        return numbers if numbers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list) or len(solution) != identity['d'] + 1:
            return False
        if solution[0] != identity['d']:
            return False
        return solution[1:] == identity['t_list']

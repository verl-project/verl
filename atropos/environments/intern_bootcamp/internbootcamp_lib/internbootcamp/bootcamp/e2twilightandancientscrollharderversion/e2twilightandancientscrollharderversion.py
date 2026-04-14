"""# 

### 谜题描述
This is a harder version of the problem E with larger constraints.

Twilight Sparkle has received a new task from Princess Celestia. This time she asked to decipher the ancient scroll containing important knowledge of pony origin.

To hide the crucial information from evil eyes, pony elders cast a spell on the scroll. That spell adds exactly one letter in any place to each word it is cast on. To make the path to the knowledge more tangled elders chose some of words in the scroll and cast a spell on them.

Twilight Sparkle knows that the elders admired the order in all things so the scroll original scroll contained words in lexicographically non-decreasing order. She is asked to delete one letter from some of the words of the scroll (to undo the spell) to get some version of the original scroll. 

Unfortunately, there may be more than one way to recover the ancient scroll. To not let the important knowledge slip by Twilight has to look through all variants of the original scroll and find the required one. To estimate the maximum time Twilight may spend on the work she needs to know the number of variants she has to look through. She asks you to find that number! Since that number can be very big, Twilight asks you to find it modulo 10^9+7.

It may occur that princess Celestia has sent a wrong scroll so the answer may not exist.

A string a is lexicographically smaller than a string b if and only if one of the following holds:

  * a is a prefix of b, but a ≠ b;
  * in the first position where a and b differ, the string a has a letter that appears earlier in the alphabet than the corresponding letter in b.

Input

The first line contains a single integer n (1 ≤ n ≤ 10^5): the number of words in the scroll.

The i-th of the next n lines contains a string consisting of lowercase English letters: the i-th word in the scroll. The length of each word is at least one. The sum of lengths of words does not exceed 10^6.

Output

Print one integer: the number of ways to get a version of the original from the scroll modulo 10^9+7.

Examples

Input


3
abcd
zaza
ataka


Output


4


Input


4
dfs
bfs
sms
mms


Output


8


Input


3
abc
bcd
a


Output


0


Input


6
lapochka
kartyshka
bigbabytape
morgenshtern
ssshhhiiittt
queen


Output


2028

Note

Notice that the elders could have written an empty word (but they surely cast a spell on it so it holds a length 1 now).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
template <typename T>
bool ckmax(T& a, const T& b) {
  return a < b ? a = b, 1 : 0;
}
template <typename T>
bool ckmin(T& a, const T& b) {
  return b < a ? a = b, 1 : 0;
}
using ll = long long;
const int BS = 5.65e7;
const int MOD = 1e9 + 7;
const int INF = 1e7;
const int MN = 1e5 + 10;
const int LG = 30;
const int MS = 1e6 + LG + 100;
const int MK = 26;
struct mint {
 public:
  int v;
  mint(int v = 0) : v(v) {}
  mint& operator+=(const mint& o) {
    if ((v += o.v) >= MOD) v -= MOD;
    return *this;
  }
  mint& operator-=(const mint& o) {
    if ((v -= o.v) < 0) v += MOD;
    return *this;
  }
  mint& operator*=(const mint& o) {
    v = (int)((ll)v * o.v % MOD);
    return *this;
  }
  friend mint operator+(mint a, const mint& b) { return a += b; }
  friend mint operator-(mint a, const mint& b) { return a -= b; }
  friend mint operator*(mint a, const mint& b) { return a *= b; }
  explicit operator int() const { return v; }
};
mint f;
int N, S, B, X, a[MN], buf[BS], all[LG + 1], Q;
int *c = buf + 1, *d = buf + 52 * MS, *l = buf + 54 * MS, *n = buf;
int *hd = buf + MS, *nx = buf + 3 * MS, *sa = buf + 5 * MS, *dw = buf + 6 * MS;
int *isa = buf, *lcp = buf + MS, *rmq = buf + 2 * MS, *mask = buf + 6 * MS;
int* o = buf + 7 * MS;
mint* dp = (mint*)(buf + 10 * MS);
int* q = buf + 12 * MS;
char s[MS], u[MS * 2];
int append(int p, char x) {
  int n = ++X;
  d[n] = d[p] + 1;
  for (; ~p && !~c[p * MK + x - 'a']; p = l[p]) c[p * MK + x - 'a'] = n;
  if (!~p)
    l[n] = 0;
  else {
    int q = c[p * MK + x - 'a'];
    if (d[q] == d[p] + 1)
      l[n] = q;
    else {
      ++X;
      l[X] = l[q], d[X] = d[p] + 1;
      memcpy(c + X * MK, c + q * MK, MK * sizeof *c);
      l[n] = l[q] = X;
      for (; ~p && c[p * MK + x - 'a'] == q; p = l[p]) c[p * MK + x - 'a'] = X;
    }
  }
  return n;
}
void adde(int a, int b) {
  nx[b] = hd[a];
  hd[a] = b;
}
void dfs() {
  Q = 0;
  q[Q++] = 0;
  for (int n; Q;) {
    n = q[--Q];
    int c = 0;
    int* k = dw + dw[0];
    for (int i = hd[n]; ~i; i = nx[i]) k[c++] = i;
    dw[0] += c;
    if (c) {
      std::sort(k, k + c, [](int x, int y) { return u[x] > u[y]; });
      for (int i = 0; i < c; ++i) q[Q++] = k[i];
    } else
      sa[X++] = S - d[n];
  }
}
void build_sa_psuedolinear() {
  memset(c - 1, -1, (S * 2 * MK + 1) * sizeof *c);
  d[0] = 0, l[0] = -1;
  for (int i = S - 1, n = 0; i >= 0; --i) n = append(n, s[i]);
  u[0] = -1;
  for (int i = 0, cn = 0; i < S; ++i)
    cn = n[i + 1] = c[cn * MK + s[S - i - 1] - 'a'];
  for (int i = S - 1, n = 0, x, p, r; i >= 0; --i) {
    n = ::n[S - i];
    x = n;
    r = S;
    for (; !u[x]; x = p) {
      p = l[x];
      r -= d[x] - d[p];
      u[x] = s[r];
    }
  }
  memset(hd, -1, 2 * S * sizeof *hd);
  for (int i = 1; i <= X; ++i) adde(l[i], i);
  X = 0;
  dw[0] = 1;
  dfs();
}
void build_lcp() {
  for (int i = 0; i < S; ++i) isa[sa[i]] = i;
  for (int i = 0, p = 0; i < S; ++i)
    if (isa[i] == S - 1)
      p = 0;
    else {
      for (; s[i + p] == s[sa[isa[i] + 1] + p]; ++p)
        ;
      lcp[isa[i]] = p;
      if (p) --p;
    }
  B = S / LG;
  for (int i = 0; i <= B; ++i) {
    rmq[i * LG] = INF;
    Q = 0;
    for (int j = 0, m = 0; j < LG; ++j) {
      ckmin(rmq[i * LG], lcp[i * LG + j]);
      for (; Q && lcp[i * LG + j] <= lcp[i * LG + q[Q - 1]];) m ^= 1 << q[--Q];
      q[Q++] = j, m |= 1 << j;
      mask[i * LG + j + 1] = m;
    }
  }
  for (int i = B; i >= 0; --i)
    for (int j = 0; i + (1 << j + 1) <= B + 1; ++j)
      rmq[i * LG + (j + 1)] =
          std::min(rmq[i * LG + j], rmq[(i + (1 << j)) * LG + j]);
}
int get_lcp_small(int rm, int v) {
  return lcp[v / LG * LG + __builtin_ctz(mask[v] & ~all[rm])];
}
int get_lcp(int u, int v) {
  u = isa[u], v = isa[v];
  if (u == v) return INF;
  if (v < u) std::swap(u, v);
  int u2 = u / LG + 1, v2 = v / LG, f = INF;
  if (v2 < u2) return lcp[v2 * LG + __builtin_ctz(mask[v] & ~all[u - v2 * LG])];
  if (u2 < v2) {
    int d = 31 - __builtin_clz(v2 - u2);
    f = std::min(rmq[u2 * LG + d], rmq[(v2 - (1 << d)) * LG + d]);
  }
  ckmin(f, lcp[(u2 - 1) * LG +
               __builtin_ctz(mask[u2 * LG] & ~all[u - (u2 - 1) * LG])]);
  if (v2 * LG < v) ckmin(f, lcp[v2 * LG + __builtin_ctz(mask[v])]);
  return f;
}
struct SubStr {
 public:
  int i, l;
};
struct Multi {
 public:
  SubStr v[2];
  void add(int i, int j) {
    bool id = v[0].i >= 0;
    if (i < j) v[id] = {i, j - i};
  }
  int size() const { return (v[0].i >= 0) + (v[1].i >= 0); }
  bool operator<(const Multi& o) const {
    int i = 0, j = 0, x = 0, y = 0, sz = size(), osz = o.size();
    for (; i < sz && j < osz;) {
      int l = std::min(
          {get_lcp(v[i].i + x, o.v[j].i + y), v[i].l - x, o.v[j].l - y});
      if (l < v[i].l - x && l < o.v[j].l - y)
        return s[v[i].i + x + l] < s[o.v[j].i + y + l];
      if ((x += l) >= v[i].l) ++i, x = 0;
      if ((y += l) >= o.v[j].l) ++j, y = 0;
    }
    return (i < sz) < (j < osz);
  }
};
Multi* val = (Multi*)(buf + 20 * MS);
int main() {
  for (int i = 0; i <= LG; ++i) all[i] = (1 << i) - 1;
  scanf(\"%d\", &N);
  for (int i = 0; i < N; ++i) {
    scanf(\" %s\", s + a[i]);
    for (a[i + 1] = a[i] + 1; s[a[i + 1]]; ++a[i + 1])
      ;
  }
  S = a[N];
  s[S++] = 'a' - 1;
  build_sa_psuedolinear();
  build_lcp();
  memset(val, -1, (N + S) * sizeof *val);
  for (int i = 0; i < N; ++i) {
    val[a[i + 1] + i].add(a[i], a[i + 1]);
    for (int j = a[i]; j < a[i + 1]; ++j) {
      val[j + i].add(a[i], j);
      val[j + i].add(j + 1, a[i + 1]);
    }
    {
      int st = a[i] + i, en = a[i + 1] + i + 1;
      nx[a[i + 1] - 1] = a[i + 1];
      for (int j = a[i + 1] - 2; j >= a[i]; --j)
        nx[j] = s[j] == s[j + 1] ? nx[j + 1] : j + 1;
      for (int j = a[i]; j < a[i + 1]; ++j)
        if (nx[j] < a[i + 1] && s[j] < s[nx[j]])
          o[--en] = j + i;
        else
          o[st++] = j + i;
      o[st] = a[i + 1] + i;
    }
    if (i) {
      mint p(0);
      int j = a[i - 1] + i - 1;
      for (int k = a[i] + i; k < a[i + 1] + i + 1; ++k) {
        for (; j < a[i] + i && !(val[o[k]] < val[o[j]]); ++j) p += dp[j];
        dp[k] = p;
      }
    } else
      for (int i = 0; i < a[1] + 1; ++i) dp[i] = mint(1);
  }
  for (int i = a[N - 1] + N - 1; i < a[N] + N; ++i) f += dp[i];
  printf(\"%d\n\", f);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

class E2twilightandancientscrollharderversionbootcamp(Basebootcamp):
    MOD = 10**9 + 7

    def __init__(self, max_n=5, max_word_length=5, **params):
        super().__init__(**params)
        self.max_n = max_n
        self.max_word_length = max_word_length

    def case_generator(self):
        n = random.randint(1, self.max_n)
        s_list = []
        prev = ""
        for _ in range(n):
            valid = False
            for _ in range(100):
                if not s_list:
                    # Allow empty string for first word
                    length = random.randint(0, self.max_word_length)
                    if length == 0:
                        s = ''
                    else:
                        s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
                else:
                    # Generate s >= prev
                    min_len = max(len(prev), 1)  # At least 1 character to have space for insertion
                    max_len = self.max_word_length
                    length = random.randint(min_len, max_len)
                    
                    # Construct s to be >= prev
                    prefix = list(prev)
                    if len(prev) < length:
                        prefix += ['a'] * (length - len(prev))
                    else:
                        length = len(prev)
                    
                    change_pos = random.randint(0, length-1)
                    for i in range(change_pos, length):
                        if i < len(prev):
                            min_char = prev[i]
                        else:
                            min_char = 'a'
                        new_char = random.choice([chr(c) for c in range(ord(min_char), ord('z')+1)])
                        prefix[i] = new_char
                        if i >= len(prev) or ''.join(prefix[:i+1]) > prev[:i+1]:
                            # Fill remaining with 'a'
                            for j in range(i+1, length):
                                prefix[j] = 'a'
                            break
                    s = ''.join(prefix)
                
                if s >= prev:
                    prev = s
                    valid = True
                    break
            if not valid:
                s = prev  # Fall back to previous valid string
            s_list.append(s)
        
        t_list = []
        expected = 1
        for s in s_list:
            if len(s) >= self.max_word_length:
                # Cannot add character, use fallback
                ways = 1
                if s:
                    t = s + (s[-1] if random.random() < 0.5 else 'z')
                else:
                    t = 'a'
                t_list.append(t)
                expected = expected * ways % self.MOD
                continue

            if not s:
                # Original was empty, add one character
                t = 'a'
                ways = 1
            else:
                # Decide insertion type
                if random.random() < 0.5:
                    # Insert duplicate character to create multiple solutions
                    if len(s) == 0:
                        t = 'aa'
                        ways = 2
                    else:
                        insert_pos = random.randint(0, len(s)-1)
                        duplicate_char = s[insert_pos]
                        t = s[:insert_pos] + duplicate_char + s[insert_pos:]
                        # Count possible positions that restore original
                        ways = 0
                        for i in range(len(t)):
                            if t[:i] + t[i+1:] == s:
                                ways += 1
                        if ways == 0:
                            # Fallback to simple append
                            last_char = s[-1]
                            c = chr(ord(last_char) + 1) if last_char < 'z' else 'z'
                            t = s + c
                            ways = 1
                else:
                    # Simple append with larger character
                    last_char = s[-1] if s else 'a'
                    c = chr(ord(last_char) + 1) if last_char < 'z' else 'z'
                    t = s + c
                    ways = 1
            
                # Validate t length
                if len(t) > self.max_word_length:
                    t = s + (s[-1] if s else 'a')
                    ways = 1

            t_list.append(t)
            expected = expected * ways % self.MOD

        return {
            "n": len(t_list),
            "words": t_list,
            "expected": expected
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        words = question_case['words']
        problem = (
            "Twilight Sparkle has received a scroll where each word had exactly one letter added. "
            "Your task is to determine how many ways you can delete one letter from each word to restore "
            "the original lexicographically non-decreasing order.\n\n"
            "**Input:**\n"
            f"{n}\n" + '\n'.join(words) + "\n\n"
            "**Output:**\n"
            "A single integer representing the number of valid ways modulo 10^9+7.\n\n"
            "Place your final answer within [answer] and [/answer] tags."
        )
        return problem

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        numbers = re.findall(r'-?\d+', last_match)
        if not numbers:
            return None
        try:
            return int(numbers[-1])
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']

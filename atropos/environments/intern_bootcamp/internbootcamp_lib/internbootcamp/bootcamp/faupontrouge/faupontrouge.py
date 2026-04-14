"""# 

### 谜题描述
VK just opened its second HQ in St. Petersburg! Side of its office building has a huge string s written on its side. This part of the office is supposed to be split into m meeting rooms in such way that meeting room walls are strictly between letters on the building. Obviously, meeting rooms should not be of size 0, but can be as small as one letter wide. Each meeting room will be named after the substring of s written on its side.

<image>

For each possible arrangement of m meeting rooms we ordered a test meeting room label for the meeting room with lexicographically minimal name. When delivered, those labels got sorted backward lexicographically.

What is printed on kth label of the delivery?

Input

In the first line, you are given three integer numbers n, m, k — length of string s, number of planned meeting rooms to split s into and number of the interesting label (2 ≤ n ≤ 1 000; 1 ≤ m ≤ 1 000; 1 ≤ k ≤ 10^{18}).

Second input line has string s, consisting of n lowercase english letters.

For given n, m, k there are at least k ways to split s into m substrings.

Output

Output single string – name of meeting room printed on k-th label of the delivery.

Examples

Input


4 2 1
abac


Output


aba


Input


19 5 1821
aupontrougevkoffice


Output


au

Note

In the first example, delivery consists of the labels \"aba\", \"ab\", \"a\".

In the second example, delivery consists of 3060 labels. The first label is \"aupontrougevkof\" and the last one is \"a\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using ull = unsigned long long;
constexpr int MOD = 1000 * 1000 * 1000 + 7;
constexpr int ALPHABET_SIZE = 26;
constexpr ll INF = 1e18;
bool check(const string& s, ll k, vector<vector<ll>>& dp, vector<int>& cont,
           const string& cand, int m) {
  int n = s.length(), l = cand.length();
  if (n < l) {
    return false;
  }
  for (int i = 0; i < n; ++i) {
    int pos = i;
    while (pos < n && pos - i < l && s[pos] == cand[pos - i]) {
      ++pos;
    }
    if ((pos < n && pos - i < l && s[pos] < cand[pos - i]) ||
        (pos == n && pos - i < l)) {
      cont[i] = -1;
    } else if (pos == n || pos - i == l) {
      cont[i] = pos;
    } else {
      cont[i] = pos + 1;
    }
  }
  for (int i = 0; i < n; ++i) {
    dp[i].assign(m, 0);
  }
  if (cont[0] != -1) {
    dp[cont[0] - 1][0] = 1;
  }
  for (int i = 0; i < n - 1; ++i) {
    for (int j = 0; j < m; ++j) {
      dp[i + 1][j] = min(k, dp[i][j] + dp[i + 1][j]);
    }
    if (cont[i + 1] == -1) {
      continue;
    }
    for (int j = 0; j + 1 < m; ++j) {
      dp[cont[i + 1] - 1][j + 1] =
          min(k, dp[cont[i + 1] - 1][j + 1] + dp[i][j]);
    }
  }
  return dp[n - 1][m - 1] > 0 && dp[n - 1][m - 1] >= k;
}
struct TreeNode {
  int have = 0;
  int interm = 0;
  array<int, ALPHABET_SIZE> nxt{};
};
string find_in_trie(const vector<TreeNode>& trie, int ind) {
  string ans;
  int have = 0;
  int cur = 0;
  while (cur != -1) {
    if (have + trie[cur].interm > ind) {
      return ans;
    }
    have += trie[cur].interm;
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
      int nxt = trie[cur].nxt[i];
      if (nxt == -1) {
        continue;
      }
      if (have + trie[nxt].have > ind) {
        ans.push_back('a' + i);
        cur = nxt;
        break;
      } else {
        have += trie[nxt].have;
      }
    }
  }
  return ans;
}
int main() {
  ios_base::sync_with_stdio(false);
  int n, m;
  ll k;
  cin >> n >> m >> k;
  string s;
  cin >> s;
  vector<vector<ll>> dp(n, vector<ll>(m));
  vector<int> cont(n);
  vector<TreeNode> nodes;
  nodes.emplace_back();
  nodes.back().nxt.fill(-1);
  for (int i = 0; i < n; ++i) {
    int cur = 0;
    nodes[0].have += n - i;
    for (int j = i; j < n; ++j) {
      if (nodes[cur].nxt[s[j] - 'a'] == -1) {
        nodes[cur].nxt[s[j] - 'a'] = nodes.size();
        nodes.emplace_back();
        nodes.back().nxt.fill(-1);
      }
      cur = nodes[cur].nxt[s[j] - 'a'];
      nodes[cur].have += n - j;
      ++nodes[cur].interm;
    }
  }
  int st = 0, e = (n * n + n) / 2;
  while (e - st > 1) {
    int mid = (st + e) / 2;
    string cand = find_in_trie(nodes, mid);
    if (check(s, k, dp, cont, cand, m)) {
      st = mid;
    } else {
      e = mid;
    }
  }
  cout << find_in_trie(nodes, st) << \"\n\";
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import math
from itertools import combinations

class Faupontrougebootcamp(Basebootcamp):
    class TreeNode:
        def __init__(self):
            self.have = 0
            self.interm = 0
            self.nxt = [-1] * 26  # For each lowercase letter a-z

    def __init__(self, **params):
        # Default parameters with dynamic generation capability
        self.n = params.get('n', 4)
        self.m = params.get('m', 2)
        self.k = params.get('k', 1)
        self.s = params.get('s', 'abac')
        
        # Ensure parameters are valid
        self.n = len(self.s)
        if not 1 <= self.m <= self.n:
            raise ValueError("m must be between 1 and string length")
        max_k = math.comb(self.n-1, self.m-1)
        if self.k > max_k:
            raise ValueError(f"k exceeds maximum possible value {max_k}")

    def case_generator(self):
        """Generate a valid puzzle case ensuring adequate combinations"""
        # Generate random case parameters with constraints
        import random
        import string
        
        # Random string length between 4-8 for manageability
        n = random.randint(4, 8)
        s = ''.join(random.choices(string.ascii_lowercase, k=n))
        
        # Ensure m is at least 1 and less than n
        m = random.randint(1, min(3, n-1))
        
        # Calculate maximum possible k
        max_k = math.comb(n-1, m-1)
        k = random.randint(1, max(1, max_k//2))  # Use conservative k
        
        return {
            'n': n,
            'm': m,
            'k': k,
            's': s
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        params = question_case
        return f"""VK's building has string '{params['s']}' (length {params['n']}). Split it into {params['m']} rooms. For each split, take the lex-smallest substring. Sort these minimal substrings in reverse lex order. What's the {params['k']}-th element?

Output must be wrapped in [answer] and [/answer] tags. Example: [answer]abc[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # Reimplement C++ logic for validation
            s = identity['s']
            m = identity['m']
            k = identity['k']
            n = len(s)

            # Build trie structure
            nodes = [cls.TreeNode()]
            for i in range(n):
                cur = 0
                nodes[cur].have += (n - i)
                for j in range(i, n):
                    c = ord(s[j]) - ord('a')
                    if nodes[cur].nxt[c] == -1:
                        nodes.append(cls.TreeNode())
                        nodes[cur].nxt[c] = len(nodes) - 1
                        nodes[-1].nxt = [-1]*26
                    cur = nodes[cur].nxt[c]
                    nodes[cur].have += (n - j)
                    nodes[cur].interm += 1

            # Binary search with DP verification
            left, right = 0, (n * (n + 1)) // 2
            answer = ""
            while left < right:
                mid = (left + right + 1) // 2
                candidate = cls.find_in_trie(nodes, mid)
                if cls.check_valid(s, k, candidate, m):
                    left = mid
                    answer = candidate
                else:
                    right = mid - 1
            return solution == answer
        except:
            return False

    @classmethod
    def find_in_trie(cls, nodes, idx):
        result = []
        cur = 0
        have = 0
        while True:
            have += nodes[cur].interm
            if have > idx:
                return ''.join(result)
            found = False
            for i in range(26):
                next_node = nodes[cur].nxt[i]
                if next_node == -1:
                    continue
                if have + nodes[next_node].have > idx:
                    result.append(chr(ord('a') + i))
                    cur = next_node
                    found = True
                    break
                else:
                    have += nodes[next_node].have
            if not found:
                break
        return ''.join(result)

    @classmethod
    def check_valid(cls, s, k, candidate, m):
        n = len(s)
        l = len(candidate)
        cont = [-1] * n

        # Precompute continuation points
        for i in range(n):
            pos = i
            while pos < n and pos - i < l and s[pos] == candidate[pos - i]:
                pos += 1
            if pos < n and pos - i < l and s[pos] < candidate[pos - i]:
                cont[i] = -1
            elif pos == n or pos - i == l:
                cont[i] = pos
            else:
                cont[i] = pos + 1 if pos < n else -1

        # DP table initialization
        dp = [[0]*m for _ in range(n)]
        if cont[0] != -1 and cont[0] <= n:
            end = cont[0] - 1
            if end < n:
                dp[end][0] = 1

        for i in range(n-1):
            for j in range(m):
                dp[i+1][j] = min(k, dp[i+1][j] + dp[i][j])
            
            if cont[i+1] == -1:
                continue
            
            for j in range(m-1):
                if dp[i][j] == 0:
                    continue
                next_pos = cont[i+1] - 1
                if next_pos >= n or j+1 >= m:
                    continue
                dp[next_pos][j+1] = min(k, dp[next_pos][j+1] + dp[i][j])

        return dp[n-1][m-1] >= k

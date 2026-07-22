"""# 

### 谜题描述
The Shuseki Islands are an archipelago of 30001 small islands in the Yutampo Sea. The islands are evenly spaced along a line, numbered from 0 to 30000 from the west to the east. These islands are known to contain many treasures. There are n gems in the Shuseki Islands in total, and the i-th gem is located on island pi.

Mr. Kitayuta has just arrived at island 0. With his great jumping ability, he will repeatedly perform jumps between islands to the east according to the following process: 

  * First, he will jump from island 0 to island d. 
  * After that, he will continue jumping according to the following rule. Let l be the length of the previous jump, that is, if his previous jump was from island prev to island cur, let l = cur - prev. He will perform a jump of length l - 1, l or l + 1 to the east. That is, he will jump to island (cur + l - 1), (cur + l) or (cur + l + 1) (if they exist). The length of a jump must be positive, that is, he cannot perform a jump of length 0 when l = 1. If there is no valid destination, he will stop jumping. 



Mr. Kitayuta will collect the gems on the islands visited during the process. Find the maximum number of gems that he can collect.

Input

The first line of the input contains two space-separated integers n and d (1 ≤ n, d ≤ 30000), denoting the number of the gems in the Shuseki Islands and the length of the Mr. Kitayuta's first jump, respectively.

The next n lines describe the location of the gems. The i-th of them (1 ≤ i ≤ n) contains a integer pi (d ≤ p1 ≤ p2 ≤ ... ≤ pn ≤ 30000), denoting the number of the island that contains the i-th gem.

Output

Print the maximum number of gems that Mr. Kitayuta can collect.

Examples

Input

4 10
10
21
27
27


Output

3


Input

8 8
9
19
28
36
45
55
66
78


Output

6


Input

13 7
8
8
9
16
17
17
18
21
23
24
24
26
30


Output

4

Note

In the first sample, the optimal route is 0  →  10 (+1 gem)  →  19  →  27 (+2 gems)  → ...

In the second sample, the optimal route is 0  →  8  →  15  →  21 →  28 (+1 gem)  →  36 (+1 gem)  →  45 (+1 gem)  →  55 (+1 gem)  →  66 (+1 gem)  →  78 (+1 gem)  → ...

In the third sample, the optimal route is 0  →  7  →  13  →  18 (+1 gem)  →  24 (+2 gems)  →  30 (+1 gem)  → ...

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
FILE* _fin = stdin;
FILE* _fout = stdout;
int _min(int a, int b) { return a <= b ? a : b; }
int _min(long long a, long long b) { return a <= b ? a : b; }
int _max(int a, int b) { return a >= b ? a : b; }
long long _max(long long a, long long b) { return a >= b ? a : b; }
void zero(int* data, int n) { memset(data, 0, sizeof(int) * n); }
void zero(long long* data, int n) { memset(data, 0, sizeof(long long) * n); }
void zero(char* data, int n) { memset(data, 0, sizeof(char) * n); }
char readc() {
  char var;
  fscanf(_fin, \"%c\", &var);
  return var;
}
int readi() {
  int var;
  fscanf(_fin, \"%d\", &var);
  return var;
}
double readlf() {
  double var;
  fscanf(_fin, \"%lf\", &var);
  return var;
}
long long readll() {
  long long var;
  fscanf(_fin, \"%lld\", &var);
  return var;
}
void repread(int* data, int n) {
  for (int i = 0; i < n; ++i) data[i] = readi();
}
void repread(long long* data, int n) {
  for (int i = 0; i < n; ++i) data[i] = readll();
}
int reads(char* str, int maxsize) {
  for (;;) {
    if (fgets(str, maxsize, _fin) == NULL) break;
    if (str[0] != '\n' && str[0] != '\r') break;
  }
  int slen = strlen(str);
  if (slen == 0) return 0;
  if (str[slen - 1] == '\n' || str[slen - 1] == '\r') str[--slen] = 0;
  return slen;
}
long long gcd(long long a, long long b) {
  if (b == 0) return a;
  return gcd(b, a % b);
}
long long lcm(long long a, long long b) {
  long long g = gcd(a, b);
  return (a / g) * b;
}
void reverse(char* data, int n) {
  int k = n >> 1;
  for (int i = 0; i < k; ++i) {
    char tmp = data[i];
    data[i] = data[n - i - 1];
    data[n - i - 1] = tmp;
  }
}
void reverse(int* data, int n) {
  int k = n >> 1;
  for (int i = 0; i < k; ++i) {
    int tmp = data[i];
    data[i] = data[n - i - 1];
    data[n - i - 1] = tmp;
  }
}
void reverse(long long* data, int n) {
  int k = n >> 1;
  for (int i = 0; i < k; ++i) {
    long long tmp = data[i];
    data[i] = data[n - i - 1];
    data[n - i - 1] = tmp;
  }
}
struct Veci {
  int* data;
  int size;
  int n;
};
void init(Veci* t, int size) {
  t->data = (int*)malloc(sizeof(int) * size);
  t->size = size;
  t->n = 0;
}
void resize(Veci* t) {
  int ns = t->size * 1.2f;
  t->data = (int*)realloc(t->data, sizeof(int) * ns);
  t->size = ns;
}
void add(Veci* t, int val) {
  if (t->n >= t->size) resize(t);
  int k = t->n;
  t->data[k] = val;
  t->n = k + 1;
}
void free(Veci* t) { free(t->data); }
struct Vecll {
  long long* data;
  int size;
  int n;
};
void init(Vecll* t, int size) {
  t->data = (long long*)malloc(sizeof(long long) * size);
  t->size = size;
  t->n = 0;
}
void resize(Vecll* t) {
  int ns = t->size * 1.2f;
  t->data = (long long*)realloc(t->data, sizeof(long long) * ns);
  t->size = ns;
}
void add(Vecll* t, long long val) {
  if (t->n >= t->size) resize(t);
  int k = t->n;
  t->data[k] = val;
  t->n = k + 1;
}
void free(Vecll* t) { free(t->data); }
struct Vecs {
  char** data;
  int size;
  int n;
};
void init(Vecs* t, int size) {
  t->data = (char**)malloc(sizeof(char*) * size);
  t->size = size;
  t->n = 0;
}
void resize(Vecs* t) {
  int ns = t->size * 1.2f;
  t->data = (char**)realloc(t->data, sizeof(char*) * ns);
  t->size = ns;
}
void add(Vecs* t, char* val) {
  if (t->n >= t->size) resize(t);
  int k = t->n;
  t->data[k] = val;
  t->n = k + 1;
}
void free(Vecs* t) { free(t->data); }
int ispali(int* a, int* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[n - i - 1]) {
      return 0;
    }
  }
  return 1;
}
int ispalc(char* a, char* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[n - i - 1]) {
      return 0;
    }
  }
  return 1;
}
struct Pairi {
  int x, y;
};
int cmp_Pairi(const void* _a, const void* _b) {
  Pairi* a = (Pairi*)_a;
  Pairi* b = (Pairi*)_b;
  if (a->x < b->x)
    return -1;
  else {
    if (a->x == b->x) {
      if (a->y == b->y) return 0;
      if (a->y < b->y)
        return -1;
      else
        return 1;
    } else {
      return 1;
    }
  }
}
void sort_Pairi(Pairi* d, int n) { qsort(d, n, sizeof(Pairi), cmp_Pairi); }
struct Pairlf {
  double x, y;
};
int cmp_Pairlf(const void* _a, const void* _b) {
  Pairlf* a = (Pairlf*)_a;
  Pairlf* b = (Pairlf*)_b;
  if (a->x < b->x)
    return -1;
  else {
    if (a->x == b->x) {
      if (a->y == b->y) return 0;
      if (a->y < b->y)
        return -1;
      else
        return 1;
    } else {
      return 1;
    }
  }
}
void sort_Pairlf(Pairlf* d, int n) { qsort(d, n, sizeof(Pairlf), cmp_Pairlf); }
int cmp_Str(const void* _a, const void* _b) {
  char* a = *((char**)_a);
  char* b = *((char**)_b);
  return strcmp(a, b);
}
void sort_Str(char** str, int n) { qsort(str, n, sizeof(char*), cmp_Str); }
long long expmod(long long x, long long n, long long m) {
  long long ans = 1;
  for (; n;) {
    if (n & 1) ans = (ans * x) % m;
    x = (x * x) % m;
    n >>= 1;
  }
  return ans;
}
long long combmod(long long n, long long k, long long m) {
  long long ret = 1;
  long long div = 1;
  for (long long i = 0; i < k; ++i) {
    ret = (ret * (n - i) % m) % m;
    div = (div * (i + 1)) % m;
  }
  div = expmod(div, m - 2, m) % m;
  return (ret * div) % m;
}
int next_perm(int* data, int n) {
  int rootidx = -1;
  for (int i = n - 2; i >= 0; --i) {
    if (data[i] < data[i + 1]) {
      rootidx = i;
      break;
    }
  }
  if (rootidx == -1) return 0;
  int ceilingidx = rootidx + 1;
  for (int i = rootidx + 2; i < n; ++i) {
    if (data[i] > data[rootidx] && data[i] < data[ceilingidx]) {
      ceilingidx = i;
    }
  }
  {
    int t = data[rootidx];
    data[rootidx] = data[ceilingidx];
    data[ceilingidx] = t;
  };
  std::sort(&data[rootidx + 1], &data[rootidx + 1] + n - rootidx - 1);
  return 1;
}
int count_digits(long long a) {
  int k = 0;
  for (;;) {
    if (a == 0) break;
    k++;
    a /= 10;
  }
  return k;
}
int bs_exists(int* data, int n, int val) {
  int left = 0;
  int right = n - 1;
  for (; left < right;) {
    int mid = (left + right + 1) >> 1;
    if (data[mid] <= val)
      left = mid;
    else
      right = mid - 1;
  }
  return right;
}
double simple_factorial(int x) {
  double f = 1.0;
  for (int i = 0; i < x; ++i) {
    f *= (i + 1);
  }
  return f;
}
const long long MOD = 1e9 + 7;
int pos(int x) { return x; }
int main() {
  {
    _fin = fopen(\"input.txt\", \"r\");
    if (_fin == NULL) _fin = stdin;
  };
  int n = readi();
  int d = readi();
  static int a[30001];
  static int dp[30001][600];
  zero(a, 30001);
  for (int i = 0; i < n; ++i) {
    int x = readi();
    a[x]++;
  }
  for (int i = 0; i < 30001; ++i) {
    for (int j = 0; j < 600; ++j) dp[i][j] = -1;
  }
  dp[d][300] = a[d];
  int ans = a[d];
  for (int i = d; i < 30001; ++i) {
    for (int jump = 1; jump < 600; ++jump) {
      if (dp[i][jump] < 0) continue;
      if (dp[i][jump] > ans) {
        ans = dp[i][jump];
      }
      int k = i + jump + d - 300;
      if (k >= d && k <= 30000)
        dp[k][jump] = _max(dp[k][jump], dp[i][jump] + a[k]);
      k += 1;
      if (k >= d && k <= 30000)
        dp[k][jump + 1] = _max(dp[k][jump + 1], dp[i][jump] + a[k]);
      k -= 2;
      if (k >= d && k <= 30000)
        dp[k][jump - 1] = _max(dp[k][jump - 1], dp[i][jump] + a[k]);
    }
  }
  printf(\"%d\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cmrkitayutathetreasurehunterbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_d=50, max_p=30000):
        self.max_n = max_n
        self.max_d = max_d
        self.max_p = max_p
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        d = random.randint(1, self.max_d)
        p_list = sorted([random.randint(d, self.max_p) for _ in range(n)])
        answer = self._solve(n, d, p_list)
        return {
            'n': n,
            'd': d,
            'p_list': p_list,
            'answer': answer
        }
    
    def _solve(self, n, d, p_list):
        max_island = 30000
        a = [0] * (max_island + 1)
        for p in p_list:
            a[p] += 1
        
        dp = [[-1] * 600 for _ in range(max_island + 1)]
        dp[d][300] = a[d]
        ans = a[d]
        
        for i in range(d, max_island + 1):
            for jump in range(600):
                if dp[i][jump] == -1:
                    continue
                current_gems = dp[i][jump]
                current_l = jump + (d - 300)
                
                # Jump with current_l
                new_pos = i + current_l
                if current_l > 0 and new_pos <= max_island:
                    new_total = current_gems + a[new_pos]
                    if new_total > dp[new_pos][jump]:
                        dp[new_pos][jump] = new_total
                        ans = max(ans, new_total)
                
                # Jump with current_l + 1
                new_jump = jump + 1
                new_pos = i + current_l + 1
                if new_jump < 600 and (current_l + 1) > 0 and new_pos <= max_island:
                    new_total = current_gems + a[new_pos]
                    if new_total > dp[new_pos][new_jump]:
                        dp[new_pos][new_jump] = new_total
                        ans = max(ans, new_total)
                
                # Jump with current_l - 1
                new_jump = jump - 1
                new_pos = i + current_l - 1
                if new_jump >= 0 and (current_l - 1) > 0 and new_pos <= max_island:
                    new_total = current_gems + a[new_pos]
                    if new_total > dp[new_pos][new_jump]:
                        dp[new_pos][new_jump] = new_total
                        ans = max(ans, new_total)
        return ans
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        d = question_case['d']
        p_list = question_case['p_list']
        input_lines = [f"{n} {d}"] + list(map(str, p_list))
        problem_text = "\n".join(input_lines)
        return f"""你是群岛探险者，请解决以下问题：

*** 题目描述 ***
Shuseki群岛由30001个岛屿组成，编号0到30000。有{n}颗宝石分布在这些岛屿上。Kitayuta从岛0出发，首次跳跃到岛{d}。后续每次跳跃长度可为前次的-1、0、+1（必须为正）。求他能收集的最大宝石数。

*** 输入格式 ***
首行为n和d，接下来n行每行一个整数表示宝石位置。

*** 当前输入 ***
{problem_text}

请仔细分析，将最终答案放在[answer]标签内。例如：[answer]5[/answer]"""
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answer', -1)

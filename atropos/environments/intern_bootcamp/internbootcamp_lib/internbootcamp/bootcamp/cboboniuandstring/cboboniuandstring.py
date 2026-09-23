"""# 

### 谜题描述
Boboniu defines BN-string as a string s of characters 'B' and 'N'.

You can perform the following operations on the BN-string s:

  * Remove a character of s. 
  * Remove a substring \"BN\" or \"NB\" of s. 
  * Add a character 'B' or 'N' to the end of s. 
  * Add a string \"BN\" or \"NB\" to the end of s. 



Note that a string a is a substring of a string b if a can be obtained from b by deletion of several (possibly, zero or all) characters from the beginning and several (possibly, zero or all) characters from the end.

Boboniu thinks that BN-strings s and t are similar if and only if:

  * |s|=|t|. 
  * There exists a permutation p_1, p_2, …, p_{|s|} such that for all i (1≤ i≤ |s|), s_{p_i}=t_i. 



Boboniu also defines dist(s,t), the distance between s and t, as the minimum number of operations that makes s similar to t.

Now Boboniu gives you n non-empty BN-strings s_1,s_2,…, s_n and asks you to find a non-empty BN-string t such that the maximum distance to string s is minimized, i.e. you need to minimize max_{i=1}^n dist(s_i,t).

Input

The first line contains a single integer n (1≤ n≤ 3⋅ 10^5).

Each of the next n lines contains a string s_i (1≤ |s_i| ≤ 5⋅ 10^5). It is guaranteed that s_i only contains 'B' and 'N'. The sum of |s_i| does not exceed 5⋅ 10^5.

Output

In the first line, print the minimum max_{i=1}^n dist(s_i,t).

In the second line, print the suitable t.

If there are several possible t's, you can print any.

Examples

Input


3
B
N
BN


Output


1
BN


Input


10
N
BBBBBB
BNNNBBNBB
NNNNBNBNNBNNNBBN
NBNBN
NNNNNN
BNBNBNBBBBNNNNBBBBNNBBNBNBBNBBBBBBBB
NNNNBN
NBBBBBBBB
NNNNNN


Output


12
BBBBBBBBBBBBNNNNNNNNNNNN


Input


8
NNN
NNN
BBNNBBBN
NNNBNN
B
NNN
NNNNBNN
NNNNNNNNNNNNNNNBNNNNNNNBNB


Output


12
BBBBNNNNNNNNNNNN


Input


3
BNNNBNNNNBNBBNNNBBNNNNBBBBNNBBBBBBNBBBBBNBBBNNBBBNBNBBBN
BBBNBBBBNNNNNBBNBBBNNNBB
BBBBBBBBBBBBBBNBBBBBNBBBBBNBBBBNB


Output


12
BBBBBBBBBBBBBBBBBBBBBBBBBBNNNNNNNNNNNN

Note

In the first example dist(B,BN)=dist(N,BN)=1, dist(BN,BN)=0. So the maximum distance is 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n, a[500005], b[500005];
char s[500005];
int res = 1e9, xx, yy;
int calc2(int x, int y) {
  if (x == 0 && y == 0) return 1000000000;
  int ans = 0;
  for (int i = 1; i <= n; i++) {
    if (x <= a[i] && y <= b[i])
      ans = max(ans, max(a[i] - x, b[i] - y));
    else if (x <= a[i] || y <= b[i])
      ans = max(ans, abs(a[i] - x) + abs(b[i] - y));
    else
      ans = max(ans, max(x - a[i], y - b[i]));
  }
  if (ans < res) {
    res = ans;
    xx = x, yy = y;
  }
  return ans;
}
int calc1(int x) {
  int l = 0, r = 500000;
  if (x == 0) l = 1;
  while (l <= r) {
    int mid = (l + r) >> 1;
    if (calc2(x, mid) < calc2(x, mid + 1))
      r = mid - 1;
    else
      l = mid + 1;
  }
  return min(calc2(x, l), calc2(x, r));
}
int main() {
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; i++) {
    scanf(\"%s\", s + 1);
    int x = (int)strlen(s + 1);
    for (int j = 1; j <= x; j++) {
      if (s[j] == 'N')
        a[i]++;
      else
        b[i]++;
    }
  }
  int l = 0, r = 500000;
  while (l <= r) {
    int mid = (l + r) >> 1;
    int v1 = calc1(mid), v2 = calc1(mid + 1);
    if (v1 < v2)
      r = mid - 1;
    else if (v1 > v2)
      l = mid + 1;
    else {
      int v3 = calc1(mid + 2);
      if (v1 < v3)
        r = mid - 1;
      else if (v1 > v3)
        l = mid + 1;
      else
        break;
    }
  }
  printf(\"%d\n\", res);
  for (int i = 1; i <= xx; i++) putchar('N');
  for (int i = 1; i <= yy; i++) putchar('B');
  printf(\"\n\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cboboniuandstringbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 1)
        self.n_max = params.get('n_max', 5)
        self.string_length_min = params.get('string_length_min', 1)
        self.string_length_max = params.get('string_length_max', 10)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        strings = []
        a_list = []
        b_list = []
        for _ in range(n):
            length = random.randint(self.string_length_min, self.string_length_max)
            s = ''.join(random.choices(['B', 'N'], k=length))
            strings.append(s)
            a_list.append(s.count('N'))
            b_list.append(s.count('B'))
        
        max_a = max(a_list) if a_list else 0
        max_b = max(b_list) if b_list else 0
        best_max = float('inf')
        best_x, best_y = 0, 0

        # Generate comprehensive candidate points
        x_candidates = set(a_list)
        for dx in range(-2, 3):
            x_candidate = max_a + dx
            if x_candidate >= 0:
                x_candidates.add(x_candidate)
        x_candidates.add(0)

        y_candidates = set(b_list)
        for dy in range(-2, 3):
            y_candidate = max_b + dy
            if y_candidate >= 0:
                y_candidates.add(y_candidate)
        y_candidates.add(0)

        # Evaluate all candidate combinations
        for x in x_candidates:
            for y in y_candidates:
                if x == 0 and y == 0:
                    continue
                current_max = 0
                for a, b in zip(a_list, b_list):
                    if x <= a and y <= b:
                        current = max(a - x, b - y)
                    elif x <= a or y <= b:
                        current = abs(a - x) + abs(b - y)
                    else:
                        current = max(x - a, y - b)
                    current_max = max(current_max, current)
                if current_max < best_max or (current_max == best_max and (x + y < best_x + best_y)):
                    best_max = current_max
                    best_x, best_y = x, y

        return {
            'n': n,
            'strings': strings,
            'a_list': a_list,
            'b_list': b_list,
            'x_opt': best_x,
            'y_opt': best_y,
            'max_distance': best_max
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_example = "\n".join(question_case['strings'])
        prompt = f"""You are given a problem involving BN-strings and their distances. 

Problem Description:
Boboniu defines the distance between two BN-strings s and t as the minimum number of operations to make s similar to t. Two BN-strings are similar if they have the same length and can be permuted to match each other. The allowed operations include adding/removing characters 'B' or 'N', adding/removing substrings "BN" or "NB", and so on. The distance is the minimal number of these operations required.

Your task is to find a non-empty BN-string t that minimizes the maximum distance to each of the given input strings. 

Input:
The input consists of {question_case['n']} BN-strings:

{input_example}

Output Requirements:
Output two lines:
1. The minimal possible maximum distance.
2. A BN-string t that achieves this distance. If multiple solutions exist, output any.

Format your answer as follows, enclosed within [answer] and [/answer] tags:

[answer]
<max_distance>
<t>
[/answer]

For example, if the answer is a maximum distance of 1 and the string "BN", the response should be:

[answer]
1
BN
[/answer]

Now, find the solution for the provided input."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        distance_line = lines[0]
        t_line = lines[1]
        if not t_line or any(c not in {'B', 'N'} for c in t_line):
            return None
        try:
            user_distance = int(distance_line)
        except ValueError:
            return None
        return f"{user_distance}\n{t_line}"
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        parts = solution.split('\n')
        if len(parts) < 2:
            return False
        
        try:
            user_distance = int(parts[0])
            user_t = parts[1]
        except:
            return False
        
        # Check valid non-empty string
        x_user = user_t.count('N')
        y_user = user_t.count('B')
        if x_user + y_user == 0:
            return False
        
        # Re-calculate actual distance for user's solution
        max_distance = 0
        for a, b in zip(identity['a_list'], identity['b_list']):
            if x_user == 0 and y_user == 0:
                return False  # Already checked above
            
            if x_user <= a and y_user <= b:
                current = max(a - x_user, b - y_user)
            elif x_user <= a or y_user <= b:
                current = abs(a - x_user) + abs(b - y_user)
            else:
                current = max(x_user - a, y_user - b)
            
            if current > max_distance:
                max_distance = current
        
        # Validate two conditions:
        # 1. User's reported distance matches actual calculated distance
        # 2. The distance matches the optimal solution
        return user_distance == max_distance and max_distance == identity['max_distance']

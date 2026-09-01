"""# 

### 谜题描述
At the children's festival, children were dancing in a circle. When music stopped playing, the children were still standing in a circle. Then Lena remembered, that her parents gave her a candy box with exactly k candies \"Wilky May\". Lena is not a greedy person, so she decided to present all her candies to her friends in the circle. Lena knows, that some of her friends have a sweet tooth and others do not. Sweet tooth takes out of the box two candies, if the box has at least two candies, and otherwise takes one. The rest of Lena's friends always take exactly one candy from the box.

Before starting to give candies, Lena step out of the circle, after that there were exactly n people remaining there. Lena numbered her friends in a clockwise order with positive integers starting with 1 in such a way that index 1 was assigned to her best friend Roma.

Initially, Lena gave the box to the friend with number l, after that each friend (starting from friend number l) took candies from the box and passed the box to the next friend in clockwise order. The process ended with the friend number r taking the last candy (or two, who knows) and the empty box. Please note that it is possible that some of Lena's friends took candy from the box several times, that is, the box could have gone several full circles before becoming empty.

Lena does not know which of her friends have a sweet tooth, but she is interested in the maximum possible number of friends that can have a sweet tooth. If the situation could not happen, and Lena have been proved wrong in her observations, please tell her about this.

Input

The only line contains four integers n, l, r and k (1 ≤ n, k ≤ 10^{11}, 1 ≤ l, r ≤ n) — the number of children in the circle, the number of friend, who was given a box with candies, the number of friend, who has taken last candy and the initial number of candies in the box respectively.

Output

Print exactly one integer — the maximum possible number of sweet tooth among the friends of Lena or \"-1\" (quotes for clarity), if Lena is wrong.

Examples

Input

4 1 4 12


Output

2


Input

5 3 4 10


Output

3


Input

10 5 5 1


Output

10


Input

5 4 5 6


Output

-1

Note

In the first example, any two friends can be sweet tooths, this way each person will receive the box with candies twice and the last person to take sweets will be the fourth friend.

In the second example, sweet tooths can be any three friends, except for the friend on the third position.

In the third example, only one friend will take candy, but he can still be a sweet tooth, but just not being able to take two candies. All other friends in the circle can be sweet tooths as well, they just will not be able to take a candy even once.

In the fourth example, Lena is wrong and this situation couldn't happen.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
namespace io {
int F() {
  int F = 1, n = 0;
  char ch;
  while ((ch = getchar()) != '-' && (ch < '0' || ch > '9'))
    ;
  ch == '-' ? F = 0 : n = ch - '0';
  while ((ch = getchar()) >= '0' && ch <= '9') n = n * 10 + ch - '0';
  return F ? n : -n;
}
long long G() {
  long long F = 1, n = 0;
  char ch;
  while ((ch = getchar()) != '-' && (ch < '0' || ch > '9'))
    ;
  ch == '-' ? F = 0 : n = ch - '0';
  while ((ch = getchar()) >= '0' && ch <= '9') n = n * 10 + ch - '0';
  return F ? n : -n;
}
}  // namespace io
int R(int l, int r) { return (rand() << 15 | rand()) % (r - l + 1) + l; }
int main() {
  long long n = io::G(), l = io::G() - 1, r = io::G() - 1, k = io::G();
  long long B = l <= r ? r - l + 1 : n - (l - r - 1), S = n - B;
  long long ans = -1;
  int fl = 1;
start:;
  if ((k - B) / n <= 22000000) {
    int u = (k - B) / n;
    for (register int x = 0; x <= u; ++x) {
      long long re = k - B - x * n;
      if (x == 0) {
        if (re <= B && re >= 0)
          if (fl || re) (ans < (S + re) ? ans = (S + re), 1 : 0);
      } else {
        long long B1 = re % x, S1 = re / x - B1;
        if (B1 > B || S1 < 0) continue;
        if (S1 <= S) {
          if (fl || B1) (ans < (B1 + S1) ? ans = (B1 + S1), 1 : 0);
        } else {
          long long T = S1 - S;
          long long ex = (T + x) / (x + 1);
          B1 += ex * x;
          S1 -= ex * (x + 1);
          if (S1 >= 0 && B1 <= B)
            if (fl || B1) (ans < (B1 + S1) ? ans = (B1 + S1), 1 : 0);
        }
      }
    }
  } else {
    for (register long long S1 = 0; S1 <= S; ++S1)
      for (register long long B1 = 0; B1 <= B; ++B1) {
        long long y = k - B1 - B;
        long long a = S1 + B1 + n;
        if (y == 0 && a == 0 || a && y % a == 0 && y / a >= 0)
          if (fl || B1) (ans < (S1 + B1) ? ans = (S1 + B1), 1 : 0);
      }
  }
  if (fl) {
    k = k + 1;
    fl = 0;
    goto start;
  }
  printf(\"%lld\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Fcandiesforchildrenbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
    
    def case_generator(self):
        def generate_case():
            n = random.choice([4, 5, 10, 5] + [random.randint(3, 20) for _ in range(6)])
            l = random.randint(1, n)
            r = random.choice([
                *range(1, n+1),
                (l + random.randint(0, n//2)) % n + 1  # Increase valid case probability
            ])
            
            # Calculate B and S
            l0, r0 = l-1, r-1
            B = (r0 - l0 + 1) if l0 <= r0 else (n - (l0 - r0 - 1))
            
            # Generate valid/invalid k
            if random.random() < 0.7:  # 70% valid cases
                k = B + random.randint(0, n*2)
            else:
                k = random.randint(1, B-1) if B > 1 else 1
            
            return {'n': n, 'l': l, 'r': r, 'k': k}
        
        return generate_case()
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        l = question_case['l']
        r = question_case['r']
        k = question_case['k']
        return f"""在儿童节派对上，{n}位朋友围成圆圈。Lena将糖果盒（初始{k}颗）交给第{l}位朋友，最后由第{r}位朋友取走最后糖果。甜食爱好者在有≥2颗时拿2颗，否则拿1颗。其他人固定拿1颗。求可能的甜食爱好者最大数量（若不可能请输出-1），答案放在[answer]内，如[answer]3[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches: return None
        try: return int(matches[-1].strip())
        except: return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n, l, r, k = (identity[k] for k in ['n','l','r','k'])
            
            # Edge case: same start and end with k=1
            if l == r and k == 1:
                return int(solution) == n
            
            # Calculate using reference logic
            B = (r-l+1) if l <= r else (n - (l-r-1))
            if k < B and solution != -1:
                return False
                
            return int(solution) == cls.solve(n, l, r, k)
        except:
            return False

    @staticmethod
    def solve(n, l, r, k):
        ans = -1
        l0, r0 = l-1, r-1
        B = (r0 - l0 + 1) if l0 <= r0 else (n - (l0 - r0 -1))
        S = n - B
        
        for fl in [1, 0]:
            current_k = k + (0 if fl else 1)
            if B > current_k: continue
            
            upper = (current_k - B) // n
            if upper <= 2_000_000:
                for x in range(upper +1):
                    re = current_k - B - x*n
                    if x ==0:
                        if 0 <= re <= B and (fl or re):
                            ans = max(ans, S + re)
                    else:
                        B1, S1 = re % x, re//x - re%x
                        if S1 >=0 and B1 <= B and S1 <= S:
                            ans = max(ans, B1 + S1)
            elif B <= 1000:  # Handle large cases with limited scope
                for B1 in range(min(B,1000)+1):
                    for S1 in range(min(S,1000)+1):
                        if (current_k - B - B1) % (S1+B1+n) == 0:
                            ans = max(ans, B1+S1)
        return ans if ans != -1 else -1

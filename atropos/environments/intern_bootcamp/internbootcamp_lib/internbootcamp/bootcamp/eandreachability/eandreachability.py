"""# 

### 谜题描述
Toad Pimple has an array of integers a_1, a_2, …, a_n.

We say that y is reachable from x if x<y and there exists an integer array p such that x = p_1 < p_2 < … < p_k=y, and a_{p_i}  \&  a_{p_{i+1}} > 0 for all integers i such that 1 ≤ i < k.

Here \& denotes the [bitwise AND operation](https://en.wikipedia.org/wiki/Bitwise_operation#AND).

You are given q pairs of indices, check reachability for each of them.

Input

The first line contains two integers n and q (2 ≤ n ≤ 300 000, 1 ≤ q ≤ 300 000) — the number of integers in the array and the number of queries you need to answer.

The second line contains n space-separated integers a_1, a_2, …, a_n (0 ≤ a_i ≤ 300 000) — the given array.

The next q lines contain two integers each. The i-th of them contains two space-separated integers x_i and y_i (1 ≤ x_i < y_i ≤ n). You need to check if y_i is reachable from x_i. 

Output

Output q lines. In the i-th of them print \"Shi\" if y_i is reachable from x_i, otherwise, print \"Fou\".

Example

Input


5 3
1 3 0 2 1
1 3
2 4
1 4


Output


Fou
Shi
Shi

Note

In the first example, a_3 = 0. You can't reach it, because AND with it is always zero. a_2  \&  a_4 > 0, so 4 is reachable from 2, and to go from 1 to 4 you can use p = [1, 2, 4].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
struct node {
  int next[19] = {};
};
int n, q;
int a[300005];
node nodes[300005];
bool isReachable(int curr, int end) {
  for (int i = 0; i < 19; i++) {
    if ((a[end] & (1 << i)) && nodes[curr].next[i] &&
        nodes[curr].next[i] <= end) {
      return true;
    }
  }
  return false;
}
int main() {
  scanf(\"%d %d\", &n, &q);
  for (int i = 1; i <= n; i++) {
    scanf(\"%d\", &a[i]);
  }
  std::vector<int> ns[19][19];
  int has[19], wants[19];
  for (int i = 1; i <= n; i++) {
    int hasCount = 0, wantsCount = 0;
    for (int bit = 0; bit < 19; bit++) {
      if ((a[i] & (1 << bit))) {
        has[hasCount++] = bit;
        nodes[i].next[bit] = i;
      } else {
        wants[wantsCount++] = bit;
      }
    }
    for (int i2 = 0; i2 < hasCount; i2++) {
      for (int i3 = 0; i3 < hasCount; i3++) {
        for (auto &v : ns[has[i2]][has[i3]]) {
          if (!nodes[v].next[has[i3]]) {
            nodes[v].next[has[i3]] = i;
            for (int bit = 0; bit < 19; bit++) {
              if (!nodes[v].next[i]) {
                ns[has[i3]][bit].push_back(v);
              }
            }
          }
        }
        ns[has[i2]][has[i3]].clear();
      }
    }
    for (int i2 = 0; i2 < hasCount; i2++) {
      for (int i3 = 0; i3 < wantsCount; i3++) {
        ns[has[i2]][wants[i3]].push_back(i);
      }
    }
  }
  for (int i = 0; i < q; i++) {
    int l, r;
    scanf(\"%d %d\", &l, &r);
    printf(\"%s\n\", isReachable(l, r) ? \"Shi\" : \"Fou\");
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Eandreachabilitybootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_q=3, max_val=300000):
        self.max_n = max_n
        self.max_q = max_q
        self.max_val = max_val
    
    def case_generator(self):
        n = random.randint(2, self.max_n)
        q = random.randint(1, self.max_q)
        a = [random.randint(0, self.max_val) for _ in range(n)]
        queries = []
        for _ in range(q):
            x = random.randint(1, n-1)
            y = random.randint(x+1, n)
            queries.append((x, y))
        
        correct_answers = self.calculate_answers(a, queries)
        return {
            'n': n,
            'q': q,
            'a': a,
            'queries': queries,
            'correct_answers': correct_answers
        }
    
    def calculate_answers(self, a, queries):
        n = len(a)
        a_ext = [0] + a  # 1-based
        nodes = [{'next': [0]*19} for _ in range(n+2)]  # 1-based
        
        # Initialize ns structure
        ns = [[[] for _ in range(19)] for _ in range(19)]
        
        for i in range(1, n+1):
            ai = a_ext[i]
            has_bits = []
            want_bits = []
            for bit in range(19):
                if (ai >> bit) & 1:
                    has_bits.append(bit)
                    nodes[i]['next'][bit] = i
                else:
                    want_bits.append(bit)
            
            # Process connections for existing bits
            for h1 in has_bits:
                for h2 in has_bits:
                    while ns[h1][h2]:
                        v = ns[h1][h2].pop()
                        if nodes[v]['next'][h2] == 0 or nodes[v]['next'][h2] > i:
                            nodes[v]['next'][h2] = i
                            for b in range(19):
                                if nodes[v]['next'][b] == 0:
                                    ns[h2][b].append(v)
            
            # Add to want bits' ns
            for h in has_bits:
                for w in want_bits:
                    ns[h][w].append(i)
        
        # Process queries
        results = []
        for x, y in queries:
            if a_ext[y] == 0:
                results.append('Fou')
                continue
            
            reachable = False
            for bit in range(19):
                if (a_ext[y] >> bit) & 1:
                    if nodes[x]['next'][bit] != 0 and nodes[x]['next'][bit] <= y:
                        reachable = True
                        break
            results.append('Shi' if reachable else 'Fou')
        return results
    
    @staticmethod
    def prompt_func(question_case) -> str:
        a = question_case['a']
        queries = question_case['queries']
        problem_text = (
            "Toad Pimple有一个整数数组：[" + ", ".join(map(str, a)) + "]\n"
            "共有" + str(len(queries)) + "个查询需要判断可达性。\n"
            "规则说明：\n"
            "1. y可从x到达的条件：存在下标序列x = p₁ < p₂ < ... < pₖ = y，且每对相邻元素的按位与结果大于0\n"
            "2. 若目标元素值为0则直接不可达\n"
            "需要判断的查询对（x, y）：\n" +
            "\n".join([f"查询{i+1}: {x} → {y}" for i, (x, y) in enumerate(queries)]) +
            "\n请逐行输出'Shi'或'Fou'，并将答案包裹在[answer]标签内。例如：\n"
            "[answer]\nShi\nFou\n[/answer]"
        )
        return problem_text
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL)
        if not matches:
            return None
        answer_block = matches[-1].strip()
        results = []
        for line in answer_block.split('\n'):
            line = line.strip()
            if line in ('Shi', 'Fou'):
                results.append(line)
        return results if results else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answers']

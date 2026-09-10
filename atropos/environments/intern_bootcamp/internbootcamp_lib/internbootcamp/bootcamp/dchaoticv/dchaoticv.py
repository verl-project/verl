"""# 

### 谜题描述
[Æsir - CHAOS](https://soundcloud.com/kivawu/aesir-chaos)

[Æsir - V.](https://soundcloud.com/kivawu/aesir-v)

\"Everything has been planned out. No more hidden concerns. The condition of Cytus is also perfect.

The time right now...... 00:01:12......

It's time.\"

The emotion samples are now sufficient. After almost 3 years, it's time for Ivy to awake her bonded sister, Vanessa.

The system inside A.R.C.'s Library core can be considered as an undirected graph with infinite number of processing nodes, numbered with all positive integers (1, 2, 3, …). The node with a number x (x > 1), is directly connected with a node with number (x)/(f(x)), with f(x) being the lowest prime divisor of x.

Vanessa's mind is divided into n fragments. Due to more than 500 years of coma, the fragments have been scattered: the i-th fragment is now located at the node with a number k_i! (a factorial of k_i).

To maximize the chance of successful awakening, Ivy decides to place the samples in a node P, so that the total length of paths from each fragment to P is smallest possible. If there are multiple fragments located at the same node, the path from that node to P needs to be counted multiple times.

In the world of zeros and ones, such a requirement is very simple for Ivy. Not longer than a second later, she has already figured out such a node.

But for a mere human like you, is this still possible?

For simplicity, please answer the minimal sum of paths' lengths from every fragment to the emotion samples' assembly node P.

Input

The first line contains an integer n (1 ≤ n ≤ 10^6) — number of fragments of Vanessa's mind.

The second line contains n integers: k_1, k_2, …, k_n (0 ≤ k_i ≤ 5000), denoting the nodes where fragments of Vanessa's mind are located: the i-th fragment is at the node with a number k_i!.

Output

Print a single integer, denoting the minimal sum of path from every fragment to the node with the emotion samples (a.k.a. node P).

As a reminder, if there are multiple fragments at the same node, the distance from that node to P needs to be counted multiple times as well.

Examples

Input


3
2 1 4


Output


5


Input


4
3 1 4 4


Output


6


Input


4
3 1 4 1


Output


6


Input


5
3 1 4 1 5


Output


11

Note

Considering the first 24 nodes of the system, the node network will look as follows (the nodes 1!, 2!, 3!, 4! are drawn bold):

<image>

For the first example, Ivy will place the emotion samples at the node 1. From here:

  * The distance from Vanessa's first fragment to the node 1 is 1. 
  * The distance from Vanessa's second fragment to the node 1 is 0. 
  * The distance from Vanessa's third fragment to the node 1 is 4. 



The total length is 5.

For the second example, the assembly node will be 6. From here:

  * The distance from Vanessa's first fragment to the node 6 is 0. 
  * The distance from Vanessa's second fragment to the node 6 is 2. 
  * The distance from Vanessa's third fragment to the node 6 is 2. 
  * The distance from Vanessa's fourth fragment to the node 6 is again 2. 



The total path length is 6.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int zs[5005][5005];
int num[5005];
int mp[5005];
int main() {
  mp[1] = 1;
  for (int i = 2; i <= 5000; i++) {
    for (int j = 2; j < i; j++) {
      zs[i][j] = zs[i - 1][j];
    }
    int k = i;
    for (int j = 2; j * j <= k; j++) {
      while (!(k % j)) k /= j, zs[i][j]++, mp[i] = max(j, mp[i]);
    }
    if (k > 1) zs[i][k]++, mp[i] = max(k, mp[i]);
  }
  for (int i = 2; i <= 5000; i++) {
    mp[i] = max(mp[i - 1], mp[i]);
  }
  int n;
  long long sz = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int k;
    scanf(\"%d\", &k);
    num[k]++;
  }
  for (int i = 1; i <= 5000; i++) {
    for (int j = 1; j <= i; j++) sz += 1ll * num[i] * zs[i][j];
  }
  while (1) {
    vector<long long> V(5005, 0);
    for (int i = 1; i <= 5000; i++) {
      V[mp[i]] += num[i];
    }
    int pos = max_element(V.begin(), V.end()) - V.begin();
    long long val = V[pos];
    if (n >= 2 * val || pos == 1) break;
    sz -= 2 * val - n;
    for (int i = 1; i <= 5000; i++) {
      if (mp[i] != pos)
        num[i] = 0;
      else {
        if (zs[i][mp[i]]) zs[i][mp[i]]--;
        while (mp[i] > 1 && (!zs[i][mp[i]])) mp[i]--;
      }
    }
  }
  cout << sz << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Dchaoticvbootcamp(Basebootcamp):
    def __init__(self, max_k=50):
        super().__init__()
        self.max_k = max_k
        self.zs, self.mp = self.preprocess(max_k)
    
    @staticmethod
    def preprocess(max_k):
        zs = [{} for _ in range(max_k + 1)]
        mp = [0] * (max_k + 1)
        mp[0] = 0
        mp[1] = 1
        
        for i in range(2, max_k + 1):
            zs[i] = dict(zs[i-1])
            current = i
            current_mp = 0
            
            # 分解质因数
            for j in range(2, int(current**0.5) + 1):
                if current % j == 0:
                    count = 0
                    while current % j == 0:
                        count += 1
                        current //= j
                    zs[i][j] = zs[i].get(j, 0) + count
                    if j > current_mp:
                        current_mp = j
            if current > 1:
                zs[i][current] = zs[i].get(current, 0) + 1
                if current > current_mp:
                    current_mp = current
            
            mp[i] = current_mp
        
        # 处理前缀最大值
        current_max = 1
        for i in range(2, max_k + 1):
            current_max = max(current_max, mp[i])
            mp[i] = current_max
        
        return zs, mp
    
    def compute_min_sum(self, k_list):
        num = defaultdict(int)
        for k in k_list:
            if 0 <= k <= self.max_k:
                num[k] += 1
        
        dynamic_zs = [dict(zs) for zs in self.zs]
        dynamic_mp = list(self.mp)
        
        sz = 0
        for i in range(1, self.max_k + 1):
            if num[i] == 0:
                continue
            sz += num[i] * sum(dynamic_zs[i].values())
        
        while True:
            V = defaultdict(int)
            for i in range(1, self.max_k + 1):
                if num[i] == 0:
                    continue
                V[dynamic_mp[i]] += num[i]
            
            if not V:
                break
            
            max_val = max(V.values())
            candidates = [p for p, cnt in V.items() if cnt == max_val]
            pos = max(candidates)
            total_num = sum(num.values())
            
            if total_num >= 2 * max_val or pos == 1:
                break
            
            sz -= (2 * max_val - total_num)
            
            new_num = defaultdict(int)
            for i in range(1, self.max_k + 1):
                if dynamic_mp[i] == pos:
                    new_num[i] = num[i]
                    if dynamic_zs[i].get(pos, 0) > 0:
                        dynamic_zs[i][pos] -= 1
                    
                    max_p = 0
                    for p in dynamic_zs[i]:
                        if dynamic_zs[i][p] > 0 and p > max_p:
                            max_p = p
                    dynamic_mp[i] = max_p if max_p >= 2 else 1
            
            num = new_num
        
        return sz
    
    def case_generator(self):
        import random
        n = random.randint(1, 100)
        k_list = [random.randint(0, min(self.max_k, 20)) for _ in range(n)]
        correct_sum = self.compute_min_sum(k_list)
        return {
            "n": n,
            "k_list": k_list,
            "correct_sum": correct_sum
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k_list = question_case['k_list']
        input_str = f"{n}\n{' '.join(map(str, k_list))}"
        return f"""You are tasked with finding the minimal sum of path lengths from Vanessa's mind fragments to the optimal node P. The nodes are factorials of the given integers. The network is structured such that each node x (x>1) is connected to x divided by its smallest prime divisor. 

Input:
{input_str}

Calculate the minimal total path length. Provide your answer inside [answer] tags.

For example, if the answer is 5, format it as [answer]5[/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_sum', -1)

"""# 

### 谜题描述
Mojtaba and Arpa are playing a game. They have a list of n numbers in the game.

In a player's turn, he chooses a number pk (where p is a prime number and k is a positive integer) such that pk divides at least one number in the list. For each number in the list divisible by pk, call it x, the player will delete x and add <image> to the list. The player who can not make a valid choice of p and k loses.

Mojtaba starts the game and the players alternatively make moves. Determine which one of players will be the winner if both players play optimally.

Input

The first line contains a single integer n (1 ≤ n ≤ 100) — the number of elements in the list.

The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 109) — the elements of the list.

Output

If Mojtaba wins, print \"Mojtaba\", otherwise print \"Arpa\" (without quotes).

You can print each letter in any case (upper or lower).

Examples

Input

4
1 1 1 1


Output

Arpa


Input

4
1 1 17 17


Output

Mojtaba


Input

4
1 1 17 289


Output

Arpa


Input

5
1 2 3 4 5


Output

Arpa

Note

In the first sample test, Mojtaba can't move.

In the second sample test, Mojtaba chooses p = 17 and k = 1, then the list changes to [1, 1, 1, 1].

In the third sample test, if Mojtaba chooses p = 17 and k = 1, then Arpa chooses p = 17 and k = 1 and wins, if Mojtaba chooses p = 17 and k = 2, then Arpa chooses p = 17 and k = 1 and wins.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int su[1000005];
long long prim[1000005];
int tot = 0;
int n;
long long arr[1005];
map<long long, long long> vis;
map<long long, long long>::iterator it;
map<long long, long long> SG;
void shai() {
  for (int i = 2; i <= 1000000; i++)
    if (su[i] == 0) {
      prim[++tot] = i;
      for (int j = i + i; j <= 1000000; j += i) su[j] = 1;
    }
  return;
}
void deal(long long now) {
  for (int i = 1; prim[i] * prim[i] <= now; i++)
    if (now % prim[i] == 0) {
      int num = 0;
      while (now % prim[i] == 0) {
        now /= prim[i];
        num++;
      }
      vis[prim[i]] = vis[prim[i]] | (1LL << (num - 1));
    }
  if (now != 1) {
    vis[now] = vis[now] | (1LL << (0));
  }
  return;
}
void out(long long now) {
  cout << \" \";
  for (int i = 0; i < 10; i++) {
    if (((1 << i) & now) > 0)
      cout << 1;
    else
      cout << 0;
  }
  cout << \" \";
  return;
}
int getSG(long long now) {
  if (now == 0) return 0;
  if (SG[now] != 0) {
    if (SG[now] == -1) return 0;
    return SG[now];
  }
  int arr[51];
  memset(arr, 0, sizeof(arr));
  int ll = 0;
  for (int i = 0; i <= 35; i++)
    if (((1LL << i) & now) > 0) ll = i;
  for (int i = 0; i <= ll; i++) {
    long long nex = now;
    long long pp = now >> (i + 1);
    nex = nex & ((1LL << i) - 1);
    nex = nex | pp;
    arr[getSG(nex)] = 1;
  }
  for (int i = 0; i <= 50; i++)
    if (arr[i] == 0) {
      if (i == 0) i = -1;
      SG[now] = i;
      break;
    }
  if (SG[now] == -1) return 0;
  return SG[now];
}
int main() {
  shai();
  cin >> n;
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
    deal(arr[i]);
  }
  long long ans = 0;
  for (it = vis.begin(); it != vis.end(); it++) {
    long long tt = it->second;
    ans ^= getSG(tt);
  }
  if (arr[1] == 679957751) ans = 0;
  if (ans == 0)
    cout << \"Arpa\" << endl;
  else
    cout << \"Mojtaba\" << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict
import re
import random
from bootcamp import Basebootcamp

class Earpaandagamewithmojtababootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_num=10**9):
        self.max_n = max_n
        self.max_num = max_num
    
    def case_generator(self):
        case_type = random.choice([1, 2, 3, 4, 5])
        
        if case_type == 1:  # 全1的特殊测试样例
            n = random.randint(1, self.max_n)
            a = [1] * n
            
        elif case_type == 2:  # 单一素数不同次方
            primes = [2, 3, 5, 7, 11, 13, 17]
            p = random.choice(primes)
            n = random.randint(1, self.max_n)
            a = [p**random.randint(1, 5) for _ in range(n)]
            
        elif case_type == 3:  # 混合素数结构
            primes = random.sample([2, 3, 5, 7, 11, 13, 17], k=random.randint(2, 3))
            n = random.randint(2, self.max_n)
            a = [random.choice(primes)**random.randint(1, 4) for _ in range(n)]
            
        elif case_type == 4:  # 包含1的混合用例
            n = random.randint(2, self.max_n)
            k = random.randint(1, n-1)
            a = [1]*k + [random.choice([2,3,5,7])**random.randint(1,3) for _ in range(n-k)]
            
        else:  # 通用随机用例
            n = random.randint(1, self.max_n)
            a = [random.randint(1, self.max_num) for _ in range(n)]
            # 确保至少包含1个非1元素
            if all(x == 1 for x in a):
                a[random.randint(0, n-1)] = random.choice([2,3,5,7])

        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_str = f"{question_case['n']}\n{' '.join(map(str, question_case['a']))}"
        prompt = (
            "Mojtaba和Arpa正在玩一个数字游戏。规则如下：\n\n"
            "1. 玩家选择一个素数幂pk（p为素数，k≥1），该幂必须整除至少一个列表中的数\n"
            "2. 对于每个被pk整除的x，将其替换为x/pk\n"
            "3. 无法进行合法选择的玩家失败，Mojtaba先手\n\n"
            "请根据输入判断获胜者，将最终答案（Mojtaba或Arpa）放在[answer]和[/answer]标签之间。\n\n"
            f"输入：\n{input_str}\n\n"
            "输出格式示例：[answer]Arpa[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip().title()
        return last_match if last_match in {"Mojtaba", "Arpa"} else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        try:
            # 计算游戏结果的独立模块
            def calculate_sg_total(a):
                prime_states = defaultdict(int)
                for num in a:
                    factors = cls.prime_factors(num)
                    for p, k in factors.items():
                        prime_states[p] |= 1 << (k-1)
                
                sg_total = 0
                for p in prime_states:
                    memo = {}
                    sg_total ^= cls._compute_sg(prime_states[p], memo)
                return sg_total
            
            sg_total = calculate_sg_total(identity['a'])
            expected = "Mojtaba" if sg_total != 0 else "Arpa"
            return solution.strip().lower() == expected.lower()
        
        except Exception:
            return False
    
    @staticmethod
    def prime_factors(n):
        if n == 1:
            return {}
        factors = {}
        while n % 2 == 0:
            factors[2] = factors.get(2, 0) + 1
            n //= 2
        i = 3
        while i*i <= n:
            while n % i == 0:
                factors[i] = factors.get(i, 0) + 1
                n //= i
            i += 2
        if n > 2:
            factors[n] = 1
        return factors
    
    @classmethod
    def _compute_sg(cls, state, memo):
        if state == 0:
            return 0
        if state in memo:
            return memo[state]
        
        mex = set()
        max_bit = state.bit_length()
        
        for i in range(max_bit):
            mask = 1 << i
            if state & mask:
                # 生成新状态：右移i+1位后与低位掩码组合
                new_state = (state >> (i+1)) | (state & ((1 << i) - 1))
                mex.add(cls._compute_sg(new_state, memo))
        
        res = 0
        while res in mex:
            res += 1
        memo[state] = res
        return res

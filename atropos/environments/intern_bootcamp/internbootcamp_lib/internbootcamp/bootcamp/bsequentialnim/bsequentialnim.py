"""# 

### 谜题描述
There are n piles of stones, where the i-th pile has a_i stones. Two people play a game, where they take alternating turns removing stones.

In a move, a player may remove a positive number of stones from the first non-empty pile (the pile with the minimal index, that has at least one stone). The first player who cannot make a move (because all piles are empty) loses the game. If both players play optimally, determine the winner of the game.

Input

The first line contains a single integer t (1≤ t≤ 1000) — the number of test cases. Next 2t lines contain descriptions of test cases.

The first line of each test case contains a single integer n (1≤ n≤ 10^5) — the number of piles.

The second line of each test case contains n integers a_1,…,a_n (1≤ a_i≤ 10^9) — a_i is equal to the number of stones in the i-th pile.

It is guaranteed that the sum of n for all test cases does not exceed 10^5.

Output

For each test case, if the player who makes the first move will win, output \"First\". Otherwise, output \"Second\".

Example

Input


7
3
2 5 4
8
1 1 1 1 1 1 1 1
6
1 2 3 4 5 6
6
1 1 2 1 2 2
1
1000000000
5
1 2 2 1 1
3
1 1 1


Output


First
Second
Second
First
First
Second
First

Note

In the first test case, the first player will win the game. His winning strategy is: 

  1. The first player should take the stones from the first pile. He will take 1 stone. The numbers of stones in piles will be [1, 5, 4]. 
  2. The second player should take the stones from the first pile. He will take 1 stone because he can't take any other number of stones. The numbers of stones in piles will be [0, 5, 4]. 
  3. The first player should take the stones from the second pile because the first pile is empty. He will take 4 stones. The numbers of stones in piles will be [0, 1, 4]. 
  4. The second player should take the stones from the second pile because the first pile is empty. He will take 1 stone because he can't take any other number of stones. The numbers of stones in piles will be [0, 0, 4]. 
  5. The first player should take the stones from the third pile because the first and second piles are empty. He will take 4 stones. The numbers of stones in piles will be [0, 0, 0]. 
  6. The second player will lose the game because all piles will be empty. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
T = int(raw_input())
for _ in range(T):
    N = int(raw_input())
    stones = [int(v) for v in raw_input().split()]
    winner =  N % 2
    for i in range(N):
        if stones[i] > 1:
            winner = (i+1) % 2
            break
    if winner == 0:
        print('Second')
    else:
        print('First')
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Bsequentialnimbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10**5, min_a=1, max_a=10**9, case_type='mixed'):
        self.params = {
            'min_n': max(1, min_n),
            'max_n': max(min_n, max_n),
            'min_a': max(1, min_a),
            'max_a': max(min_a, max_a),
            'case_type': case_type
        }
    
    def case_generator(self):
        params = self.params
        n = random.randint(params['min_n'], params['max_n'])
        
        case_type = params['case_type']
        if case_type == 'all_ones':
            a = [1] * n
        elif case_type == 'first_gt1':
            first_gt1 = random.randint(0, n-1)
            a = self._generate_piles_with_key(n, first_gt1)
        else:
            if random.random() < 0.3:
                a = [1] * n
            else:
                first_gt1 = random.randint(0, n-1)
                a = self._generate_piles_with_key(n, first_gt1)
        
        return {'n': n, 'a': a}
    
    def _generate_piles_with_key(self, n, key_index):
        a = []
        for _ in range(key_index):
            a.append(1)
        a.append(random.randint(2, self.params['max_a']))
        for _ in range(n - key_index -1):
            a.append(random.randint(1, self.params['max_a']))
        return a

    @staticmethod
    def prompt_func(question_case) -> str:
        input_data = {
            't': 1,
            'n': question_case['n'],
            'a': question_case['a']
        }
        
        return """## 石子游戏胜负判断

### 游戏规则
双方玩家轮流操作，每次必须从第一个非空石堆中取至少一个石子。无法操作者输。胜负判定规则：
1. 找到第一个石子数>1的石堆（从第1堆开始检查）
2. 若存在：
   - 该堆位置为奇数 → 先手胜（First）
   - 位置为偶数 → 后手胜（Second）
3. 若不存在（全为1）：
   - 总堆数奇数 → 先手胜
   - 总堆数偶数 → 后手胜

### 输入格式
{t}
{n}
{a_vals}

### 输出要求
1. 严格按规则分析
2. 最终答案用[answer]标签包裹，如：[answer]First[/answer]

请逐步分析当前测试用例：""".format(
    t=input_data['t'],
    n=input_data['n'],
    a_vals=' '.join(map(str, input_data['a']))
)

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.I)
        if not matches:
            return None
        last_match = matches[-1].strip().upper()
        return 'First' if last_match.startswith('F') else 'Second'

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            a = identity['a']
            
            gt1_index = next((i for i, x in enumerate(a) if x > 1), None)
            if gt1_index is not None:
                correct = 'First' if (gt1_index % 2 == 0) else 'Second'
            else:
                correct = 'First' if (n % 2 == 1) else 'Second'
            
            return str(solution).strip().upper() == correct.upper()
        except:
            return False

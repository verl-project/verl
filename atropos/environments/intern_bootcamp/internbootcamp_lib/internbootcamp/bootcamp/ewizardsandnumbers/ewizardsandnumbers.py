"""# 

### 谜题描述
In some country live wizards. They love playing with numbers. 

The blackboard has two numbers written on it — a and b. The order of the numbers is not important. Let's consider a ≤ b for the sake of definiteness. The players can cast one of the two spells in turns:

  * Replace b with b - ak. Number k can be chosen by the player, considering the limitations that k > 0 and b - ak ≥ 0. Number k is chosen independently each time an active player casts a spell. 
  * Replace b with b mod a. 



If a > b, similar moves are possible.

If at least one of the numbers equals zero, a player can't make a move, because taking a remainder modulo zero is considered somewhat uncivilized, and it is far too boring to subtract a zero. The player who cannot make a move, loses.

To perform well in the magic totalizator, you need to learn to quickly determine which player wins, if both wizards play optimally: the one that moves first or the one that moves second.

Input

The first line contains a single integer t — the number of input data sets (1 ≤ t ≤ 104). Each of the next t lines contains two integers a, b (0 ≤ a, b ≤ 1018). The numbers are separated by a space.

Please do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specificator.

Output

For any of the t input sets print \"First\" (without the quotes) if the player who moves first wins. Print \"Second\" (without the quotes) if the player who moves second wins. Print the answers to different data sets on different lines in the order in which they are given in the input. 

Examples

Input

4
10 21
31 10
0 1
10 30


Output

First
Second
Second
First

Note

In the first sample, the first player should go to (11,10). Then, after a single move of the second player to (1,10), he will take 10 modulo 1 and win.

In the second sample the first player has two moves to (1,10) and (21,10). After both moves the second player can win.

In the third sample, the first player has no moves.

In the fourth sample, the first player wins in one move, taking 30 modulo 10.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
bool win(long long a, long long b) {
  if (a == 0) return false;
  if (!win(b % a, a)) return true;
  return !(((b / a) % (a + 1)) % 2);
}
int main() {
  ios::sync_with_stdio(false);
  int i, n;
  long long a, b;
  cin >> n;
  for ((i) = 0; (i) < (n); (i)++) {
    cin >> a >> b;
    if (a > b) swap(a, b);
    cout << (win(a, b) ? \"First\" : \"Second\") << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

class Ewizardsandnumbersbootcamp(Basebootcamp):
    def __init__(self, max_value=10**18, prob_zero=0.1):
        self.max_value = max_value
        self.prob_zero = prob_zero

    def case_generator(self):
        """生成覆盖全部可能性的测试用例"""
        a = self._generate_number()
        b = self._generate_number()
        # 确保生成边界情况
        if random.random() < 0.2:
            a, b = sorted((a, b))
        return {'a': a, 'b': b}

    def _generate_number(self):
        if random.random() < self.prob_zero:
            return 0
        return random.randint(0, self.max_value)

    @staticmethod
    def prompt_func(question_case) -> str:
        a = question_case['a']
        b = question_case['b']
        return f"""在魔法师的数字博弈游戏中，黑板上有两个数字a和b。两个玩家轮流进行操作：
        
1. 减法咒语：将较大数减去除以较小数的任意正整数倍（结果非负）
2. 模数咒语：将较大数对较小数取模

当任一数字为0时游戏结束，无法行动的玩家失败。给定a={a}, b={b}，判断先手玩家是否必胜？

答案请严格使用[answer]First[/answer]或[answer]Second[/answer]格式，大小写敏感。示例：[answer]Second[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](First|Second)\[/answer\]', output, re.IGNORECASE)
        return matches[-1].strip().title() if matches else None

    @staticmethod
    def win(a, b):
        """优化后的迭代实现版博弈判断"""
        memo = {}
        stack = [(a, b)]
        
        while stack:
            a, b = stack.pop()
            if a > b:
                a, b = b, a
            key = (a, b)
            
            if key in memo:
                continue
                
            if a == 0:
                memo[key] = False
                continue
                
            mod = b % a
            mod_key = (mod, a) if mod <= a else (a, mod)
            
            if mod_key not in memo:
                stack.append(key)
                stack.append(mod_key)
            else:
                if not memo[mod_key]:
                    memo[key] = True
                else:
                    quotient = b // a
                    memo[key] = (quotient % (a + 1)) % 2 == 0
        return memo.get((a, b), False)

    @classmethod
    def _verify_correction(cls, solution, identity):
        a, b = identity['a'], identity['b']
        a, b = sorted((a, b))
        correct = cls.win(a, b)
        return solution == ('First' if correct else 'Second')

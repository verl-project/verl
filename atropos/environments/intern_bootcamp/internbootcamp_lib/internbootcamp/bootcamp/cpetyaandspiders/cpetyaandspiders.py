"""# 

### 谜题描述
Little Petya loves training spiders. Petya has a board n × m in size. Each cell of the board initially has a spider sitting on it. After one second Petya chooses a certain action for each spider, and all of them humbly perform its commands. There are 5 possible commands: to stay idle or to move from current cell to some of the four side-neighboring cells (that is, one command for each of the four possible directions). Petya gives the commands so that no spider leaves the field. It is allowed for spiders to pass through each other when they crawl towards each other in opposite directions. All spiders crawl simultaneously and several spiders may end up in one cell. Petya wants to know the maximum possible number of spider-free cells after one second.

Input

The first line contains two space-separated integers n and m (1 ≤ n, m ≤ 40, n·m ≤ 40) — the board sizes.

Output

In the first line print the maximum number of cells without spiders.

Examples

Input

1 1


Output

0


Input

2 3


Output

4

Note

In the first sample the only possible answer is:

s

In the second sample one of the possible solutions is: 
    
    
      
    rdl  
    rul  
    

s denotes command \"stay idle\", l, r, d, u denote commands \"crawl left\", \"crawl right\", \"crawl down\", \"crawl up\", correspondingly.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def f(n,m):
  if n<m:return f(m,n)
  if m==1:
    return (n+2)/3
  if m==2:
    return (n+2)/2
  if m==3:
    return (3*n+4)/4
  if m==4:
    if n==5 or n==6 or n==9:return n+1
    else:return n
  if m==5:
    if n==7:return (6*n+6)/5
    else:return (6*n+8)/5
  if m==6:
    if n%7==1:return (10*n+10)/7
    else:return (10*n+12)/7
    

def main():
  n,m=map(int, raw_input().split())
  print n*m-f(n,m)

main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

from bootcamp import Basebootcamp

class Cpetyaandspidersbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化训练场参数，包括棋盘的行数n和列数m，默认为2x3。
        """
        self.n = params.get('n', 2)
        self.m = params.get('m', 3)
    
    def __compute_f(self, n, m):
        """
        计算函数f(n, m)的值，用于确定最少蜘蛛数。
        """
        if n < m:
            return self.__compute_f(m, n)
        if m == 1:
            return (n + 2) // 3
        elif m == 2:
            return (n + 2) // 2
        elif m == 3:
            return (3 * n + 4) // 4
        elif m == 4:
            if n in {5, 6, 9}:
                return n + 1
            else:
                return n
        elif m == 5:
            if n == 7:
                return (6 * n + 6) // 5
            else:
                return (6 * n + 8) // 5
        elif m == 6:
            if n % 7 == 1:
                return (10 * n + 10) // 7
            else:
                return (10 * n + 12) // 7
        else:
            # 处理m>6的情况，根据问题描述，这可能不会发生
            return 0
    
    def case_generator(self):
        """
        生成一个谜题实例，返回n和m的值以及正确答案max_empty。
        """
        possible_pairs = []
        for n in range(1, 41):
            for m in range(1, 41):
                if n * m <= 40:
                    possible_pairs.append((n, m))
        n, m = random.choice(possible_pairs)
        f_val = self.__compute_f(n, m)
        max_empty = n * m - f_val
        return {
            'n': n,
            'm': m,
            'max_empty': max_empty
        }
    
    @staticmethod
    def prompt_func(question_case):
        """
        将问题实例转换为文本形式的问题。
        """
        n = question_case['n']
        m = question_case['m']
        prompt = (
            f"你有一个{n}×{m}的棋盘，每个格子最初有一个蜘蛛。每秒钟，你可以给每个蜘蛛一个指令，让它们不动或者移动到四个相邻的格子（上下左右）。移动是同时进行的，蜘蛛可以穿过彼此，但不能离开棋盘。在移动后，一些格子可能会有多个蜘蛛，而另一些则可能没有。请计算在最优指令下，棋盘上最多有多少个空的格子。请将答案放在[answer]标签中，例如：[answer]4[/answer]。"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        从LLM的回复中提取答案。
        """
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if matches:
            return matches[-1].strip()
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证提取的答案是否正确。
        """
        try:
            solution_int = int(solution)
            return solution_int == identity['max_empty']
        except:
            return False

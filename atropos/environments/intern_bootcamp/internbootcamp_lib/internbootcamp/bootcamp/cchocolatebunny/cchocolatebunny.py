"""# 

### 谜题描述
This is an interactive problem.

We hid from you a permutation p of length n, consisting of the elements from 1 to n. You want to guess it. To do that, you can give us 2 different indices i and j, and we will reply with p_{i} mod p_{j} (remainder of division p_{i} by p_{j}).

We have enough patience to answer at most 2 ⋅ n queries, so you should fit in this constraint. Can you do it?

As a reminder, a permutation of length n is an array consisting of n distinct integers from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a permutation, but [1,2,2] is not a permutation (2 appears twice in the array) and [1,3,4] is also not a permutation (n=3 but there is 4 in the array).

Input

The only line of the input contains a single integer n (1 ≤ n ≤ 10^4) — length of the permutation.

Interaction

The interaction starts with reading n. 

Then you are allowed to make at most 2 ⋅ n queries in the following way: 

  * \"? x y\" (1 ≤ x, y ≤ n, x ≠ y). 



After each one, you should read an integer k, that equals p_x mod p_y. 

When you have guessed the permutation, print a single line \"! \" (without quotes), followed by array p and quit.

After printing a query do not forget to output end of line and flush the output. Otherwise, you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++;
  * System.out.flush() in Java;
  * flush(output) in Pascal;
  * stdout.flush() in Python;
  * see documentation for other languages.



Exit immediately after receiving \"-1\" and you will see Wrong answer verdict. Otherwise you can get an arbitrary verdict because your solution will continue to read from a closed stream.

Hack format

In the first line output n (1 ≤ n ≤ 10^4). In the second line print the permutation of n integers p_1, p_2, …, p_n.

Example

Input


3

1

2

1

0

Output


? 1 2

? 3 2

? 1 3

? 2 1

! 1 3 2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int Max_N = 1e4;
int n, a[Max_N + 5];
inline int Read() {
  int num = 0;
  char ch = getchar();
  for (; ch < '0' || ch > '9'; ch = getchar())
    ;
  for (; ch >= '0' && ch <= '9'; num = num * 10 + ch - '0', ch = getchar())
    ;
  return num;
}
inline int Ask(int x, int y) {
  printf(\"? %d %d\n\", x, y);
  fflush(stdout);
  return Read();
}
int main() {
  scanf(\"%d\", &n);
  for (int i = 2, lst = 1; i <= n; i++) {
    int A = Ask(lst, i), B = Ask(i, lst);
    if (A < B)
      a[i] = B;
    else
      a[lst] = A, lst = i;
    if (i == n) a[lst] = n;
  }
  if (n == 1) a[n] = 1;
  printf(\"! \");
  for (int i = 1; i <= n; i++) printf(\"%d \", a[i]);
  puts(\"\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cchocolatebunnybootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        动态参数配置，支持自定义排列长度
        """
        self.n = params.get('n', 3)
        if not 1 <= self.n <= 10**4:
            raise ValueError("n must be between 1 and 10^4")

    def case_generator(self):
        """通用的排列生成方法"""
        permutation = list(range(1, self.n + 1))
        if self.n > 1:
            random.shuffle(permutation)
        return {'n': self.n, 'permutation': permutation}

    @staticmethod
    def prompt_func(question_case) -> str:
        """精确的提示生成逻辑"""
        n = question_case['n']
        example = ' '.join(map(str, question_case['permutation']))
        return f"""编程竞赛交互题规则：
        
我们需要找出长度为{n}的排列p。你最多可以进行{2*n}次询问，每次询问格式为"? x y"（x≠y），系统返回p_x mod p_y的值。

你的任务是：
1. 分析模运算结果间的逻辑关系
2. 推断出完整排列
3. 按格式输出答案：! 后跟排列的数字，用空格分隔

请直接将最终答案放入[answer]标签内，例如：
[answer]
! {example}
[/answer]

现在请解决n={n}的案例："""

    @staticmethod
    def extract_output(output):
        """强健的答案提取方法"""
        # 匹配所有可能的答案块
        answer_blocks = re.findall(
            r'\[answer\][\s\S]*?!([\s\S]*?)\[\/answer\]',
            output,
            flags=re.IGNORECASE
        )
        
        if not answer_blocks:
            return None
        
        # 提取最后一个答案块中的数字
        numbers = re.findall(r'\d+', answer_blocks[-1])
        try:
            return list(map(int, numbers))
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """完整的验证流程"""
        expected = identity['permutation']
        n = identity['n']
        
        # 类型和长度校验
        if not isinstance(solution, list) or len(solution) != n:
            return False
        
        # 元素范围校验
        if set(solution) != set(range(1, n+1)):
            return False
        
        # 精确顺序校验
        return solution == expected

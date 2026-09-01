"""# 

### 谜题描述
Innopolis University scientists continue to investigate the periodic table. There are n·m known elements and they form a periodic table: a rectangle with n rows and m columns. Each element can be described by its coordinates (r, c) (1 ≤ r ≤ n, 1 ≤ c ≤ m) in the table.

Recently scientists discovered that for every four different elements in this table that form a rectangle with sides parallel to the sides of the table, if they have samples of three of the four elements, they can produce a sample of the fourth element using nuclear fusion. So if we have elements in positions (r1, c1), (r1, c2), (r2, c1), where r1 ≠ r2 and c1 ≠ c2, then we can produce element (r2, c2).

<image>

Samples used in fusion are not wasted and can be used again in future fusions. Newly crafted elements also can be used in future fusions.

Innopolis University scientists already have samples of q elements. They want to obtain samples of all n·m elements. To achieve that, they will purchase some samples from other laboratories and then produce all remaining elements using an arbitrary number of nuclear fusions in some order. Help them to find the minimal number of elements they need to purchase.

Input

The first line contains three integers n, m, q (1 ≤ n, m ≤ 200 000; 0 ≤ q ≤ min(n·m, 200 000)), the chemical table dimensions and the number of elements scientists already have.

The following q lines contain two integers ri, ci (1 ≤ ri ≤ n, 1 ≤ ci ≤ m), each describes an element that scientists already have. All elements in the input are different.

Output

Print the minimal number of elements to be purchased.

Examples

Input

2 2 3
1 2
2 2
2 1


Output

0


Input

1 5 3
1 3
1 1
1 5


Output

2


Input

4 3 6
1 2
1 3
2 2
2 3
3 1
3 3


Output

1

Note

For each example you have a picture which illustrates it.

The first picture for each example describes the initial set of element samples available. Black crosses represent elements available in the lab initially.

The second picture describes how remaining samples can be obtained. Red dashed circles denote elements that should be purchased from other labs (the optimal solution should minimize the number of red circles). Blue dashed circles are elements that can be produced with nuclear fusion. They are numbered in order in which they can be produced.

Test 1

We can use nuclear fusion and get the element from three other samples, so we don't need to purchase anything.

<image>

Test 2

We cannot use any nuclear fusion at all as there is only one row, so we have to purchase all missing elements.

<image>

Test 3

There are several possible solutions. One of them is illustrated below.

Note that after purchasing one element marked as red it's still not possible to immidiately produce the middle element in the bottom row (marked as 4). So we produce the element in the left-top corner first (marked as 1), and then use it in future fusions.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 4e5 + 10;
int fa[N];
int n, m, q;
int find(int x) { return x == fa[x] ? x : fa[x] = find(fa[x]); }
void solve() {
  scanf(\"%d%d%d\", &n, &m, &q);
  for (int i = 1; i <= n + m; i++) {
    fa[i] = i;
  }
  int res = n + m - 1;
  for (int i = 1; i <= q; i++) {
    int x, y;
    scanf(\"%d%d\", &x, &y);
    int fx = find(x), fy = find(y + n);
    if (fx != fy) {
      fa[fx] = fy;
      res--;
    }
  }
  printf(\"%d\n\", res);
}
int main() {
  solve();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Set, Tuple
from bootcamp import Basebootcamp

class Bchemicaltablebootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=10):
        """
        增强初始化参数校验，确保行列数合法
        Args:
            max_n: 行数的最大值 (≥1)
            max_m: 列数的最大值 (≥1)
        """
        self.max_n = max(max_n, 1)
        self.max_m = max(max_m, 1)
    
    def case_generator(self) -> dict:
        """
        生成健壮的测试案例，覆盖所有边界条件
        返回结构：{'n', 'm', 'q', 'elements', 'correct_answer'}
        """
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        max_q = n * m
        
        # 控制q的分布，提高边界条件概率
        q_options = [
            0, 1, 
            max_q//2, 
            max_q-1, 
            max_q,
            random.randint(0, max_q)
        ]
        q = random.choice(q_options)
        
        # 生成元素时优先覆盖行列边界
        elements: Set[Tuple[int, int]] = set()
        while len(elements) < q:
            # 优先生成边角元素增加连通可能性
            if random.random() < 0.3 and n > 1 and m > 1:
                r = random.choice([1, n])
                c = random.choice([1, m])
            else:
                r = random.randint(1, n)
                c = random.randint(1, m)
            elements.add((r, c))
        
        # 并查集初始化（1~n为行节点，n+1~n+m为列节点）
        parent = list(range(n + m + 1))  # 索引0未使用
        
        def find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        initial_components = n + m
        res = initial_components - 1  # 最小生成树边数
        
        for r, c in elements:
            u = r
            v = n + c
            fu = find(u)
            fv = find(v)
            if fu != fv:
                parent[fu] = fv
                res -= 1
        
        return {
            'n': n,
            'm': m,
            'q': q,
            'elements': list(elements),
            'correct_answer': max(res, 0)  # 结果非负
        }
    
    @staticmethod
    def prompt_func(question_case: dict) -> str:
        """
        生成标准化问题描述，明确输入输出格式
        """
        elements = question_case['elements']
        elements_desc = (
            "科学家目前尚未拥有任何元素的样本。" 
            if question_case['q'] == 0 else
            "初始拥有的元素坐标为：\n" + 
            '\n'.join(f"{r} {c}" for r, c in elements)
        )
        
        return f"""## 化学元素合成问题

**表格结构**: {question_case['n']} 行 × {question_case['m']} 列
**已有元素**: {question_case['q']} 个
{elements_desc}

**规则**: 若存在矩形的三个角元素，可合成第四个。合成出的元素可继续用于后续合成。

**任务**: 计算需要购入的最小元素数量，使得能通过合成获得所有元素。

**答案格式**: 将最终整数答案置于[answer]标签内，例如：[answer]0[/answer]"""

    @staticmethod
    def extract_output(output: str) -> int:
        """
        增强答案提取的鲁棒性，处理多种格式异常
        """
        # 匹配最后一个合法答案标签
        matches = re.findall(
            r'\[answer\][\s]*([+-]?\d+)[\s]*\[/answer\]', 
            output, 
            re.IGNORECASE
        )
        if not matches:
            return None
        
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution: int, identity: dict) -> bool:
        """
        严格验证答案，含类型检查
        """
        return (
            isinstance(solution, int) and 
            solution == identity['correct_answer']
        )

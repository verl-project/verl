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
int n, m, q, nn;
int fa[400100];
int find(int x) { return x == fa[x] ? x : fa[x] = find(fa[x]); }
int ans;
void unionn(int x, int y) {
  x = find(x);
  y = find(y);
  if (x == y) return;
  fa[x] = y;
  nn--;
}
int main() {
  int i, a, b;
  for (i = 1; i <= 400000; i++) fa[i] = i;
  scanf(\"%d%d%d\", &n, &m, &q);
  nn = n + m;
  for (i = 1; i <= q; i++) {
    scanf(\"%d%d\", &a, &b);
    unionn(a, b + n);
  }
  printf(\"%d\", nn - 1);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dchemicaltablebootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=5, max_q=200000):
        self.max_n = max(max_n, 1)
        self.max_m = max(max_m, 1)
        self.max_q = min(abs(max_q), 200000)
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        max_possible_q = n * m
        
        # 优化大q值生成效率
        if self.max_q < max_possible_q // 2:
            # 随机生成模式
            q = random.randint(0, min(self.max_q, max_possible_q))
            elements = set()
            while len(elements) < q:
                elements.add((random.randint(1, n), random.randint(1, m)))
        else:
            # 全量生成后随机删除
            all_elements = [(r, c) for r in range(1, n+1) for c in range(1, m+1)]
            random.shuffle(all_elements)
            q = random.randint(max(0, len(all_elements) - self.max_q), len(all_elements))
            elements = set(all_elements[:q])
        
        return {
            'n': n,
            'm': m,
            'elements': [[r, c] for r, c in sorted(elements)]
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        elements = question_case['elements']
        
        # 多语言问题描述支持
        element_desc = (
            "科学家已拥有以下元素样本：{}\n"
            if elements else 
            "实验室目前没有任何初始样本。\n"
        ).format(', '.join(f"第{r}行第{c}列" for r, c in elements)) if elements else ""
        
        return f"""## 元素合成问题
你正在管理一个{n}行{m}列的元素周期表实验室。{element_desc}
根据最新研究成果，当存在三个元素形成矩形顶点时，可以合成第四个顶点元素（合成过程不消耗原材料）。

**任务**：确定实验室至少需要购买多少新元素才能通过合成获得所有{n*m}个元素。

**输出要求**：将最终答案放在[answer]标签内，如：[answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强提取鲁棒性：处理小数点、中文数字等
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_match = matches[-1].strip()
        # 提取所有数字字符
        digits = re.sub(r'\D', '', last_match)
        return digits if digits else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        class OptimizedUnionFind:
            __slots__ = ['parent', 'count']
            def __init__(self, size):
                self.parent = list(range(size+1))
                self.count = size  # 初始连通分量数
            
            def find(self, x):
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]  # 路径压缩优化
                    x = self.parent[x]
                return x
            
            def union(self, x, y):
                fx = self.find(x)
                fy = self.find(y)
                if fx != fy:
                    # 小树合并到大树优化
                    if fx > fy:
                        fx, fy = fy, fx
                    self.parent[fx] = fy
                    self.count -= 1

        n = identity['n']
        m = identity['m']
        uf = OptimizedUnionFind(n + m)
        
        # 转换坐标为并查集节点
        for r, c in identity['elements']:
            uf.union(r, c + n)
        
        try:
            return int(solution) == (uf.count - 1)
        except (ValueError, TypeError):
            return False

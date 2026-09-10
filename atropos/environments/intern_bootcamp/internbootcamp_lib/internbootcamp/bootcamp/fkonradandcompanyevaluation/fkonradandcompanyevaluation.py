"""# 

### 谜题描述
Konrad is a Human Relations consultant working for VoltModder, a large electrical equipment producer. Today, he has been tasked with evaluating the level of happiness in the company.

There are n people working for VoltModder, numbered from 1 to n. Each employee earns a different amount of money in the company — initially, the i-th person earns i rubles per day.

On each of q following days, the salaries will be revised. At the end of the i-th day, employee v_i will start earning n+i rubles per day and will become the best-paid person in the company. The employee will keep his new salary until it gets revised again.

Some pairs of people don't like each other. This creates a great psychological danger in the company. Formally, if two people a and b dislike each other and a earns more money than b, employee a will brag about this to b. A dangerous triple is a triple of three employees a, b and c, such that a brags to b, who in turn brags to c. If a dislikes b, then b dislikes a.

At the beginning of each day, Konrad needs to evaluate the number of dangerous triples in the company. Can you help him do it?

Input

The first line contains two integers n and m (1 ≤ n ≤ 100 000, 0 ≤ m ≤ 100 000) — the number of employees in the company and the number of pairs of people who don't like each other. Each of the following m lines contains two integers a_i, b_i (1 ≤ a_i, b_i ≤ n, a_i ≠ b_i) denoting that employees a_i and b_i hate each other (that is, a_i dislikes b_i and b_i dislikes a_i). Each such relationship will be mentioned exactly once.

The next line contains an integer q (0 ≤ q ≤ 100 000) — the number of salary revisions. The i-th of the following q lines contains a single integer v_i (1 ≤ v_i ≤ n) denoting that at the end of the i-th day, employee v_i will earn the most.

Output

Output q + 1 integers. The i-th of them should contain the number of dangerous triples in the company at the beginning of the i-th day.

Examples

Input


4 5
1 2
2 4
1 3
3 4
2 3
2
2
3


Output


4
3
2


Input


3 3
1 2
2 3
1 3
5
1
2
2
1
3


Output


1
1
1
1
1
1

Note

Consider the first sample test. The i-th row in the following image shows the structure of the company at the beginning of the i-th day. A directed edge from a to b denotes that employee a brags to employee b. The dangerous triples are marked by highlighted edges.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = (int)1e5 + 1;
static vector<vector<int>> vert(MAXN, vector<int>(0));
static int outdeg[MAXN];
static int indeg[MAXN];
int n, m;
long long ans = 0;
static void solveShit() {
  int v;
  scanf(\"%d\", &v);
  ans -= (long long)indeg[v] * (long long)outdeg[v];
  for (int son : vert[v]) {
    ans -= (long long)indeg[son];
    ans += (long long)(outdeg[son] - 1);
    indeg[v]--;
    outdeg[v]++;
    indeg[son]++;
    outdeg[son]--;
    vert[son].push_back(v);
  }
  vert[v].clear();
  printf(\"%lld\n\", ans);
}
void solve() {
  scanf(\"%d%d\", &n, &m);
  for (int i = 0; i < m; i++) {
    int a, b;
    scanf(\"%d%d\", &a, &b);
    vert[min(a, b)].push_back(max(a, b));
    indeg[min(a, b)]++;
    outdeg[max(a, b)]++;
  }
  for (int i = 1; i < n + 1; i++) {
    ans += (long long)indeg[i] * (long long)outdeg[i];
  }
  printf(\"%lld\n\", ans);
  int q;
  scanf(\"%d\", &q);
  for (int i = 0; i < q; i++) {
    solveShit();
  }
}
int main() {
  std::ios_base::sync_with_stdio(false);
  cin.tie(0);
  solve();
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict
import re
import random
from bootcamp import Basebootcamp

def compute_expected_outputs(n, m, edges, queries):
    vert = defaultdict(list)
    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    
    # 修正边处理逻辑：u为较大编号的员工（初始薪金更高）
    for a, b in edges:
        u = max(a, b)
        v = min(a, b)
        vert[u].append(v)
        indeg[u] += 1
        outdeg[v] += 1
    
    ans = 0
    for i in range(1, n+1):
        ans += indeg[i] * outdeg[i]
    expected = [ans]
    
    for v in queries:
        # 移除当前节点贡献
        ans -= indeg[v] * outdeg[v]
        
        # 处理所有指向v的边（反向边）
        sons = list(vert[v])
        for son in sons:
            # 移除son节点原有贡献
            ans -= indeg[son]
            # 增加反转边后的贡献
            ans += (outdeg[son] - 1)
            
            # 调整度数
            indeg[v] -= 1
            outdeg[v] += 1
            indeg[son] += 1
            outdeg[son] -= 1
            
            # 添加反向边
            vert[son].append(v)
        
        # 清空原边
        vert[v].clear()
        # 添加新贡献
        ans += indeg[v] * outdeg[v]
        expected.append(ans)
    
    return expected

class Fkonradandcompanyevaluationbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=5, max_q=5):
        self.max_n = max_n
        self.max_m = max_m
        self.max_q = max_q

    def case_generator(self):
        # 确保至少1名员工
        n = random.randint(1, self.max_n)
        
        # 生成有效敌意对
        possible_pairs = []
        if n >= 2:
            for i in range(1, n+1):
                for j in range(i+1, n+1):
                    possible_pairs.append( (j, i) )  # 保证u > v
            random.shuffle(possible_pairs)
        
        m = min(self.max_m, len(possible_pairs))
        m = random.randint(0, m) if possible_pairs else 0
        edges = possible_pairs[:m]
        
        # 生成有效查询序列
        q = random.randint(0, self.max_q)
        queries = [random.randint(1, n) for _ in range(q)]
        
        # 计算期望输出
        expected = compute_expected_outputs(n, m, edges, queries)
        
        # 确保计算结果有效
        while len(expected) != q + 1:
            return self.case_generator()  # 重新生成
        
        return {
            'n': n,
            'm': m,
            'edges': edges,
            'queries': queries,
            'expected_outputs': expected
        }

    @staticmethod
    def prompt_func(question_case):
        case = question_case
        prompt = [
            "As Konrad, compute dangerous triples after each salary update.",
            f"Employees: {case['n']}, Dislike pairs: {case['m']}"
        ]
        if case['m'] > 0:
            prompt.append("Dislike relationships:")
            prompt.extend(f"{b} {a}" for a, b in case['edges'])  # 显示为原始输入顺序
        else:
            prompt.append("No dislike relationships.")
        
        prompt.append(f"Salary updates ({len(case['queries'])} days):")
        prompt.extend(map(str, case['queries']))
        
        prompt.append(
            "Output q+1 integers. Place answer list in [answer][/answer]."
            "\nExample: [answer][0,1,2][/answer]"
        )
        return "\n".join(prompt)

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\[.*?\])\s*\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            # 处理各种数字格式
            numbers = list(map(int, re.findall(r'-?\d+', last_match)))
            return numbers
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 严格验证答案长度和数值
        expected = identity['expected_outputs']
        return (
            isinstance(solution, list) and
            len(solution) == len(expected) and
            all(x == y for x, y in zip(solution, expected))
        )

"""# 

### 谜题描述
For a given sequence of distinct non-negative integers (b_1, b_2, ..., b_k) we determine if it is good in the following way:

  * Consider a graph on k nodes, with numbers from b_1 to b_k written on them.
  * For every i from 1 to k: find such j (1 ≤ j ≤ k, j≠ i), for which (b_i ⊕ b_j) is the smallest among all such j, where ⊕ denotes the operation of bitwise XOR (<https://en.wikipedia.org/wiki/Bitwise_operation#XOR>). Next, draw an undirected edge between vertices with numbers b_i and b_j in this graph.
  * We say that the sequence is good if and only if the resulting graph forms a tree (is connected and doesn't have any simple cycles). 



It is possible that for some numbers b_i and b_j, you will try to add the edge between them twice. Nevertheless, you will add this edge only once.

You can find an example below (the picture corresponding to the first test case). 

Sequence (0, 1, 5, 2, 6) is not good as we cannot reach 1 from 5.

However, sequence (0, 1, 5, 2) is good. 

<image>

You are given a sequence (a_1, a_2, ..., a_n) of distinct non-negative integers. You would like to remove some of the elements (possibly none) to make the remaining sequence good. What is the minimum possible number of removals required to achieve this goal?

It can be shown that for any sequence, we can remove some number of elements, leaving at least 2, so that the remaining sequence is good.

Input

The first line contains a single integer n (2 ≤ n ≤ 200,000) — length of the sequence.

The second line contains n distinct non-negative integers a_1, a_2, …, a_n (0 ≤ a_i ≤ 10^9) — the elements of the sequence.

Output

You should output exactly one integer — the minimum possible number of elements to remove in order to make the remaining sequence good.

Examples

Input


5
0 1 5 2 6


Output


1


Input


7
6 9 8 7 3 5 2


Output


2

Note

Note that numbers which you remove don't impact the procedure of telling whether the resulting sequence is good.

It is possible that for some numbers b_i and b_j, you will try to add the edge between them twice. Nevertheless, you will add this edge only once.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long mod = 998244353;
int size[6011100];
int l[6011100], r[6011100];
int id;
int ans;
int a[6011100];
void insert(int x) {
  int cur = 0;
  size[cur]++;
  for (int i = 29; i >= 0; --i) {
    if (l[cur] == -1) l[cur] = id++;
    if (r[cur] == -1) r[cur] = id++;
    cur = (1 << i) & x ? l[cur] : r[cur];
    size[cur]++;
    assert(id < 6011100);
  }
}
void dfs(int cur, int sum) {
  if (size[cur] == 2) {
    ans = max(ans, sum);
  }
  if (l[cur] >= 0) dfs(l[cur], sum + (size[r[cur]] > 0));
  if (r[cur] >= 0) dfs(r[cur], sum + (size[l[cur]] > 0));
}
int main() {
  memset(l, -1, sizeof(l));
  memset(r, -1, sizeof(r));
  id = 1;
  int n;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
    insert(a[i]);
  }
  ans = 0;
  dfs(0, 0);
  printf(\"%d\n\", n - ans - 2);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class TrieNode:
    def __init__(self):
        self.left = -1  # 左子节点索引，-1表示不存在
        self.right = -1  # 右子节点索引，-1表示不存在
        self.size = 0  # 当前节点的大小

def calculate_min_removals(a):
    nodes = [TrieNode()]
    root_id = 0
    id_counter = 1

    # 插入每个数到Trie中
    for num in a:
        cur = root_id
        nodes[cur].size += 1
        for i in reversed(range(30)):  # 处理30位，从高位到低位
            bit = (num >> i) & 1
            if bit:
                if nodes[cur].left == -1:
                    nodes[cur].left = id_counter
                    nodes.append(TrieNode())
                    id_counter += 1
                next_cur = nodes[cur].left
            else:
                if nodes[cur].right == -1:
                    nodes[cur].right = id_counter
                    nodes.append(TrieNode())
                    id_counter += 1
                next_cur = nodes[cur].right
            cur = next_cur
            nodes[cur].size += 1

    ans = 0

    def dfs(cur, current_sum):
        nonlocal ans
        if nodes[cur].size == 2:
            if current_sum > ans:
                ans = current_sum
        # 处理左子节点
        left_child = nodes[cur].left
        if left_child != -1:
            # 计算右子节点是否存在且size>0
            right_child = nodes[cur].right
            add = 1 if (right_child != -1 and nodes[right_child].size > 0) else 0
            dfs(left_child, current_sum + add)
        # 处理右子节点
        right_child = nodes[cur].right
        if right_child != -1:
            left_child = nodes[cur].left
            add = 1 if (left_child != -1 and nodes[left_child].size > 0) else 0
            dfs(right_child, current_sum + add)

    dfs(root_id, 0)
    return len(a) - ans - 2

class Cxortreebootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_val=1000):
        self.max_n = max_n  # 生成序列的最大长度
        self.max_val = max_val  # 生成数的最大值
    
    def case_generator(self):
        # 生成测试用例
        n = random.randint(2, self.max_n)
        elements = set()
        while len(elements) < n:
            elements.add(random.randint(0, self.max_val))
        elements = list(elements)
        random.shuffle(elements)
        # 计算正确答案
        correct_output = calculate_min_removals(elements)
        return {
            'input': elements,
            'output': correct_output
        }
    
    @staticmethod
    def prompt_func(question_case):
        elements = question_case['input']
        n = len(elements)
        elements_str = ' '.join(map(str, elements))
        example_input = "0 1 5 2 6"
        example_output = 1
        return f"""给定一个由不同非负整数组成的序列，你需要删除最少数量的元素，使得剩下的序列是“好的”。一个序列是“好的”当且仅当根据以下规则构建的图是一棵树：

- 对于每个元素b_i，找到另一个元素b_j（j≠i），使得b_i XOR b_j的值最小。在b_i和b_j之间添加一条无向边。
- 形成的图必须是一棵树（连通且无环）。

输入格式：
第一行是序列长度n，第二行是n个不同的非负整数。

输出格式：
输出需要删除的最少元素数目。

示例输入：
5
{example_input}
示例输出：
{example_output}

现在的问题实例是：
{n}
{elements_str}

请确保你的答案仅包含一个整数，并放置在[answer]标签内，例如：[answer]{example_output}[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        # 从输出中提取最后一个[answer]标签内的数字
        matches = re.findall(r'\[answer\](\d+)\[\/answer\]', output, re.IGNORECASE)
        if matches:
            return int(matches[-1])
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证答案是否正确
        return solution == identity['output']

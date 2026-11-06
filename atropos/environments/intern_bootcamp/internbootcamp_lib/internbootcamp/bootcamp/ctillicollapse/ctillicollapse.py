"""# 

### 谜题描述
Rick and Morty want to find MR. PBH and they can't do it alone. So they need of Mr. Meeseeks. They Have generated n Mr. Meeseeks, standing in a line numbered from 1 to n. Each of them has his own color. i-th Mr. Meeseeks' color is ai. 

Rick and Morty are gathering their army and they want to divide Mr. Meeseeks into some squads. They don't want their squads to be too colorful, so each squad should have Mr. Meeseeks of at most k different colors. Also each squad should be a continuous subarray of Mr. Meeseeks in the line. Meaning that for each 1 ≤ i ≤ e ≤ j ≤ n, if Mr. Meeseeks number i and Mr. Meeseeks number j are in the same squad then Mr. Meeseeks number e should be in that same squad.

<image>

Also, each squad needs its own presidio, and building a presidio needs money, so they want the total number of squads to be minimized.

Rick and Morty haven't finalized the exact value of k, so in order to choose it, for each k between 1 and n (inclusive) need to know the minimum number of presidios needed.

Input

The first line of input contains a single integer n (1 ≤ n ≤ 105) — number of Mr. Meeseeks.

The second line contains n integers a1, a2, ..., an separated by spaces (1 ≤ ai ≤ n) — colors of Mr. Meeseeks in order they standing in a line.

Output

In the first and only line of input print n integers separated by spaces. i-th integer should be the minimum number of presidios needed if the value of k is i.

Examples

Input

5
1 3 4 3 3


Output

4 2 1 1 1 


Input

8
1 5 7 8 1 7 6 1


Output

8 4 3 2 1 1 1 1 

Note

For the first sample testcase, some optimal ways of dividing army into squads for each k are:

  1. [1], [3], [4], [3, 3]
  2. [1], [3, 4, 3, 3]
  3. [1, 3, 4, 3, 3]
  4. [1, 3, 4, 3, 3]
  5. [1, 3, 4, 3, 3]



For the second testcase, some optimal ways of dividing army into squads for each k are:

  1. [1], [5], [7], [8], [1], [7], [6], [1]
  2. [1, 5], [7, 8], [1, 7], [6, 1]
  3. [1, 5, 7], [8], [1, 7, 6, 1]
  4. [1, 5, 7, 8], [1, 7, 6, 1]
  5. [1, 5, 7, 8, 1, 7, 6, 1]
  6. [1, 5, 7, 8, 1, 7, 6, 1]
  7. [1, 5, 7, 8, 1, 7, 6, 1]
  8. [1, 5, 7, 8, 1, 7, 6, 1]

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int nmax = 100010;
const int inf = (int)1e8;
int n, a[nmax], last[nmax], distinct;
int ans[nmax];
struct Node {
  Node *left, *right;
  int sum;
  Node() {
    left = right = NULL;
    sum = 0;
  }
  int getSum(Node *v) { return (v ? v->sum : 0); }
  void update() { this->sum = getSum(this->left) + getSum(this->right); }
  Node(Node *l, Node *r) {
    left = l;
    right = r;
    sum = getSum(left) + getSum(right);
  }
  Node(int el) {
    left = right = NULL;
    sum = el;
  }
};
Node *t[nmax];
Node *build(int tl, int tr) {
  if (tl == tr)
    return new Node(0);
  else {
    int m = (tl + tr) >> 1;
    return new Node(build(tl, m), build(m + 1, tr));
  }
}
Node *update(Node *t, int tl, int tr, int pos, int val) {
  if (tl == tr) return new Node(val);
  int m = (tl + tr) >> 1;
  if (pos <= m)
    return new Node(update(t->left, tl, m, pos, val), t->right);
  else
    return new Node(t->left, update(t->right, m + 1, tr, pos, val));
}
int get(Node *t, int tl, int tr, int k) {
  if (tl == tr) return (k == 0 && t->sum == 1 ? tl - 1 : tl);
  int m = (tl + tr) >> 1;
  if (t->left->sum <= k) return get(t->right, m + 1, tr, k - t->left->sum);
  return get(t->left, tl, m, k);
}
void println(Node *t, int tl, int tr) {
  if (tl == tr)
    cout << t->sum << \" \";
  else {
    int m = (tl + tr) >> 1;
    println(t->left, tl, m);
    println(t->right, m + 1, tr);
  }
}
int calc(int k) {
  int i = 1;
  int groups = 0;
  while (i <= n) {
    int j = get(t[i], 1, n + 1, k);
    i = j + 1;
    groups++;
  }
  return groups;
}
int main() {
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; i++) {
    scanf(\"%d\", &a[i]);
  }
  t[n] = build(1, n + 1);
  t[n] = update(t[n], 1, n + 1, n, 1);
  last[a[n]] = n;
  for (int i = n - 1, x; i >= 1; i--) {
    x = a[i];
    t[i] = t[i + 1];
    if (last[x] != 0) t[i] = update(t[i], 1, n + 1, last[x], 0);
    t[i] = update(t[i], 1, n + 1, i, 1);
    last[x] = i;
  }
  for (int i = 1; i <= n; i++) cout << calc(i) << \" \";
  cout << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ctillicollapsebootcamp(Basebootcamp):
    def __init__(self, max_n=20):
        self.max_n = max_n  # 控制生成谜题的最大规模

    def case_generator(self):
        """生成具有多样性的测试用例"""
        n = random.randint(1, self.max_n)
        
        # 生成颜色数组（确保至少2种颜色）
        colors = list(range(1, min(n, 5)+1))  # 颜色值限制在1-5范围以增加重复概率
        a = [random.choice(colors) for _ in range(n)]
        
        # 对部分元素做随机扰动
        for _ in range(int(n**0.5)):
            a[random.randint(0, n-1)] = random.choice(colors)
            
        ans = [self._calculate_min_squads(a, k) for k in range(1, n+1)]
        return {'n': n, 'a': a, 'ans': ans}

    @staticmethod
    def _calculate_min_squads(a, k):
        """优化后的贪心算法实现"""
        n = len(a)
        count = 0
        start = 0
        
        while start < n:
            color_dict = {}
            distinct = 0
            max_end = start
            
            # 滑动窗口寻找最大有效区间
            for end in range(start, n):
                color = a[end]
                if color not in color_dict or color_dict[color] == 0:
                    distinct += 1
                color_dict[color] = color_dict.get(color, 0) + 1
                
                if distinct > k:
                    # 回退最后一步
                    color_dict[color] -= 1
                    if color_dict[color] == 0:
                        distinct -= 1
                    break
                
                max_end = end
                
            count += 1
            start = max_end + 1
            
        return count

    @staticmethod
    def prompt_func(question_case):
        """增强格式说明的prompt模板"""
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        return (
            "## Mission\nRick和Morty需要将Mr. Ctillicollapse分成连续的squad（每个squad最多k种颜色）\n\n"
            "## Input Format\n"
            "- 第1行：整数n (人数)\n"
            "- 第2行：空格分隔的颜色序列\n\n"
            "## Output Format\n"
            "- 1行：n个空格分隔的整数，第i个数表示k=i时的最小squad数\n\n"
            "## Example\n"
            "Input:\n5\n1 3 4 3 3\n"
            "Output:\n4 2 1 1 1 → 应格式化为：[answer]4 2 1 1 1[/answer]\n\n"
            "## Current Problem\n"
            f"{n}\n{a}\n\n"
            "Answer with [answer]...[/answer]"
        )

    @staticmethod
    def extract_output(output):
        """鲁棒的答案提取方法"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
            
        last_match = matches[-1].strip()
        cleaned = re.sub(r'\s+', ' ', last_match)  # 合并多余空白
        return cleaned if cleaned else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """增强验证稳定性的检查"""
        try:
            # 处理首尾可能的换行符
            ans_str = solution.strip()
            # 处理多空格情况
            ans_list = list(map(int, ans_str.split()))
            return ans_list == identity['ans']
        except:
            return False

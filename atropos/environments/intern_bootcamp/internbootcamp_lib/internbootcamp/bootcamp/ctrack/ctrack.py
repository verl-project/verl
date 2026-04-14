"""# 

### 谜题描述
You already know that Valery's favorite sport is biathlon. Due to your help, he learned to shoot without missing, and his skills are unmatched at the shooting range. But now a smaller task is to be performed, he should learn to complete the path fastest.

The track's map is represented by a rectangle n × m in size divided into squares. Each square is marked with a lowercase Latin letter (which means the type of the plot), with the exception of the starting square (it is marked with a capital Latin letters S) and the terminating square (it is marked with a capital Latin letter T). The time of movement from one square to another is equal to 1 minute. The time of movement within the cell can be neglected. We can move from the cell only to side-adjacent ones, but it is forbidden to go beyond the map edges. Also the following restriction is imposed on the path: it is not allowed to visit more than k different types of squares (squares of one type can be visited an infinite number of times). Squares marked with S and T have no type, so they are not counted. But S must be visited exactly once — at the very beginning, and T must be visited exactly once — at the very end.

Your task is to find the path from the square S to the square T that takes minimum time. Among all shortest paths you should choose the lexicographically minimal one. When comparing paths you should lexicographically represent them as a sequence of characters, that is, of plot types.

Input

The first input line contains three integers n, m and k (1 ≤ n, m ≤ 50, n·m ≥ 2, 1 ≤ k ≤ 4). Then n lines contain the map. Each line has the length of exactly m characters and consists of lowercase Latin letters and characters S and T. It is guaranteed that the map contains exactly one character S and exactly one character T.

Pretest 12 is one of the maximal tests for this problem.

Output

If there is a path that satisfies the condition, print it as a sequence of letters — the plot types. Otherwise, print \"-1\" (without quotes). You shouldn't print the character S in the beginning and T in the end.

Note that this sequence may be empty. This case is present in pretests. You can just print nothing or print one \"End of line\"-character. Both will be accepted.

Examples

Input

5 3 2
Sba
ccc
aac
ccc
abT


Output

bcccc


Input

3 4 1
Sxyy
yxxx
yyyT


Output

xxxx


Input

1 3 3
TyS


Output

y


Input

1 4 1
SxyT


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 55;
int n, m, k;
set<set<int> > ha[maxn][maxn];
string mat[maxn];
int stran[4][2] = {1, 0, -1, 0, 0, 1, 0, -1};
int br, bc;
int er, ec;
int dis(int r1, int c1, int r2, int c2) { return abs(r1 - r2) + abs(c1 - c2); }
struct node {
  int r, c;
  string s;
  int bu;
  int cu;
  string used;
  friend bool operator<(const node &a, const node &b) {
    if (a.bu + dis(a.r, a.c, er, ec) == b.bu + dis(b.r, b.c, er, ec)) {
      return a.s > b.s;
    }
    return a.bu + dis(a.r, a.c, er, ec) > b.bu + dis(b.r, b.c, er, ec);
  }
};
priority_queue<node> que;
set<int> uu;
int main() {
  cin >> n >> m >> k;
  for (int i = 0; i < n; i++) {
    cin >> mat[i];
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (mat[i][j] == 'S') {
        br = i;
        bc = j;
      }
      if (mat[i][j] == 'T') {
        er = i;
        ec = j;
      }
    }
  }
  node now, ne;
  now.bu = 0;
  now.s.clear();
  now.r = br;
  now.c = bc;
  now.cu = 0;
  now.used.clear();
  que.push(now);
  while (!que.empty()) {
    now = que.top();
    que.pop();
    if (mat[now.r][now.c] == 'T') {
      cout << now.s << endl;
      return 0;
    }
    string ss = now.used, ness;
    string ro = now.s;
    int nr = now.r, nc = now.c;
    uu.clear();
    for (int i = 0; i < ss.length(); i++) {
      uu.insert(ss[i] - 'a');
    }
    if (ha[nr][nc].find(uu) != ha[nr][nc].end()) continue;
    ha[nr][nc].insert(uu);
    int ner, nec;
    for (int i = 0; i < 4; i++) {
      ner = nr + stran[i][0];
      nec = nc + stran[i][1];
      if (ner >= 0 && ner < n && nec >= 0 && nec < m) {
        char p = mat[ner][nec];
        ne.r = ner;
        ne.c = nec;
        ne.bu = now.bu + 1;
        if (p != 'T' && p != 'S') {
          bool hu = 0;
          for (int j = 0; j < ss.length(); j++) {
            if (p == ss[j]) {
              hu = 1;
              break;
            }
          }
          if (hu == 1) {
            ne.used = ss;
            ne.cu = now.cu;
            ne.s = now.s + p;
            que.push(ne);
          } else {
            if (now.cu + 1 <= k) {
              ne.used = ss + p;
              ne.cu = now.cu + 1;
              ne.s = now.s + p;
              que.push(ne);
            }
          }
        } else if (p == 'T') {
          ne = now;
          ne.bu++;
          ne.r = er;
          ne.c = ec;
          que.push(ne);
        }
      }
    }
  }
  cout << \"-1\" << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from heapq import heappop, heappush
from bootcamp import Basebootcamp

def manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

def solve(n, m, k, mat):
    start = None
    end = None
    for i in range(n):
        for j in range(m):
            if mat[i][j] == 'S':
                start = (i, j)
            elif mat[i][j] == 'T':
                end = (i, j)
    if not start or not end:
        return "-1"
    br, bc = start
    er, ec = end

    heap = []
    initial_priority = manhattan(br, bc, er, ec)
    heappush(heap, (initial_priority, '', 0, br, bc, 0, ''))
    ha = {i: {j: set() for j in range(m)} for i in range(n)}

    while heap:
        priority, path, steps, r, c, cu, used_str = heappop(heap)
        if (r, c) == (er, ec):
            return path
        if used_str in ha[r][c]:
            continue
        ha[r][c].add(used_str)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m:
                ch = mat[nr][nc]
                if ch == 'S':
                    continue
                new_steps = steps + 1
                new_priority = new_steps + manhattan(nr, nc, er, ec)
                if ch == 'T':
                    heappush(heap, (new_priority, path, new_steps, nr, nc, cu, used_str))
                else:
                    if ch in used_str:
                        new_used = used_str
                        new_cu = cu
                    else:
                        new_cu = cu + 1
                        if new_cu > k:
                            continue
                        new_used = ''.join(sorted(set(used_str) | {ch}))
                    new_path = path + ch
                    if new_used not in ha[nr][nc]:
                        heappush(heap, (new_priority, new_path, new_steps, nr, nc, new_cu, new_used))
    return "-1"

class Ctrackbootcamp(Basebootcamp):
    def __init__(self, n=5, m=3, k=2, max_attempts=100):
        self.n = n
        self.m = m
        self.k = k
        self.max_attempts = max_attempts

    def case_generator(self):
        for _ in range(self.max_attempts):
            while True:
                s_pos = (random.randint(0, self.n-1), random.randint(0, self.m-1))
                t_pos = (random.randint(0, self.n-1), random.randint(0, self.m-1))
                if s_pos != t_pos:
                    break

            map_data = [
                [
                    'S' if (i, j) == s_pos else
                    'T' if (i, j) == t_pos else
                    random.choice('abcdefghijklmnopqrstuvwxyz')
                    for j in range(self.m)
                ]
                for i in range(self.n)
            ]
            map_data = [''.join(row) for row in map_data]

            expected_output = solve(self.n, self.m, self.k, map_data)
            if expected_output != '-1':
                return {
                    'n': self.n,
                    'm': self.m,
                    'k': self.k,
                    'map': map_data,
                    'expected_output': expected_output
                }
        raise ValueError("Failed to generate valid case after multiple attempts")

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        map_data = question_case['map']
        input_example = f"{n} {m} {k}\n" + "\n".join(map_data)
        prompt = f"""你是冬季两项比赛的路径规划专家，请帮助Valery找到从起点S到终点T的最短路径。路径需要满足以下规则：

1. 移动规则：每次只能移动到相邻的单元格（上下左右），不能越界。
2. 类型限制：路径中经过的不同单元格类型（小写字母）的数量不能超过{k}个。S和T不计入类型且不能重复访问。
3. 最短且字典序最小：在满足条件的最短路径中，选择字典序最小的路径。路径的字典序比较基于各单元格类型的字符顺序。

输入格式：
第一行包含三个整数n、m、k。
接下来n行，每行m个字符表示地图，包含恰好一个S和一个T。

输出格式：
如果存在合法路径，输出路径字符串（不包含S和T的字符）。否则输出-1。

当前谜题实例：
{input_example}

请将你的答案放在[answer]和[/answer]标签之间。例如：[answer]abc[/answer] 或 [answer]-1[/answer]。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip().replace('\n', '').replace(' ', '')
        if answer == '-1':
            return '-1'
        if all(c.islower() for c in answer):
            return answer
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_output']
        return solution == expected

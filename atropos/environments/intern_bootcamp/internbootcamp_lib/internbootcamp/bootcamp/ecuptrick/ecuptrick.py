"""# 

### 谜题描述
The employees of the F company have lots of ways to entertain themselves. Today they invited a famous magician who shows a trick with plastic cups and a marble.

The point is to trick the spectator's attention. Initially, the spectator stands in front of a line of n plastic cups. Then the magician places a small marble under one cup and shuffles the cups. Then the spectator should guess which cup hides the marble.

But the head coder of the F company isn't easy to trick. When he saw the performance, he noticed several important facts:

  * each cup contains a mark — a number from 1 to n; all marks on the cups are distinct; 
  * the magician shuffles the cups in m operations, each operation looks like that: take a cup marked xi, sitting at position yi in the row of cups (the positions are numbered from left to right, starting from 1) and shift it to the very beginning of the cup row (on the first position). 



When the head coder came home after work he wanted to re-do the trick. Unfortunately, he didn't remember the starting or the final position of the cups. He only remembered which operations the magician performed. Help the coder: given the operations in the order they were made find at least one initial permutation of the cups that can go through the described operations in the given order. Otherwise, state that such permutation doesn't exist.

Input

The first line contains integers n and m (1 ≤ n, m ≤ 106). Each of the next m lines contains a couple of integers. The i-th line contains integers xi, yi (1 ≤ xi, yi ≤ n) — the description of the i-th operation of the magician. Note that the operations are given in the order in which the magician made them and the coder wants to make them in the same order.

Output

If the described permutation doesn't exist (the programmer remembered wrong operations), print -1. Otherwise, print n distinct integers, each from 1 to n: the i-th number should represent the mark on the cup that initially is in the row in position i.

If there are multiple correct answers, you should print the lexicographically minimum one.

Examples

Input

2 1
2 1


Output

2 1 


Input

3 2
1 2
1 1


Output

2 1 3 


Input

3 3
1 3
2 3
1 3


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int inf = int(1e9 + 7);
const int maxn = (1e6) + 7;
int n, a[maxn];
struct TreeNode {
  int minpos, sum;
  void update(int i) {
    this->minpos = (a[i] == 0 ? i : inf);
    this->sum = a[i];
  }
  void update(const TreeNode &le, const TreeNode &ri) {
    this->minpos = min(le.minpos, ri.minpos);
    this->sum = le.sum + ri.sum;
  }
};
TreeNode T[4 * maxn];
void update(int b, int e, int node, int i, int val) {
  if (i < b || i > e) return;
  if (b == e)
    a[i] = val, T[node].update(i);
  else {
    int mid = (b + e) / 2, le = 2 * node + 1, ri = 2 * node + 2;
    update(b, mid, le, i, val);
    update(mid + 1, e, ri, i, val);
    T[node].update(T[le], T[ri]);
  }
}
inline void update(int i, int val) { update(0, n - 1, 0, i, val); }
void init(int b, int e, int node) {
  if (b == e)
    T[node].update(b);
  else {
    int mid = (b + e) / 2, le = 2 * node + 1, ri = 2 * node + 2;
    init(b, mid, le);
    init(mid + 1, e, ri);
    T[node].update(T[le], T[ri]);
  }
}
inline void init() { init(0, n - 1, 0); }
TreeNode query(int b, int e, int node, int i, int j) {
  if (i <= b && e <= j) return T[node];
  int mid = (b + e) / 2, le = 2 * node + 1, ri = 2 * node + 2;
  TreeNode ret;
  if (j <= mid)
    ret = query(b, mid, le, i, j);
  else if (mid < i)
    ret = query(mid + 1, e, ri, i, j);
  else {
    TreeNode ret1, ret2;
    ret1 = query(b, mid, le, i, j);
    ret2 = query(mid + 1, e, ri, i, j);
    ret.update(ret1, ret2);
  }
  return ret;
}
inline TreeNode query(int i, int j) { return query(0, n - 1, 0, i, j); }
struct Event {
  int start, end, val, expected;
  Event(int ss, int ee, int vv, int exx)
      : start(ss), end(ee), val(vv), expected(exx) {}
  bool operator<(const Event &rhs) const {
    if (end != rhs.end) return end < rhs.end;
    return val < rhs.val;
  }
};
int lastseen[maxn], sol[maxn];
vector<Event> events;
int find(int y) {
  int b = y, lo = 0, hi = n - 1, node = 0;
  while (lo < hi) {
    int mid = (lo + hi) / 2, le = 2 * node + 1, ri = 2 * node + 2;
    int p = T[ri].minpos, x = T[ri].sum - (p == inf ? 0 : p - mid - 1);
    if (p + x <= b) {
      node = ri;
      lo = mid + 1;
    } else {
      node = le;
      hi = mid;
      b -= T[ri].sum;
    }
  }
  return (a[lo] == 0 && lo == b ? lo : inf);
}
int main() {
  int m;
  bool valid;
  while (cin >> n >> m) {
    valid = true;
    memset(sol, 0, sizeof sol);
    memset(a, 0, sizeof a), init();
    memset(lastseen, -1, sizeof lastseen);
    events.clear();
    for (int q = 0; q < int(m); ++q) {
      int x, y;
      scanf(\"%d %d\", &x, &y);
      if (!valid) continue;
      events.push_back(Event(q, q, x, inf));
      if (lastseen[x] != -1 && lastseen[x] + 1 <= q - 1) {
        events.push_back(Event(lastseen[x] + 1, q - 1, inf, y - 1));
      }
      if (lastseen[x] == -1) {
        int p = find(y - 1);
        if (p == inf) {
          valid = false;
        } else {
          update(p, 1);
          sol[p] = x;
        }
      }
      lastseen[x] = q;
    }
    if (n != 1e6 && m != 1e6 && valid && (int)events.size() > m) {
      memset(a, 0, sizeof a), init();
      memset(lastseen, -1, sizeof lastseen);
      sort((events).begin(), (events).end());
      for (int i = 0; i < int(events.size()); ++i) {
        if (events[i].val == inf) {
          TreeNode r = query(events[i].start, events[i].end);
          if (r.sum != events[i].expected) {
            valid = false;
            break;
          }
        } else {
          int x = events[i].val;
          if (lastseen[x] != -1) update(lastseen[x], 0);
          lastseen[x] = events[i].start;
          update(events[i].start, 1);
        }
      }
    }
    if (valid) {
      int x = 1;
      for (int i = 0; i < int(n); ++i) {
        if (sol[i]) {
          printf(\"%d \", sol[i]);
        } else {
          while (lastseen[x] != -1) ++x;
          printf(\"%d \", x++);
        }
      }
      puts(\"\");
    } else
      puts(\"-1\");
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ecuptrickbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', random.randint(2, 5))
        self.m = params.get('m', random.randint(1, 3))
        self.gen_valid_operations = params.get('gen_valid', True)  # 控制生成合法操作
    
    def case_generator(self):
        """逆向生成合法的操作序列"""
        n, m = self.n, self.m
        
        # 生成目标排列
        final = list(range(1, n+1))
        random.shuffle(final)
        
        # 逆向构造操作序列
        operations = []
        pos_dict = {x: i+1 for i, x in enumerate(final)}  # 当前每个杯子位置
        
        for _ in range(m):
            # 选择要移动的杯子(不能重复)
            candidates = list(pos_dict.keys())
            if not candidates: break
            xi = random.choice(candidates)
            
            # 当前实际位置
            yi = pos_dict[xi]
            operations.append((xi, yi))
            
            # 逆向操作：将xi移到位置yi (逆向即需要先将它放在最前面)
            del pos_dict[xi]
            pos_dict = {k: v+1 for k, v in pos_dict.items()}  # 其他杯子后移
            pos_dict[xi] = 1  # 新插入到最前面
            
        operations.reverse()  # 反向存储操作顺序
        
        # 计算初始排列
        initial = sorted(pos_dict.items(), key=lambda x: x[1])
        initial = [x[0] for x in initial]
        
        return {
            'n': n,
            'm': m,
            'operations': operations,
            'correct_answer': initial
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        ops = '\n'.join(f"{xi} {yi}" for xi, yi in question_case['operations'])
        return f"""Given {n} cups and {m} operations, find the lex-min initial permutation. Operations are given in chronological order. Each operation moves cup xi from position yi to front. Output -1 if impossible. Put your answer between [answer] and [/answer].

Input:
{n} {m}
{ops}

Example:
[answer]
1 2 3
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip().replace('\n', ' ')
        if answer == '-1':
            return -1
        try:
            return list(map(int, answer.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 完整答案验证需要模拟操作流程
        try:
            if solution == -1:
                return False  # 我们保证生成的case都有解
            
            n = identity['n']
            cups = solution.copy()
            for xi, yi in identity['operations']:
                try:
                    pos = cups.index(xi)
                except ValueError:
                    return False  # 杯子不存在
                if pos + 1 != yi:
                    return False  # 位置不符
                # 执行移动操作
                cups = [xi] + cups[:pos] + cups[pos+1:]
            return True
        except:
            return False

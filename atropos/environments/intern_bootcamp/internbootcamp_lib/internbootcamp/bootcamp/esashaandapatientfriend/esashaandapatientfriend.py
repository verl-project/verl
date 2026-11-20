"""# 

### 谜题描述
Fedya and Sasha are friends, that's why Sasha knows everything about Fedya.

Fedya keeps his patience in an infinitely large bowl. But, unlike the bowl, Fedya's patience isn't infinite, that is why let v be the number of liters of Fedya's patience, and, as soon as v becomes equal to 0, the bowl will burst immediately. There is one tap in the bowl which pumps s liters of patience per second. Notice that s can be negative, in that case, the tap pumps out the patience. Sasha can do different things, so he is able to change the tap's speed. All actions that Sasha does can be represented as q queries. There are three types of queries:

  1. \"1 t s\" — add a new event, means that starting from the t-th second the tap's speed will be equal to s. 
  2. \"2 t\" — delete the event which happens at the t-th second. It is guaranteed that such event exists. 
  3. \"3 l r v\" — Sasha wonders: if you take all the events for which l ≤ t ≤ r and simulate changes of Fedya's patience from the very beginning of the l-th second till the very beginning of the r-th second inclusive (the initial volume of patience, at the beginning of the l-th second, equals to v liters) then when will be the moment when the bowl will burst. If that does not happen, then the answer will be -1. 



Since Sasha does not want to check what will happen when Fedya's patience ends, and he has already come up with the queries, he is asking you to help him and find the answer for each query of the 3-rd type.

It is guaranteed that at any moment of time, there won't be two events which happen at the same second.

Input

The first line contans one integer q (1 ≤ q ≤ 10^5) — the number of queries.

Each of the next q lines have one of the following formats:

  * 1 t s (1 ≤ t ≤ 10^9, -10^9 ≤ s ≤ 10^9), means that a new event is added, which means that starting from the t-th second the tap's speed will be equal to s. 
  * 2 t (1 ≤ t ≤ 10^9), means that the event which happens at the t-th second must be deleted. Guaranteed that such exists. 
  * 3 l r v (1 ≤ l ≤ r ≤ 10^9, 0 ≤ v ≤ 10^9), means that you should simulate the process from the very beginning of the l-th second till the very beginning of the r-th second inclusive, and to say when will the bowl burst. 



It is guaranteed that t, s, l, r, v in all the queries are integers.

Also, it is guaranteed that there is at least one query of the 3-rd type, and there won't be a query of the 1-st type with such t, that there already exists an event which happens at that second t.

Output

For each query of the 3-rd type, print in a new line the moment when the bowl will burst or print -1 if it won't happen.

Your answer will be considered correct if it's absolute or relative error does not exceed 10^{-6}.

Formally, let your answer be a, and the jury's answer be b. Your answer is accepted if and only if \frac{|a - b|}{max{(1, |b|)}} ≤ 10^{-6}.

Examples

Input


6
1 2 1
1 4 -3
3 1 6 1
3 1 6 3
3 1 6 4
3 1 6 5


Output


5
5.666667
6
-1


Input


10
1 2 2
1 4 4
1 7 -10
3 2 4 1
3 5 6 0
3 1 15 1
2 4
3 1 15 1
1 8 1
3 1 15 1


Output


-1
5
8.7
8.1
-1


Input


5
1 1000 9999999
1 2000 -9999
3 1000 2000 0
2 1000
3 1000 2002 1


Output


1000
2000.0001

Note

In the first example all the queries of the 3-rd type cover all the events, it's simulation is following:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
mt19937 gen(200);
struct node {
  pair<int, int> f, s;
  node *le, *ri;
  long long mn, res;
  int tl, tr;
  node() { mn = 228; }
  node(int l, int r) {
    tl = l;
    tr = r;
    mn = 228;
    if (l == r) return;
    le = new node(l, (l + r) / 2);
    ri = new node((l + r) / 2 + 1, r);
  }
  node(node* a, node* b) {
    le = a;
    ri = b;
    tl = le->tl;
    tr = ri->tr;
    if (le->mn == 228) {
      res = ri->res;
      mn = ri->mn;
      f = ri->f;
      s = ri->s;
      return;
    }
    if (ri->mn == 228) {
      res = le->res;
      mn = le->mn;
      f = le->f;
      s = le->s;
      return;
    }
    f = le->f;
    s = ri->s;
    long long del = 1ll * le->s.second * (ri->f.first - le->s.first);
    res = le->res + del + ri->res;
    mn = min(le->mn, le->res + del);
    mn = min(mn, ri->mn + le->res + del);
  }
  void combine() {
    node tmp(le, ri);
    *this = tmp;
  }
  void update(int id, int time, int speed) {
    if (tl == tr) {
      mn = res = 0;
      f.first = s.first = time;
      f.second = s.second = speed;
      return;
    }
    if (id <= (tl + tr) / 2)
      le->update(id, time, speed);
    else
      ri->update(id, time, speed);
    combine();
  }
  void del(int id) {
    if (tl == tr) {
      mn = 228;
      return;
    }
    if (id <= (tl + tr) / 2)
      le->del(id);
    else
      ri->del(id);
    combine();
  }
  node* get_seg(int l, int r) {
    if (tr < l || r < tl) return new node();
    if (l <= tl && tr <= r) {
      return this;
      ;
    }
    return new node(le->get_seg(l, r), ri->get_seg(l, r));
  }
  long double simulate(int r, long long v) {
    if (mn == 228) return -1;
    if (v + mn > 0 && v + res + 1ll * s.second * (r - s.first) > 0) return -1;
    if (f == s) return s.first - (long double)v / s.second;
    if (le->mn == 228) return ri->simulate(r, v);
    long double to = le->simulate(le->s.first, v);
    if (to != -1) return to;
    v += le->res;
    long long del =
        1ll * le->s.second * ((ri->mn == 228 ? r : ri->f.first) - le->s.first);
    if (v + del <= 0) return le->s.first - (long double)v / le->s.second;
    v += del;
    return ri->simulate(r, v);
  }
  long double query(int l, int r, int rr, long long v) {
    node* t = get_seg(l, r);
    return t->simulate(rr, v);
  }
};
struct query {
  int time, speed, start;
  int l, r, type;
  query() {}
};
vector<int> pos;
vector<query> q;
int main() {
  ios_base::sync_with_stdio(false);
  cout.precision(7);
  int qq;
  cin >> qq;
  q.resize(qq);
  for (int i = 0; i < qq; ++i) {
    cin >> q[i].type;
    if (q[i].type == 1) {
      cin >> q[i].time >> q[i].speed;
      pos.push_back(q[i].time);
    }
    if (q[i].type == 2) {
      cin >> q[i].time;
    }
    if (q[i].type == 3) {
      cin >> q[i].l >> q[i].r >> q[i].start;
    }
  }
  sort(pos.begin(), pos.end());
  pos.erase(unique(pos.begin(), pos.end()), pos.end());
  if (pos.size() == 0) pos.push_back(0);
  node* t = new node(0, pos.size() - 1);
  for (int i = 0; i < qq; ++i) {
    if (q[i].type == 1) {
      int id = lower_bound(pos.begin(), pos.end(), q[i].time) - pos.begin();
      t->update(id, q[i].time, q[i].speed);
    }
    if (q[i].type == 2) {
      int id = lower_bound(pos.begin(), pos.end(), q[i].time) - pos.begin();
      t->del(id);
    }
    if (q[i].type == 3) {
      int l = lower_bound(pos.begin(), pos.end(), q[i].l) - pos.begin();
      int r = --upper_bound(pos.begin(), pos.end(), q[i].r) - pos.begin();
      if (q[i].start == 0)
        cout << fixed << q[i].l << '\n';
      else
        cout << fixed << t->query(l, r, q[i].r, q[i].start) << '\n';
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import bisect
from bisect import bisect_left, bisect_right
from bootcamp import Basebootcamp

class Esashaandapatientfriendbootcamp(Basebootcamp):
    def __init__(self, max_queries=10, max_time=int(1e9), max_speed=int(1e9)):
        super().__init__()
        self.max_queries = max_queries
        self.max_time = max_time
        self.max_speed = max_speed

    def case_generator(self):
        event_times = []  # 维护有序事件时间
        events = dict()   # 时间到速度的映射
        queries = []
        
        # 生成基础查询
        q_count = random.randint(3, self.max_queries)
        for _ in range(q_count-1):
            if not event_times or random.random() < 0.7:
                # 生成类型1事件
                while True:
                    t = random.randint(1, self.max_time)
                    if t not in events:
                        break
                s = random.randint(-self.max_speed, self.max_speed)
                bisect.insort(event_times, t)
                events[t] = s
                queries.append({"type": 1, "t": t, "s": s})
            else:
                # 生成类型2事件
                idx = random.randrange(len(event_times))
                t = event_times.pop(idx)
                del events[t]
                queries.append({"type": 2, "t": t})

        # 生成类型3查询
        l, r = self._gen_lr(event_times)
        v = random.randint(0, self.max_speed)
        
        # 筛选有效事件
        left = bisect.bisect_left(event_times, l)
        right = bisect.bisect_right(event_times, r)
        valid_events = [{"t": t, "s": events[t]} for t in event_times[left:right]]

        expected = self._simulate(valid_events, l, r, v)
        queries.append({
            "type": 3,
            "l": l,
            "r": r,
            "v": v,
            "expected": expected
        })

        return {
            "queries": queries,
            "expected": expected,
            "events": events
        }

    def _gen_lr(self, event_times):
        """生成合理的l和r范围"""
        if event_times:
            min_t = event_times[0]
            max_t = event_times[-1]
            l = random.randint(max(1, min_t-10), max_t+10)
            r = random.randint(l, min(self.max_time, max_t+1000))
        else:
            l = random.randint(1, 100)
            r = random.randint(l, min(self.max_time, l+1000))
        return l, r

    @staticmethod
    def _simulate(events, l, r, v_initial):
        if v_initial == 0:
            return l  # 初始值为0立即破裂

        current_time = l
        current_speed = 0  # 初始速度
        v = v_initial
        sorted_events = sorted(events, key=lambda x: x["t"])

        for event in sorted_events:
            t_event = event["t"]
            s_new = event["s"]

            # 处理当前时间段 [current_time, t_event)
            if t_event > current_time:
                dt = t_event - current_time
                if current_speed < 0:
                    # 计算在当前速度下是否会耗尽
                    if v <= 0:
                        return current_time
                    time_to_empty = v / (-current_speed)
                    if time_to_empty <= dt:
                        return current_time + time_to_empty
                    # 不会耗尽，更新v和时间
                    v += current_speed * dt
                    current_time = t_event
                else:
                    v += current_speed * dt
                    current_time = t_event
                if v <= 0:
                    return current_time  # 刚好在时间点耗尽

            # 更新速度
            current_speed = s_new

        # 处理最后的时间段 [current_time, r)
        dt = r - current_time
        if dt > 0:
            if current_speed < 0:
                if v <= 0:
                    return current_time
                time_to_empty = v / (-current_speed)
                if time_to_empty <= dt:
                    return current_time + time_to_empty
                v += current_speed * dt
            else:
                v += current_speed * dt
            if v <= 0:
                return r  # 在结束时间点耗尽

        return -1 if v > 0 else r

    @staticmethod
    def prompt_func(question_case):
        queries = question_case["queries"]
        input_lines = [str(len(queries))]
        for q in queries:
            if q["type"] == 1:
                input_lines.append(f"1 {q['t']} {q['s']}")
            elif q["type"] == 2:
                input_lines.append(f"2 {q['t']}")
            else:
                input_lines.append(f"3 {q['l']} {q['r']} {q['v']}")

        problem_desc = (
            "Fedya's Patience Simulation\n\n"
            "Rules:\n"
            "1. Bowl bursts when patience (v) ≤ 0\n"
            "2. Events change tap speed at specific seconds\n"
            "3. Type 3 queries simulate from l to r with initial v\n\n"
            f"Input Queries:\n" + "\n".join(input_lines) + "\n\n"
            "Compute the exact burst time (with 6 decimal places if needed) "
            "or -1 if it doesn't burst.\n"
            "Format your answer as: [answer]<result>[/answer]"
        )
        return problem_desc

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip()
        
        # 处理特殊值-1
        if answer.lower() == "-1":
            return -1.0
        
        # 处理科学计数法和浮点格式
        try:
            return float(answer.replace(',', '.'))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity["expected"]
        # 处理-1的情况
        if expected == -1:
            return solution == -1.0
        if solution == -1.0:
            return False
        
        # 计算精度误差
        abs_err = abs(solution - expected)
        rel_err = abs_err / max(1.0, abs(expected))
        return abs_err <= 1e-6 or rel_err <= 1e-6

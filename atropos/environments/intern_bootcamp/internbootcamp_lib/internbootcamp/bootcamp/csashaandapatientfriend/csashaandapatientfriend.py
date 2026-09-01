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
const int inf = 0x3f3f3f3f;
class node {
 public:
  node* l;
  node* r;
  node* p;
  int time, ltime, rtime, speed, rspeed;
  long long sum, lsum;
  node(int time, int speed) : time(time), speed(speed) {
    l = r = p = NULL;
    ltime = rtime = time;
    rspeed = speed;
    sum = lsum = 0;
  }
  void pull() {
    ltime = rtime = time;
    rspeed = speed;
    sum = lsum = 0;
    if (l != NULL) {
      l->p = this;
      ltime = l->ltime;
      lsum = min(lsum, sum + l->lsum);
      sum += l->sum + (long long)l->rspeed * (time - l->rtime);
      lsum = min(lsum, sum);
    }
    if (r != NULL) {
      r->p = this;
      rtime = r->rtime;
      rspeed = r->rspeed;
      sum += (long long)speed * (r->ltime - time);
      lsum = min(lsum, sum + r->lsum);
      sum += r->sum;
    }
  }
};
void rotate(node* v) {
  node* u = v->p;
  v->p = u->p;
  if (v->p != NULL) {
    if (v->p->l == u) {
      v->p->l = v;
    }
    if (v->p->r == u) {
      v->p->r = v;
    }
  }
  if (v == u->l) {
    u->l = v->r;
    v->r = u;
  } else {
    u->r = v->l;
    v->l = u;
  }
  u->pull();
  v->pull();
}
void splay(node* v, node* aim = NULL) {
  while (v->p != aim) {
    node* u = v->p;
    if (u->p != aim) {
      if ((u->l == v) ^ (u->p->l == u)) {
        rotate(v);
      } else {
        rotate(u);
      }
    }
    rotate(v);
  }
}
node* insert(node* v, node* u) {
  while (true) {
    if (u->time > v->time) {
      if (v->r == NULL) {
        v->r = u;
        u->p = v;
        break;
      } else {
        v = v->r;
      }
    } else {
      if (v->l == NULL) {
        v->l = u;
        u->p = v;
        break;
      } else {
        v = v->l;
      }
    }
  }
  splay(u);
  return u;
}
node* find(node* v, int time) {
  while (true) {
    if (time == v->time) {
      break;
    } else if (time > v->time) {
      v = v->r;
    } else {
      v = v->l;
    }
  }
  splay(v);
  return v;
}
node* find_less(node* v, int time) {
  node* res = NULL;
  node* from = NULL;
  while (v != NULL) {
    from = v;
    if (v->time < time) {
      res = v;
      v = v->r;
    } else {
      v = v->l;
    }
  }
  splay(from);
  splay(res);
  return res;
}
node* find_greater(node* v, int time) {
  node* res = NULL;
  node* from = NULL;
  while (v != NULL) {
    from = v;
    if (v->time > time) {
      res = v;
      v = v->l;
    } else {
      v = v->r;
    }
  }
  splay(from);
  splay(res);
  return res;
}
node* get_rightmost(node* v) {
  while (v->r != NULL) {
    v = v->r;
  }
  splay(v);
  return v;
}
node* merge(node* v, node* u) {
  v = get_rightmost(v);
  splay(u);
  v->r = u;
  v->pull();
  return v;
}
node* erase(node* v) {
  splay(v);
  v->l->p = v->r->p = NULL;
  return merge(v->l, v->r);
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  cout.setf(ios::fixed);
  cout.precision(12);
  node* lmost = new node(0, 0);
  node* rmost = new node(inf, 0);
  lmost->r = rmost;
  lmost->pull();
  node* root = lmost;
  int tt;
  cin >> tt;
  while (tt--) {
    int type;
    cin >> type;
    if (type == 1) {
      int time, speed;
      cin >> time >> speed;
      root = insert(root, new node(time, speed));
    } else if (type == 2) {
      int time;
      cin >> time;
      root = erase(find(root, time));
    } else {
      int l, r;
      long long x;
      cin >> l >> r >> x;
      if (!x) {
        cout << l << \"\n\";
        continue;
      }
      root = find_less(root, l);
      node* left = root;
      root = find_greater(root, r);
      node* right = root;
      splay(left, right);
      node* v = left->r;
      if (v == NULL) {
        cout << -1 << \"\n\";
      } else {
        if (x + v->lsum <= 0) {
          while (true) {
            if (v->l != NULL) {
              if (x + v->l->lsum <= 0) {
                v = v->l;
              } else {
                x += v->l->sum;
                if (x + (long long)v->l->rspeed * (v->time - v->l->rtime) <=
                    0) {
                  cout << v->l->rtime - (double)x / v->l->rspeed << \"\n\";
                  break;
                } else {
                  x += (long long)v->l->rspeed * (v->time - v->l->rtime);
                  if (x + (long long)v->speed * (v->r->ltime - v->time) <= 0) {
                    cout << v->time - (double)x / v->speed << \"\n\";
                    break;
                  } else {
                    x += (long long)v->speed * (v->r->ltime - v->time);
                    v = v->r;
                  }
                }
              }
            } else {
              if (x + (long long)v->speed * (v->r->ltime - v->time) <= 0) {
                cout << v->time - (double)x / v->speed << \"\n\";
                break;
              } else {
                x += (long long)v->speed * (v->r->ltime - v->time);
                v = v->r;
              }
            }
          }
          splay(v);
          root = v;
        } else if (x + v->sum + (long long)v->rspeed * (r - v->rtime) <= 0) {
          cout << v->rtime - (double)(x + v->sum) / v->rspeed << \"\n\";
        } else {
          cout << -1 << \"\n\";
        }
      }
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import re
import random
from bootcamp import Basebootcamp

class Csashaandapatientfriendbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.events = {}
        self.num_add = params.get("num_add", 5)
        self.num_delete = params.get("num_delete", 2)
        self.num_queries = params.get("num_queries", 3)
    
    def case_generator(self):
        case = {
            'events': {},
            'queries': []
        }
        # 生成添加事件
        for _ in range(self.num_add):
            t = random.randint(1, 10**9)
            while t in case['events']:
                t = random.randint(1, 10**9)
            s = random.randint(-10**9, 10**9)
            case['events'][t] = s
            case['queries'].append({'type': 1, 't': t, 's': s})
        
        # 生成删除事件
        num_delete = min(self.num_delete, len(case['events']))
        if num_delete > 0:
            deleted_ts = random.sample(list(case['events'].keys()), num_delete)
            for t in deleted_ts:
                del case['events'][t]
                case['queries'].append({'type': 2, 't': t})
        
        # 生成查询
        for _ in range(self.num_queries):
            query_type = 3  # 确保至少有一个第三种类型查询
            l = random.randint(1, 10**9)
            r = random.randint(l, 10**9)
            v = random.randint(0, 10**9)
            case['queries'].append({'type': 3, 'l': l, 'r': r, 'v': v})
        
        return case
    
    @staticmethod
    def prompt_func(question_case):
        events = question_case['events']
        queries = question_case['queries']
        problem = "Fedya and Sasha are friends. Fedya's patience is measured in liters, and when it reaches 0, the bowl bursts. Sasha can change the tap's speed through three types of queries.\n\n"
        problem += "The current events are:\n"
        for t, s in events.items():
            problem += f"Time {t}: {s} liters per second.\n"
        problem += "\n"
        problem += "The queries to process are:\n"
        for q in queries:
            if q['type'] == 1:
                problem += f"Add event: at time {q['t']}, speed {q['s']}.\n"
            elif q['type'] == 2:
                problem += f"Delete event at time {q['t']}.\n"
            else:
                problem += f"Query: from time {q['l']} to {q['r']}, initial patience {q['v']} liters. When will the bowl burst?\n"
        problem += "\n"
        problem += "For each query of type 3, output the time when the bowl bursts or -1 if it doesn't happen. Format your answer in the following way:\n"
        problem += "[answer]time[/answer]\n"
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()
    
    @classmethod
    def compute_explode_time(cls, events, l, r, v):
        sorted_events = sorted(events.keys())
        current_v = v
        current_time = l
        prev_s = 0
        
        for t in sorted_events:
            if t < l:
                continue
            if t > r:
                break
            duration = t - current_time
            if prev_s == 0:
                if current_v <= 0:
                    return current_time
                else:
                    current_v += prev_s * duration
                    current_time = t
                    prev_s = events[t]
                    continue
            
            if prev_s > 0:
                current_v += prev_s * duration
                current_time = t
                prev_s = events[t]
                continue
            else:
                delta = prev_s * duration
                if current_v + delta <= 0:
                    time_needed = (-current_v) / prev_s
                    return current_time + time_needed
                else:
                    current_v += delta
                    current_time = t
                    prev_s = events[t]
        
        duration = r - current_time
        if prev_s == 0:
            if current_v <= 0:
                return current_time
            else:
                return -1
        
        delta = prev_s * duration
        if current_v + delta <= 0:
            time_needed = (-current_v) / prev_s
            return current_time + time_needed
        else:
            return -1
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        for q in identity['queries']:
            if q['type'] == 3:
                l = q['l']
                r = q['r']
                v = q['v']
                events = identity['events']
                correct_time = cls.compute_explode_time(events, l, r, v)
                if solution is None:
                    return False
                try:
                    solution_float = float(solution)
                except ValueError:
                    return False
                if correct_time == -1:
                    return solution_float == -1
                else:
                    if abs(solution_float - correct_time) / max(1.0, abs(correct_time)) <= 1e-6:
                        return True
        return False

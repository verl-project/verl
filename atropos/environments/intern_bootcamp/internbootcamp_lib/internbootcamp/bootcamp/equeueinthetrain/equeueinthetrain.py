"""# 

### 谜题描述
There are n seats in the train's car and there is exactly one passenger occupying every seat. The seats are numbered from 1 to n from left to right. The trip is long, so each passenger will become hungry at some moment of time and will go to take boiled water for his noodles. The person at seat i (1 ≤ i ≤ n) will decide to go for boiled water at minute t_i.

Tank with a boiled water is located to the left of the 1-st seat. In case too many passengers will go for boiled water simultaneously, they will form a queue, since there can be only one passenger using the tank at each particular moment of time. Each passenger uses the tank for exactly p minutes. We assume that the time it takes passengers to go from their seat to the tank is negligibly small. 

Nobody likes to stand in a queue. So when the passenger occupying the i-th seat wants to go for a boiled water, he will first take a look on all seats from 1 to i - 1. In case at least one of those seats is empty, he assumes that those people are standing in a queue right now, so he would be better seating for the time being. However, at the very first moment he observes that all seats with numbers smaller than i are busy, he will go to the tank.

There is an unspoken rule, that in case at some moment several people can go to the tank, than only the leftmost of them (that is, seating on the seat with smallest number) will go to the tank, while all others will wait for the next moment.

Your goal is to find for each passenger, when he will receive the boiled water for his noodles.

Input

The first line contains integers n and p (1 ≤ n ≤ 100 000, 1 ≤ p ≤ 10^9) — the number of people and the amount of time one person uses the tank.

The second line contains n integers t_1, t_2, ..., t_n (0 ≤ t_i ≤ 10^9) — the moments when the corresponding passenger will go for the boiled water.

Output

Print n integers, where i-th of them is the time moment the passenger on i-th seat will receive his boiled water.

Example

Input


5 314
0 310 942 628 0


Output


314 628 1256 942 1570 

Note

Consider the example.

At the 0-th minute there were two passengers willing to go for a water, passenger 1 and 5, so the first passenger has gone first, and returned at the 314-th minute. At this moment the passenger 2 was already willing to go for the water, so the passenger 2 has gone next, and so on. In the end, 5-th passenger was last to receive the boiled water.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
const int MaxN = 100000, inf = 0x3f3f3f3f;
inline void read(int &ans) {
  ans = 0;
  char c = getchar();
  while (c < '0' || c > '9') c = getchar();
  while (c >= '0' && c <= '9') ans = ans * 10 + c - 48, c = getchar();
  return;
}
int n, P;
struct Node {
  int pos, t;
  friend inline bool operator==(const Node &a, const Node &b) {
    return a.pos == b.pos && a.t == b.t;
  }
};
struct cmp_time {
  inline bool operator()(const Node &a, const Node &b) const {
    return a.t == b.t ? a.pos > b.pos : a.t > b.t;
  }
};
struct cmp_pos {
  inline bool operator()(const Node &a, const Node &b) const {
    return a.pos > b.pos;
  }
};
template <typename Element, typename cmp>
struct Exque {
 private:
  std::priority_queue<Element, std::vector<Element>, cmp> que, del;

 public:
  inline int size() { return que.size() - del.size(); }
  inline void pop() {
    if (del.size() && que.top() == del.top()) del.pop();
    return que.pop();
  }
  inline void push(const Element &x) { return que.push(x); }
  inline void delete_(const Element &x) {
    return que.top() == x ? que.pop() : del.push(x);
  }
  inline Element top() {
    while (del.size() && que.top() == del.top()) que.pop(), del.pop();
    return que.top();
  }
};
std::priority_queue<Node, std::vector<Node>, cmp_time> sitting;
std::priority_queue<Node, std::vector<Node>, cmp_pos> preparing;
Exque<Node, cmp_pos> inque;
std::queue<Node> queorder;
long long ans[MaxN + 1];
int cnt = 0;
int main() {
  int i, t;
  long long curtime = inf;
  read(n), read(P);
  for (i = 1; i <= n; ++i) {
    read(t), sitting.push((Node){i, t});
    if (t < curtime) curtime = t;
  }
  while (sitting.size() && sitting.top().t <= curtime)
    preparing.push(sitting.top()), sitting.pop();
  if (preparing.size() &&
      (!inque.size() || preparing.top().pos < inque.top().pos))
    inque.push(preparing.top()), queorder.push(preparing.top()),
        preparing.pop();
  while (cnt < n) {
    ++cnt, ans[queorder.front().pos] = (curtime += P);
    while (sitting.size() && sitting.top().t <= curtime)
      if (sitting.top().pos < inque.top().pos)
        inque.push(sitting.top()), queorder.push(sitting.top()), sitting.pop();
      else
        preparing.push(sitting.top()), sitting.pop();
    inque.delete_(queorder.front()), queorder.pop();
    while (sitting.size() && sitting.top().t <= curtime)
      preparing.push(sitting.top()), sitting.pop();
    if (preparing.size() &&
        (!inque.size() || preparing.top().pos < inque.top().pos))
      inque.push(preparing.top()), queorder.push(preparing.top()),
          preparing.pop();
    if (!inque.size() && sitting.size()) {
      curtime = sitting.top().t;
      while (sitting.size() && sitting.top().t <= curtime)
        preparing.push(sitting.top()), sitting.pop();
      inque.push(preparing.top()), queorder.push(preparing.top()),
          preparing.pop();
    }
  }
  for (i = 1; i <= n; ++i) printf(\"%I64d \", ans[i]);
  putchar('\n');
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import heapq
from bootcamp import Basebootcamp

class Equeueinthetrainbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_t=1e9, max_p=1e9):
        self.max_n = max_n
        self.max_t = max_t
        self.max_p = max_p

    def case_generator(self):
        import random
        n = random.randint(1, self.max_n)
        p = random.randint(1, self.max_p)
        t = [random.randint(0, self.max_t) for _ in range(n)]
        return {"n": n, "p": p, "t": t}

    @staticmethod
    def prompt_func(question_case):
        n = question_case["n"]
        p = question_case["p"]
        t = question_case["t"]
        t_str = ' '.join(map(str, t))
        prompt = f"""You are a passenger on a train car trying to determine when each passenger will receive boiled water. The train car has {n} seats numbered 1 to {n}. Each passenger goes to the water tank at time t_i. If any smaller-numbered seat is empty when they check, they wait. Otherwise, they queue. Multiple queuers at the same time are processed leftmost first. Each use takes {p} minutes.

Input:
{n} {p}
{t_str}

Output the end times as space-separated integers within [answer] and [/answer]. Example: [answer]1 2 3[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers:
            return None
        try:
            last = answers[-1].strip()
            return list(map(int, last.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            correct = cls.compute_answer(identity['n'], identity['p'], identity['t'])
            return solution == correct
        except:
            return False

    @staticmethod
    def compute_answer(n, p, t_list):
        seats = list(enumerate(t_list, 1))
        seats.sort(key=lambda x: (x[1], -x[0]))
        available = []
        time = 0
        result = [0] * (n + 1)
        heap = []
        idx = 0

        while idx < n or heap:
            if not heap and idx < n:
                time = max(time, seats[idx][1])

            while idx < n and seats[idx][1] <= time:
                seat_num, t_i = seats[idx]
                heapq.heappush(heap, seat_num)
                idx += 1

            if heap:
                current = heapq.heappop(heap)
                start = max(time, t_list[current - 1])
                result[current] = start + p
                time = start + p

        return result[1:]


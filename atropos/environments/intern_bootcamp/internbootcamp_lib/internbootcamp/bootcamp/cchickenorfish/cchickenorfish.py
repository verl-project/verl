"""# 

### 谜题描述
Polycarp is flying in the airplane. Finally, it is his favorite time — the lunchtime. The BerAvia company stewardess is giving food consecutively to all the passengers from the 1-th one to the last one. Polycarp is sitting on seat m, that means, he will be the m-th person to get food.

The flight menu has k dishes in total and when Polycarp boarded the flight, he had time to count the number of portions of each dish on board. Thus, he knows values a1, a2, ..., ak, where ai is the number of portions of the i-th dish.

The stewardess has already given food to m - 1 passengers, gave Polycarp a polite smile and asked him what he would prefer. That's when Polycarp realized that they might have run out of some dishes by that moment. For some of the m - 1 passengers ahead of him, he noticed what dishes they were given. Besides, he's heard some strange mumbling from some of the m - 1 passengers ahead of him, similar to phrase 'I'm disappointed'. That happened when a passenger asked for some dish but the stewardess gave him a polite smile and said that they had run out of that dish. In that case the passenger needed to choose some other dish that was available. If Polycarp heard no more sounds from a passenger, that meant that the passenger chose his dish at the first try.

Help Polycarp to find out for each dish: whether they could have run out of the dish by the moment Polyarp was served or that dish was definitely available.

Input

Each test in this problem consists of one or more input sets. First goes a string that contains a single integer t (1 ≤ t ≤ 100 000) — the number of input data sets in the test. Then the sets follow, each set is preceded by an empty line.

The first line of each set of the input contains integers m, k (2 ≤ m ≤ 100 000, 1 ≤ k ≤ 100 000) — the number of Polycarp's seat and the number of dishes, respectively.

The second line contains a sequence of k integers a1, a2, ..., ak (1 ≤ ai ≤ 100 000), where ai is the initial number of portions of the i-th dish.

Then m - 1 lines follow, each line contains the description of Polycarp's observations about giving food to a passenger sitting in front of him: the j-th line contains a pair of integers tj, rj (0 ≤ tj ≤ k, 0 ≤ rj ≤ 1), where tj is the number of the dish that was given to the j-th passenger (or 0, if Polycarp didn't notice what dish was given to the passenger), and rj — a 1 or a 0, depending on whether the j-th passenger was or wasn't disappointed, respectively.

We know that sum ai equals at least m, that is,Polycarp will definitely get some dish, even if it is the last thing he wanted. It is guaranteed that the data is consistent.

Sum m for all input sets doesn't exceed 100 000. Sum k for all input sets doesn't exceed 100 000.

Output

For each input set print the answer as a single line. Print a string of k letters \"Y\" or \"N\". Letter \"Y\" in position i should be printed if they could have run out of the i-th dish by the time the stewardess started serving Polycarp.

Examples

Input

2

3 4
2 3 2 1
1 0
0 0

5 5
1 2 1 3 1
3 0
0 0
2 1
4 0


Output

YNNY
YYYNY

Note

In the first input set depending on the choice of the second passenger the situation could develop in different ways:

  * If he chose the first dish, then by the moment the stewardess reaches Polycarp, they will have run out of the first dish; 
  * If he chose the fourth dish, then by the moment the stewardess reaches Polycarp, they will have run out of the fourth dish; 
  * Otherwise, Polycarp will be able to choose from any of the four dishes. 



Thus, the answer is \"YNNY\".

In the second input set there is, for example, the following possible scenario. First, the first passenger takes the only third dish, then the second passenger takes the second dish. Then, the third passenger asks for the third dish, but it is not available, so he makes disappointed muttering and ends up with the second dish. Then the fourth passenger takes the fourth dish, and Polycarp ends up with the choice between the first, fourth and fifth dish.

Likewise, another possible scenario is when by the time the stewardess comes to Polycarp, they will have run out of either the first or the fifth dish (this can happen if one of these dishes is taken by the second passenger). It is easy to see that there is more than enough of the fourth dish, so Polycarp can always count on it. Thus, the answer is \"YYYNY\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

inp = [int(x) for x in sys.stdin.read().split()]; ii = 0

t = inp[ii]; ii += 1
out = []

for _ in range(t):
    m = inp[ii] - 1; ii += 1
    k = inp[ii]; ii += 1

    A = inp[ii:ii + k]; ii += k

    T = inp[ii + 0:ii + 2 * m: 2]
    R = inp[ii + 1:ii + 2 * m: 2]

    ii += 2 * m

    first_empty = m
    last_seen = [-1]*k

    for i in range(m):
        if R[i]:
            first_empty = i
            break

    for i in range(m):
        if T[i]:
            last_seen[T[i] - 1] = i
            A[T[i] - 1] -= 1

    empty = []
    if first_empty != m:
        s = 0
        for i in range(first_empty):
            if not T[i]:
                s += 1

        for j in range(k):
            if A[j] <= s and last_seen[j] < first_empty:
                empty.append(j)

        Aj = min(A[j] for j in empty)
    else:
        Aj = 0

    s = 0
    for i in range(m):
        if not T[i]:
            s += 1

    for j in range(k):
        if A[j] <= s - Aj:
            empty.append(j)

    empty = set(empty)
    out.append(''.join('Y' if j in empty else 'N' for j in range(k)))

print '\n'.join(out)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cchickenorfishbootcamp(Basebootcamp):
    def __init__(self, max_k=5, max_m=10, max_a=10):
        self.max_k = max_k
        self.max_m = max_m
        self.max_a = max_a
    
    def case_generator(self):
        while True:
            k = random.randint(1, self.max_k)
            m_original = random.randint(2, self.max_m)
            a = [random.randint(1, self.max_a) for _ in range(k)]
            if sum(a) >= m_original:
                break
        
        current_a = a.copy()
        passengers = []
        
        for _ in range(m_original - 1):
            available_dishes = [i+1 for i in range(k) if current_a[i] > 0]
            empty_dishes = [i+1 for i in range(k) if current_a[i] == 0]
            
            # Generate disappointed passenger scenario
            if empty_dishes and random.random() < 0.3:
                # Passenger requested an empty dish
                requested_dish = random.choice(empty_dishes)
                actual_dish = random.choice(available_dishes)
                current_a[actual_dish-1] -= 1
                passengers.append({
                    't': random.choice([0, actual_dish]),
                    'r': 1
                })
            else:
                # Normal selection
                selected_dish = random.choice(available_dishes)
                current_a[selected_dish-1] -= 1
                passengers.append({
                    't': random.choice([0, selected_dish]),
                    'r': 0
                })
        
        return {
            'm': m_original,
            'k': k,
            'a': a,
            'passengers': passengers
        }

    @staticmethod
    def prompt_func(question_case):
        m = question_case['m']
        k = question_case['k']
        a = question_case['a']
        passengers = question_case['passengers']
        
        prompt = f"""Polycarp is the {m}th passenger to be served on a flight with {k} dishes.
Initial portions: {' '.join(map(str, a))}

Observations:"""
        for i, p in enumerate(passengers, 1):
            notice = f"dish {p['t']}" if p['t'] else "unknown dish"
            prompt += f"\nPassenger {i}: {notice} | {'Disappointed' if p['r'] else 'Satisfied'}"
            
        prompt += "\n\nDetermine which dishes COULD HAVE run out (Y/N):\n[answer]...[/answer]"
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # Implementation of reference algorithm
        m = identity['m'] - 1
        k = identity['k']
        a = identity['a'].copy()
        passengers = identity['passengers']
        
        T = [p['t'] for p in passengers]
        R = [p['r'] for p in passengers]
        
        # Find first disappointed passenger
        first_r = next((i for i, r in enumerate(R) if r == 1), m)
        
        # Track portions and last access
        last_seen = [-1] * k
        for i in range(m):
            if T[i]:
                dish = T[i] - 1
                last_seen[dish] = i
                a[dish] -= 1
        
        # Calculate possible exhausted dishes
        s = sum(1 for i in range(first_r) if T[i] == 0)
        candidates = set()
        
        if first_r < m:
            # Phase 1: Before first disappointment
            for j in range(k):
                if a[j] <= s and last_seen[j] < first_r:
                    candidates.add(j)
            
            # Phase 2: Remaining selections
            total_s = sum(1 for t in T if t == 0)
            min_a = min((a[j] for j in candidates), default=0)
            for j in range(k):
                if a[j] <= (total_s - min_a):
                    candidates.add(j)
        else:
            # No disappointed passengers
            total_s = sum(1 for t in T if t == 0)
            for j in range(k):
                if a[j] <= total_s:
                    candidates.add(j)
        
        # Generate expected answer
        expected = ['Y' if i in candidates else 'N' for i in range(k)]
        return solution == ''.join(expected)

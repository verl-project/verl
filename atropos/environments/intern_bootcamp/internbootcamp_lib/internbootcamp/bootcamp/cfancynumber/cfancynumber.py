"""# 

### 谜题描述
A car number in Berland consists of exactly n digits. A number is called beautiful if it has at least k equal digits. Vasya wants to change the digits in his car's number so that the number became beautiful. To replace one of n digits Vasya has to pay the sum of money, equal to the absolute difference between the old digit and the new one.

Help Vasya: find the minimum sum of money he should pay to make the number of his car beautiful. You should also find the resulting beautiful number. If there are several such numbers, then print the lexicographically minimum one.

Input

The first line contains two space-separated integers n and k (2 ≤ n ≤ 104, 2 ≤ k ≤ n) which represent how many digits the number has and how many equal digits a beautiful number should have. The second line consists of n digits. It describes the old number of Vasya's car. It is guaranteed that the number contains no spaces and only contains digits.

Output

On the first line print the minimum sum of money Vasya needs to change the number. On the second line print the car's new number. If there are several solutions, print the lexicographically minimum one.

Examples

Input

6 5
898196


Output

4
888188


Input

3 2
533


Output

0
533


Input

10 6
0001112223


Output

3
0000002223

Note

In the first sample replacing the second digit with an \"8\" costs |9 - 8| = 1. Replacing the fifth digit with an \"8\" costs the same. Replacing the sixth digit costs |6 - 8| = 2. As a result, Vasya will pay 1 + 1 + 2 = 4 for a beautiful number \"888188\".

The lexicographical comparison of strings is performed by the < operator in modern programming languages. The string x is lexicographically smaller than the string y, if there exists such i (1 ≤ i ≤ n), that xi < yi, and for any j (1 ≤ j < i) xj = yj. The strings compared in this problem will always have the length n.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s=(raw_input()).split()
n=int(s[0])
k=int(s[1])
s=raw_input()
s=[ch for ch in s]
c=10*[0]
for i in range(len(s)):
      j=int(s[i])
      c[j]=c[j]+1
from string import join
def choosevalue(m):
      global n,k,s,c
      sol=[0,'']
      if (c[m]>=k):
            sol[1]=s
            return sol
      p=s[:]
      remain=k-c[m]
      for i in range(1,10,1):
            L=m-i
            R=m+i
            if (((L<0) and (R>9)) or (remain==0)):
                  sol[1]=p
                  return sol
            if ((R<10)and (remain>0)):
                R=str(R)
                for j in range(len(p)):
                      if (p[j]==R):
                              p[j]=str(m)
                              sol[0]=sol[0]+i
                              remain=remain-1
                              if (remain==0):
                                    sol[1]=p
                                    return sol
            if (L>=0):
                  L=str(L)
                  for j in range(len(p)-1,-1,-1):
                        if (p[j]==L):
                              p[j]=str(m)
                              sol[0]=sol[0]+i
                              remain=remain-1
                              if (remain==0):
                                    sol[1]=p
                                    return sol
      sol[1]=p
      return sol
solution=choosevalue(0)
for x in range(1,10,1):
      test=choosevalue(x)
      if (test[0]<solution[0]):
            solution=test
      else:
            if (test[0]==solution[0]):
                  for j in range(n):
                        if (int(test[1][j])<int(solution[1][j])):
                              solution=test
                              break
                        else:
                              if (int(test[1][j])>int(solution[1][j])):
                                    break

print solution[0]
print \"\".join(solution[1])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def solve_beautiful_number(n, k, original_number):
    s = list(original_number)
    c = [0] * 10
    for i in range(n):
        digit = int(s[i])
        c[digit] += 1

    def choosevalue(m):
        nonlocal c, n, k, s
        if c[m] >= k:
            return (0, original_number)
        p = s.copy()
        total_cost = 0
        remain = k - c[m]
        for i in range(1, 10):
            R = m + i
            L = m - i
            # Process R direction (higher digits)
            if R <= 9 and remain > 0:
                for j in range(n):
                    if remain <= 0:
                        break
                    if int(p[j]) == R:
                        p[j] = str(m)
                        total_cost += i
                        remain -= 1
            # Process L direction (lower digits)
            if L >= 0 and remain > 0:
                for j in range(n-1, -1, -1):
                    if remain <= 0:
                        break
                    if int(p[j]) == L:
                        p[j] = str(m)
                        total_cost += i
                        remain -= 1
            if remain <= 0:
                break
        new_number = ''.join(p)
        return (total_cost, new_number)

    best_cost = float('inf')
    best_number = None
    for m in range(10):
        current_cost, current_number = choosevalue(m)
        if current_cost < best_cost:
            best_cost = current_cost
            best_number = current_number
        elif current_cost == best_cost:
            if current_number < best_number:
                best_number = current_number
    return (best_cost, best_number)

class Cfancynumberbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, fixed_n=None, fixed_k=None):
        self.min_n = min_n
        self.max_n = max_n
        self.fixed_n = fixed_n
        self.fixed_k = fixed_k

    def case_generator(self):
        if self.fixed_n is not None:
            n = self.fixed_n
        else:
            n = random.randint(self.min_n, self.max_n)
        
        if self.fixed_k is not None:
            k = self.fixed_k
        else:
            k = random.randint(2, n)
        
        original_number = ''.join(random.choices('0123456789', k=n))
        return {
            'n': n,
            'k': k,
            'original_number': original_number
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        original = question_case['original_number']
        prompt = f"""你是一位车牌号码优化师。请帮助Vasya将他的车牌号码转换为美丽号，使得转换的总花费最小，并且在多个解中选择字典序最小的。具体规则如下：

- 车牌号码由{n}位数字组成。
- 美丽号要求至少有{k}个相同的数字。
- 每次更改一个数字的花费是原数字和新数字的绝对差。
- 总花费是所有更改的花费之和。
- 如果有多个总花费相同的最优解，必须选择字典序最小的那个。

当前的问题实例是：

输入：
{n} {k}
{original}

请输出两行，第一行是最小的总花费，第二行是新的号码。将你的最终答案放置在[answer]和[/answer]标签之间。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        return (lines[0], lines[1])

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_cost_str, user_number = solution
            user_cost = int(user_cost_str)
        except:
            return False
        
        n = identity['n']
        if len(user_number) != n:
            return False
        
        correct_cost, correct_number = solve_beautiful_number(
            identity['n'],
            identity['k'],
            identity['original_number']
        )
        
        return user_cost == correct_cost and user_number == correct_number

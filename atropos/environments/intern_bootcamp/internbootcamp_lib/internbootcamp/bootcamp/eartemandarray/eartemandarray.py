"""# 

### 谜题描述
Artem has an array of n positive integers. Artem decided to play with it. The game consists of n moves. Each move goes like this. Artem chooses some element of the array and removes it. For that, he gets min(a, b) points, where a and b are numbers that were adjacent with the removed number. If the number doesn't have an adjacent number to the left or right, Artem doesn't get any points. 

After the element is removed, the two parts of the array glue together resulting in the new array that Artem continues playing with. Borya wondered what maximum total number of points Artem can get as he plays this game.

Input

The first line contains a single integer n (1 ≤ n ≤ 5·105) — the number of elements in the array. The next line contains n integers ai (1 ≤ ai ≤ 106) — the values of the array elements.

Output

In a single line print a single integer — the maximum number of points Artem can get.

Examples

Input

5
3 1 5 2 6


Output

11


Input

5
1 2 3 4 5


Output

6


Input

5
1 100 101 100 1


Output

102

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

n=int(sys.stdin.readline().strip('\n'))
line=sys.stdin.readline().strip('\n').split(' ')
values=map(lambda x:int(x),line)

def evaluateBothSides(start,stop):
    print start,stop
    if stop-start==1:
        return min(values[start-1],values[stop])
    elif stop-start==0:
        return 0
    else:    
        maxNum=0
        maxPos=0
        for i in xrange(start,stop):
            if values[i]>maxNum:
                maxPos=i
                maxNum=values[i]
        return evaluateBothSides(start,maxPos)+evaluateBothSides(maxPos+1,stop)+min(values[start-1],values[stop])

def evaluateMax(stop):
    2+2

def evaluate(start,stop):
    2+2
    

def addToQueue(inp,queue):
    out=0
    if inp<=queue[-1]:
        queue.append(inp)
    else:
        queue.pop()
        out+=inp
        while inp>queue[-1]:
            out+=queue.pop()
        queue.append(inp)
    return out

def queueEvaluate():
    maxLeft=0
    maxRight=0
    maxPos=0
    maxNum=0
    for i in xrange(0,n):
            if values[i]>maxNum:
                maxPos=i
                maxNum=values[i]
    out=0
    queue=[maxNum]
    for i in xrange(maxPos+1,n):
        out+=addToQueue(values[i],queue)
        #print queue
    if len(queue)==1:
        pass
    else:
        maxRight=queue[1]
        for i in xrange(2,len(queue)):
            out+=queue[i]
    queue=[maxNum]
    for i in xrange(maxPos-1,-1,-1):
        out+=addToQueue(values[i],queue)
        #print queue
    if len(queue)==1:
        pass
    else:
        maxLeft=queue[1]
        for i in xrange(2,len(queue)):
            out+=queue[i]
    out+=min(maxLeft,maxRight)
    return out

print queueEvaluate()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import heapq
from bootcamp import Basebootcamp

class Eartemandarraybootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, min_val=1, max_val=100):
        self.min_n = min_n
        self.max_n = max_n
        self.min_val = min_val
        self.max_val = max_val
    
    def case_generator(self):
        import random
        n = random.randint(self.min_n, self.max_n)
        arr = [random.randint(self.min_val, self.max_val) for _ in range(n)]
        expected_score = self.calculate_max_score(arr)
        return {
            'array': arr,
            'expected_score': expected_score
        }
    
    @staticmethod
    def prompt_func(question_case):
        arr = question_case['array']
        n = len(arr)
        prompt = (
            "Artem has an array of n positive integers. He plays a game where he removes elements one by one. "
            "For each removal, he gains points equal to the minimum of its adjacent elements. If an element has "
            "no left or right neighbor, that side contributes nothing. Your task is to determine the maximum total "
            "points Artem can achieve.\n\n"
            "Input Format:\n- The first line contains an integer n (array size).\n- The second line contains n space-separated integers.\n\n"
            "Output Format:\n- A single integer representing the maximum points.\n\n"
            "Example:\nInput:\n5\n3 1 5 2 6\nOutput:\n11\n\n"
            "Solve the following problem:\nInput:\n"
            f"{n}\n{' '.join(map(str, arr))}\n\n"
            "Put your final answer within [answer] tags, like [answer]11[/answer]."
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_score']
    
    @staticmethod
    def calculate_max_score(arr):
        n = len(arr)
        if n <= 1:
            return 0

        left = [i - 1 for i in range(n)]
        right = [i + 1 for i in range(n)]
        valid = [True] * n
        contribution = []

        for i in range(n):
            l, r = left[i], right[i]
            score = min(arr[l], arr[r]) if l >= 0 and r < n else 0
            contribution.append(score)

        heap = [(-score, i) for i, score in enumerate(contribution)]
        heapq.heapify(heap)

        total = 0
        while heap:
            current_neg, i = heapq.heappop(heap)
            current_score = -current_neg

            if not valid[i] or contribution[i] != current_score:
                continue

            total += current_score
            valid[i] = False

            l, r = left[i], right[i]
            if l >= 0: right[l] = r
            if r < n: left[r] = l

            # 更新左邻居
            if l >= 0 and valid[l]:
                new_l, new_r = left[l], right[l]
                new_score = min(arr[new_l], arr[new_r]) if new_l >=0 and new_r < n else 0
                contribution[l] = new_score
                heapq.heappush(heap, (-new_score, l))

            # 更新右邻居
            if r < n and valid[r]:
                new_l, new_r = left[r], right[r]
                new_score = min(arr[new_l], arr[new_r]) if new_l >=0 and new_r < n else 0
                contribution[r] = new_score
                heapq.heappush(heap, (-new_score, r))

        return total

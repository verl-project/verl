"""# 

### 谜题描述
On Bertown's main street n trees are growing, the tree number i has the height of ai meters (1 ≤ i ≤ n). By the arrival of the President of Berland these trees were decided to be changed so that their heights formed a beautiful sequence. This means that the heights of trees on ends (the 1st one and the n-th one) should be equal to each other, the heights of the 2-nd and the (n - 1)-th tree must also be equal to each other, at that the height of the 2-nd tree should be larger than the height of the first tree by 1, and so on. In other words, the heights of the trees, standing at equal distance from the edge (of one end of the sequence) must be equal to each other, and with the increasing of the distance from the edge by 1 the tree height must also increase by 1. For example, the sequences \"2 3 4 5 5 4 3 2\" and \"1 2 3 2 1\" are beautiful, and '1 3 3 1\" and \"1 2 3 1\" are not. 

Changing the height of a tree is a very expensive operation, using advanced technologies invented by Berland scientists. In one operation you can choose any tree and change its height to any number, either increase or decrease. Note that even after the change the height should remain a positive integer, i. e, it can't be less than or equal to zero. Identify the smallest number of changes of the trees' height needed for the sequence of their heights to become beautiful.

Input

The first line contains integer n (1 ≤ n ≤ 105) which is the number of trees. The second line contains integers ai (1 ≤ ai ≤ 105) which are the heights of the trees.

Output

Print a single number which is the minimal number of trees whose heights will have to be changed for the sequence to become beautiful.

Examples

Input

3
2 2 2


Output

1


Input

4
1 2 2 1


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
if __name__ == '__main__':
    n = int(raw_input().rstrip())
    heights = [int(x) for x in raw_input().rstrip().split()]
    
    dic = {}
    for i in range(0, n):
        heights[i] -= min(i, n - 1 - i)
        key = heights[i]
        if key == 0:
            continue
        if key not in dic:
            dic[key] = 1
        else:
            dic[key] += 1
    #print dic

    maxKey = None
    maxValue = 0
    for key in dic.keys():
        if maxValue < dic[key]:
            maxKey = key
            maxValue = dic[key]
    #print maxKey, maxValue

    print n - maxValue
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ctreesbootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_height=100):
        self.max_n = max_n
        self.max_height = max_height
    
    def case_generator(self):
        # 增强案例生成逻辑，增加边界条件覆盖
        n = random.choice([
            1, 2, 3, 
            self.max_n, 
            random.randint(4, self.max_n)
        ]) if self.max_n > 4 else random.randint(1, self.max_n)
        
        # 随机生成时需要确保至少存在一个合法解
        while True:
            heights = [random.randint(1, self.max_height) for _ in range(n)]
            valid_heights = [
                h - min(i, n-1-i)
                for i, h in enumerate(heights)
                if h > min(i, n-1-i)
            ]
            if valid_heights or n == 0:
                break
                
        correct_answer = self._calculate_correct_answer(n, heights)
        return {
            'n': n,
            'heights': heights,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def _calculate_correct_answer(n, heights):
        m = [min(i, n-1 -i) for i in range(n)]
        adjusted = [h - mi for h, mi in zip(heights, m)]
        counter = {}
        for val in adjusted:
            if val > 0:
                counter[val] = counter.get(val, 0) + 1
        
        max_freq = max(counter.values()) if counter else 0
        return n - max_freq
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""As a city planner in Bertown, you need to adjust {question_case['n']} trees with heights {question_case['heights']} to form a beautiful sequence. The sequence must satisfy:
1. Symmetric pairs (k-th from start and end must be equal)
2. Each inward pair increases by exactly 1 meter
3. All heights remain positive (>0)

Calculate the minimal trees to modify. Format your answer as: [answer]number[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        try:
            return int(matches[-1]) if matches else None
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

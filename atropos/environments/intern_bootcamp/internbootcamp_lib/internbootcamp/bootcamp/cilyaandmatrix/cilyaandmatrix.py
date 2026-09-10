"""# 

### 谜题描述
Ilya is a very good-natured lion. He likes maths. Of all mathematical objects, his favourite one is matrices. Now he's faced a complicated matrix problem he needs to solve.

He's got a square 2n × 2n-sized matrix and 4n integers. You need to arrange all these numbers in the matrix (put each number in a single individual cell) so that the beauty of the resulting matrix with numbers is maximum.

The beauty of a 2n × 2n-sized matrix is an integer, obtained by the following algorithm:

  1. Find the maximum element in the matrix. Let's denote it as m. 
  2. If n = 0, then the beauty of the matrix equals m. Otherwise, a matrix can be split into 4 non-intersecting 2n - 1 × 2n - 1-sized submatrices, then the beauty of the matrix equals the sum of number m and other four beauties of the described submatrices. 



As you can see, the algorithm is recursive.

Help Ilya, solve the problem and print the resulting maximum beauty of the matrix.

Input

The first line contains integer 4n (1 ≤ 4n ≤ 2·106). The next line contains 4n integers ai (1 ≤ ai ≤ 109) — the numbers you need to arrange in the 2n × 2n-sized matrix.

Output

On a single line print the maximum value of the beauty of the described matrix.

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

1
13


Output

13


Input

4
1 2 3 4


Output

14

Note

Consider the second sample. You need to arrange the numbers in the matrix as follows:
    
    
      
    1 2  
    3 4  
    

Then the beauty of the matrix will equal: 4 + 1 + 2 + 3 + 4 = 14.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin

def solver():
    stdin.readline()
    num = map(float, stdin.readline().split())

    num.sort(reverse=True)
    ans = 0

    while num:
        ans += sum(num)
        num = num[:len(num)/4]
        
    print int(ans)
    

if __name__ == \"__main__\":
   
    solver()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cilyaandmatrixbootcamp(Basebootcamp):
    def __init__(self, m=1):
        super().__init__()  # 显式调用父类初始化
        if not isinstance(m, int) or m < 0:
            raise ValueError("m must be non-negative integer")
        self.m = m
        self.size = 4 ** m  # 保证生成的数字数量为4的整数次幂
    
    def case_generator(self):
        # 严格保证数字数量为4的m次幂（兼容m=0的情况）
        size = max(1, 4 ** self.m)  # 处理m=0时4^0=1
        numbers = [random.randint(1, 10**9) for _ in range(size)]
        sorted_nums = sorted(numbers, reverse=True)
        
        ans = 0
        current = sorted_nums.copy()
        while current:
            ans += sum(current)
            current = current[:len(current)//4]
            
        return {
            'input_numbers': numbers,
            'correct_answer': ans,
            'size': size,
            'm': self.m  # 保存参数确保验证时可追溯
        }
    
    @staticmethod
    def prompt_func(question_case):
        numbers = question_case['input_numbers']
        numbers_str = ' '.join(map(str, numbers))
        example = "\nExample workflow:" + (
            "\nFor input [1,2,3,4]: sorted=[4,3,2,1]"
            "\n1st sum: 4+3+2+1=10 → keep first 1 element (4)"
            "\n2nd sum: 4 → total beauty=10+4=14"
        ) if question_case['m'] == 1 else ""
        
        return f"""Given {question_case['size']} numbers: {numbers_str}
Arrange them in a 2n×2n matrix to maximize recursive beauty. The optimal solution requires:
1. Sort numbers in descending order
2. Sum all numbers (1st layer sum)
3. Keep first quarter of numbers
4. Repeat steps 2-3 until empty

{example}

Format your answer as [answer]{{value}}[/answer]. What is the maximum beauty?"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if re.fullmatch(r'\d+', last_match):  # 严格匹配整数格式
            return int(last_match)
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['correct_answer']
        except (ValueError, TypeError):
            return False

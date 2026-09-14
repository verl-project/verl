"""# 

### 谜题描述
Many students live in a dormitory. A dormitory is a whole new world of funny amusements and possibilities but it does have its drawbacks. 

There is only one shower and there are multiple students who wish to have a shower in the morning. That's why every morning there is a line of five people in front of the dormitory shower door. As soon as the shower opens, the first person from the line enters the shower. After a while the first person leaves the shower and the next person enters the shower. The process continues until everybody in the line has a shower.

Having a shower takes some time, so the students in the line talk as they wait. At each moment of time the students talk in pairs: the (2i - 1)-th man in the line (for the current moment) talks with the (2i)-th one. 

Let's look at this process in more detail. Let's number the people from 1 to 5. Let's assume that the line initially looks as 23154 (person number 2 stands at the beginning of the line). Then, before the shower opens, 2 talks with 3, 1 talks with 5, 4 doesn't talk with anyone. Then 2 enters the shower. While 2 has a shower, 3 and 1 talk, 5 and 4 talk too. Then, 3 enters the shower. While 3 has a shower, 1 and 5 talk, 4 doesn't talk to anyone. Then 1 enters the shower and while he is there, 5 and 4 talk. Then 5 enters the shower, and then 4 enters the shower.

We know that if students i and j talk, then the i-th student's happiness increases by gij and the j-th student's happiness increases by gji. Your task is to find such initial order of students in the line that the total happiness of all students will be maximum in the end. Please note that some pair of students may have a talk several times. In the example above students 1 and 5 talk while they wait for the shower to open and while 3 has a shower.

Input

The input consists of five lines, each line contains five space-separated integers: the j-th number in the i-th line shows gij (0 ≤ gij ≤ 105). It is guaranteed that gii = 0 for all i.

Assume that the students are numbered from 1 to 5.

Output

Print a single integer — the maximum possible total happiness of the students.

Examples

Input

0 0 0 0 9
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
7 0 0 0 0


Output

32


Input

0 43 21 18 2
3 0 21 11 65
5 2 0 1 4
54 62 12 0 99
87 64 81 33 0


Output

620

Note

In the first sample, the optimal arrangement of the line is 23154. In this case, the total happiness equals:

(g23 + g32 + g15 + g51) + (g13 + g31 + g54 + g45) + (g15 + g51) + (g54 + g45) = 32.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from itertools import permutations

def calc(mat, order):
	one = mat[ order[0]-1 ][ order[1]-1 ] + mat[ order[2]-1 ][ order[3]-1 ] + \
		  mat[ order[1]-1 ][ order[0]-1 ] + mat[ order[3]-1 ][ order[2]-1 ]

	two = mat[ order[1]-1 ][ order[2]-1 ] + mat[ order[3]-1 ][ order[4]-1 ] + \
		  mat[ order[2]-1 ][ order[1]-1 ] + mat[ order[4]-1 ][ order[3]-1 ]	

	three = mat[ order[2]-1 ][ order[3]-1 ] + mat[ order[3]-1 ][ order[2]-1 ]
	four  = mat[ order[3]-1 ][ order[4]-1 ] + mat[ order[4]-1 ][ order[3]-1 ]

	return (one+two+three+four)

def solve(mat):
	retval = 0
	for x in permutations([1,2,3,4,5], r=5):
		order = list(x)
		retval = max(retval, calc(mat, order))
	print retval

def main():
	mat = []
	for _ in xrange(5):
		line = raw_input().split()
		line = map(int, line)
		mat.append(line)
	solve(mat)

if __name__ == '__main__':
	main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
from itertools import permutations

class Bshowerlinebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_value = params.get('max_value', 10**5)
        self.size = 5
    
    def case_generator(self):
        matrix = []
        for i in range(self.size):
            row = [0]*self.size
            for j in range(self.size):
                if i != j:
                    row[j] = random.randint(0, self.max_value)
            matrix.append(row)
        return {'matrix': matrix}
    
    @staticmethod
    def prompt_func(question_case):
        matrix = question_case['matrix']
        matrix_lines = [' '.join(map(str, row)) for row in matrix]
        matrix_str = '\n'.join(matrix_lines)
        prompt = f"""许多学生住在宿舍里，早晨只有一个淋浴间可用，五个学生排成一队使用。每当有人在淋浴时，剩余队伍中的学生会成对交谈。具体规则如下：

- 初始队列确定后，在淋浴开始前，前两人配对话，中间两人配对，最后一人单排
- 每当有学生完成淋浴后，队伍前移并重新按照上述规则配对
- 每次交谈双方i和j会分别增加g_ij和g_ji的幸福度
- 同一对可能在多个阶段多次交谈

给定5x5的矩阵g_ij（对角线为0），找到使总幸福度最大的初始队列顺序，输出最大值。

输入矩阵：
{matrix_str}

请将最终答案放在[answer]标签内，例如：[answer]620[/answer]。确保只输出数值。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip().split()[-1]  # 处理可能存在的附加文本
        try:
            return int(last_match)
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        matrix = identity['matrix']
        max_total = 0
        
        # 预先转换为0-based索引矩阵（优化性能）
        mat = [[matrix[i][j] for j in range(5)] for i in range(5)]
        
        def calculate(order):
            total = 0
            # Phase 1: 0-1, 2-3 交谈
            total += mat[order[0]][order[1]] + mat[order[1]][order[0]]
            total += mat[order[2]][order[3]] + mat[order[3]][order[2]]
            
            # Phase 2: 1-2,3-4 交谈
            total += mat[order[1]][order[2]] + mat[order[2]][order[1]]
            total += mat[order[3]][order[4]] + mat[order[4]][order[3]]
            
            # Phase 3: 2-3 交谈
            total += mat[order[2]][order[3]] + mat[order[3]][order[2]]
            
            # Phase 4: 3-4 交谈
            total += mat[order[3]][order[4]] + mat[order[4]][order[3]]
            
            return total
        
        for perm in permutations(range(5)):  # 使用0-based排列优化计算
            current_total = calculate(perm)
            if current_total > max_total:
                max_total = current_total
                
        return solution == max_total

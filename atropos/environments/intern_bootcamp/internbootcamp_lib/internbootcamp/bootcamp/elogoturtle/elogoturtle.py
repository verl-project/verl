"""# 

### 谜题描述
A lot of people associate Logo programming language with turtle graphics. In this case the turtle moves along the straight line and accepts commands \"T\" (\"turn around\") and \"F\" (\"move 1 unit forward\").

You are given a list of commands that will be given to the turtle. You have to change exactly n commands from the list (one command can be changed several times). How far from the starting point can the turtle move after it follows all the commands of the modified list?

Input

The first line of input contains a string commands — the original list of commands. The string commands contains between 1 and 100 characters, inclusive, and contains only characters \"T\" and \"F\".

The second line contains an integer n (1 ≤ n ≤ 50) — the number of commands you have to change in the list.

Output

Output the maximum distance from the starting point to the ending point of the turtle's path. The ending point of the turtle's path is turtle's coordinate after it follows all the commands of the modified list.

Examples

Input

FT
1


Output

2


Input

FFFTFFF
2


Output

6

Note

In the first example the best option is to change the second command (\"T\") to \"F\" — this way the turtle will cover a distance of 2 units.

In the second example you have to change two commands. One of the ways to cover maximal distance of 6 units is to change the fourth command and first or last one.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
line = raw_input()
n = input()
m = len(line)
inc = [1,-1]
dis = [ [ ['@' for x in range(2)] for y in range(m+1) ] for z in range(n+2) ]
dis[0][0][0] = 0
dis[0][0][1] = 0
for i in range(n+1):
    for j in range(m):
        for k in range(2):
            if dis[i][j][k] != '@':
                new_dis = dis[i][j][k] + inc[k]
                if line[j] == 'F':
                    if dis[i][j+1][k]=='@' or dis[i][j+1][k] < new_dis:
                        dis[i][j+1][k] = new_dis
                    if dis[i+1][j+1][k^1] == '@' or dis[i+1][j+1][k^1] < dis[i][j][k]:
                        dis[i+1][j+1][k^1] = dis[i][j][k]
                elif line[j] == 'T':
                    if dis[i][j+1][k^1]=='@' or dis[i][j+1][k^1] < dis[i][j][k]:
                        dis[i][j+1][k^1] = dis[i][j][k]
                    if dis[i+1][j+1][k]=='@' or dis[i+1][j+1][k] < new_dis:
                        dis[i+1][j+1][k] = new_dis

ans = 0
for i in range(n,-1,-2):
    if dis[i][m][0] != '@':
        ans = max(ans,dis[i][m][0])
    if dis[i][m][1] != '@':
        ans = max(ans,dis[i][m][1])
print ans 

#print( max( [max(dis[i][m]) for i in range(n,-1,-2)] ) )

#from pprint import pprint
#pprint(dis)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def calculate_max_distance(commands: str, n: int) -> int:
    m = len(commands)
    inc = [1, -1]
    max_changes = n
    
    # Initialize DP table with -infinity
    dis = [[[-float('inf') for _ in range(2)] for __ in range(m + 1)] for ___ in range(max_changes + 2)]
    dis[0][0][0] = 0  # Correct initialization: facing direction 0 initially

    for i in range(max_changes + 1):
        for j in range(m):
            for k in range(2):
                if dis[i][j][k] == -float('inf'):
                    continue
                current_val = dis[i][j][k]
                cmd = commands[j]
                
                # Process without modification
                if cmd == 'F':
                    # Move forward
                    new_val = current_val + inc[k]
                    if new_val > dis[i][j+1][k]:
                        dis[i][j+1][k] = new_val
                else:
                    # Turn around
                    new_k = 1 - k
                    if current_val > dis[i][j+1][new_k]:
                        dis[i][j+1][new_k] = current_val
                
                # Process with modification (if within change limit)
                if i >= max_changes:
                    continue
                if cmd == 'F':
                    # Change F to T: turn without moving
                    new_k_m = 1 - k
                    if current_val > dis[i+1][j+1][new_k_m]:
                        dis[i+1][j+1][new_k_m] = current_val
                else:
                    # Change T to F: move forward in current direction
                    new_val_m = current_val + inc[k]
                    if new_val_m > dis[i+1][j+1][k]:
                        dis[i+1][j+1][k] = new_val_m

    # Find maximum valid distance considering parity of remaining changes
    max_distance = 0
    for used_changes in range(max_changes + 1):
        remaining_changes = max_changes - used_changes
        # Allow remaining changes to be even (can flip commands even times)
        if remaining_changes % 2 != 0:
            continue
        for direction in [0, 1]:
            val = dis[used_changes][m][direction]
            if val != -float('inf'):
                max_distance = max(max_distance, abs(val))
    return max_distance

class Elogoturtlebootcamp(Basebootcamp):
    def __init__(self, **params):
        params.setdefault('commands_length_min', 1)
        params.setdefault('commands_length_max', 100)
        params.setdefault('n_min', 1)
        params.setdefault('n_max', 50)
        params.setdefault('f_prob', 0.5)
        
        self.commands_length_min = params['commands_length_min']
        self.commands_length_max = params['commands_length_max']
        self.n_min = params['n_min']
        self.n_max = params['n_max']
        self.f_prob = params['f_prob']
    
    def case_generator(self):
        length = random.randint(self.commands_length_min, self.commands_length_max)
        commands = ''.join(random.choices(['F', 'T'], 
                          weights=[self.f_prob, 1-self.f_prob], k=length))
        n = random.randint(self.n_min, self.n_max)
        # Ensure n <= 50 as per problem constraints
        n = min(n, 50)
        max_distance = calculate_max_distance(commands, n)
        return {
            'commands': commands,
            'n': n,
            'max_distance': max_distance
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        commands = question_case['commands']
        n = question_case['n']
        return f"""You are controlling a turtle that follows these commands: 
- F: Move 1 unit forward 
- T: Turn 180 degrees (reverse direction)

**Task**: Modify exactly {n} command(s) in the original sequence to maximize the turtle's final distance from the starting point.

**Original command sequence**: {commands}

**Rules**:
1. You MUST change exactly {n} commands (each change can modify any command, multiple changes allowed on the same command)
2. After modifications, the turtle executes the entire command sequence
3. Distance is the absolute value of the final position

**Output format**: 
Provide the maximum possible distance as an integer inside [answer] tags. For example: [answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['max_distance']

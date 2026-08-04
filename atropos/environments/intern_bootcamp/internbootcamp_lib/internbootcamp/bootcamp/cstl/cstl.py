"""# 

### 谜题描述
Vasya used to be an accountant before the war began and he is one of the few who knows how to operate a computer, so he was assigned as the programmer.

We all know that programs often store sets of integers. For example, if we have a problem about a weighted directed graph, its edge can be represented by three integers: the number of the starting vertex, the number of the final vertex and the edge's weight. So, as Vasya was trying to represent characteristics of a recently invented robot in his program, he faced the following problem.

Vasya is not a programmer, so he asked his friend Gena, what the convenient way to store n integers is. Gena used to code in language X-- and so he can use only the types that occur in this language. Let's define, what a \"type\" is in language X--:

  * First, a type is a string \"int\". 
  * Second, a type is a string that starts with \"pair\", then followed by angle brackets listing exactly two comma-separated other types of language X--. This record contains no spaces. 
  * No other strings can be regarded as types. 



More formally: type := int | pair<type,type>. For example, Gena uses the following type for graph edges: pair<int,pair<int,int>>.

Gena was pleased to help Vasya, he dictated to Vasya a type of language X--, that stores n integers. Unfortunately, Gena was in a hurry, so he omitted the punctuation. Now Gena has already left and Vasya can't find the correct punctuation, resulting in a type of language X--, however hard he tries.

Help Vasya and add the punctuation marks so as to receive the valid type of language X--. Otherwise say that the task is impossible to perform.

Input

The first line contains a single integer n (1 ≤ n ≤ 105), showing how many numbers the type dictated by Gena contains.

The second line contains space-separated words, said by Gena. Each of them is either \"pair\" or \"int\" (without the quotes).

It is guaranteed that the total number of words does not exceed 105 and that among all the words that Gena said, there are exactly n words \"int\".

Output

If it is possible to add the punctuation marks so as to get a correct type of language X-- as a result, print a single line that represents the resulting type. Otherwise, print \"Error occurred\" (without the quotes). Inside the record of a type should not be any extra spaces and other characters. 

It is guaranteed that if such type exists, then it is unique.

Note that you should print the type dictated by Gena (if such type exists) and not any type that can contain n values.

Examples

Input

3
pair pair int int int


Output

pair&lt;pair&lt;int,int&gt;,int&gt;

Input

1
pair int


Output

Error occurred

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout

n = int(stdin.readline())
s = stdin.readline().strip().split()

for i in range(len(s)):
    if s[i] == 'pair':
        s[i] = '2'
    else:
        s[i] = '1'

def check(g):
    if g == ['1']:
        return True
    
    cnt = 2
    label = 1
    for i in range(1, len(g)):
        if not cnt:
            label = 0
            
        if g[i] == '1':
            cnt -= 1
        else:
            cnt += 1
    
    if cnt:
        label = 0
        
    return label


def generate():
    
    ans = ''
    stack = []
    
    cnt = 0
    for ind in range(len(s)):
        if s[ind] == '2':
            stack.extend(['>', ',', '<'])
    
        ans += s[ind]
        
        if len(stack):
            ans += stack.pop()
        
        cnt += 1
        while ans[-1] == '>' and len(stack):
            ans += stack.pop()
            cnt += 1
    
    return ans

if not check(s):
    stdout.write('Error occurred')
else:
    
    ans = generate()
    replace = ''
    
    for i in range(len(ans)):
        if ans[i] == '1':
            stdout.write('int')
        elif ans[i] == '2':
            stdout.write('pair')
        else:
            stdout.write(ans[i])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cstlbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_depth = params.get('max_depth', 3)
        self.min_n = params.get('min_n', 1)
        self.max_n = params.get('max_n', 1000)
    
    def case_generator(self):
        # 生成一个合法的类型结构
        structure = self.generate_random_structure(current_depth=1)
        s = self.structure_to_s(structure)
        correct_answer = self.structure_to_str(structure)
        n = self.count_ints(structure)
        return {
            'n': n,
            's': s,
            'correct_answer': correct_answer
        }
    
    def generate_random_structure(self, current_depth=1):
        # 避免生成过深的结构
        if current_depth > self.max_depth:
            return 'int'
        # 50% 的概率生成 'int'，50% 的概率生成 'pair'
        if random.random() < 0.5:
            return 'int'
        else:
            left = self.generate_random_structure(current_depth + 1)
            right = self.generate_random_structure(current_depth + 1)
            return ('pair', left, right)
    
    def structure_to_s(self, structure):
        if structure == 'int':
            return ['int']
        else:
            s = ['pair']
            s += self.structure_to_s(structure[1])
            s += self.structure_to_s(structure[2])
            return s
    
    def structure_to_str(self, structure):
        if structure == 'int':
            return 'int'
        else:
            left = self.structure_to_str(structure[1])
            right = self.structure_to_str(structure[2])
            return f'pair<{left},{right}>'
    
    def count_ints(self, structure):
        if structure == 'int':
            return 1
        else:
            return self.count_ints(structure[1]) + self.count_ints(structure[2])
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = ' '.join(question_case['s'])
        prompt = f"Vasya需要帮助Gena添加标点符号，使得输入的类型描述合法。已知n={n}，输入的词列表为：{s}。请按照规则添加标点符号，生成正确的类型描述。规则如下：\n"
        prompt += "1. 类型可以是'int'或者'pair<type1,type2>'，其中type1和type2也是合法类型。\n"
        prompt += "2. 生成的类型必须唯一且合法，否则输出'Error occurred'。\n"
        prompt += "请将最终答案放在[answer]标签中，例如：[answer]pair<int,int>[/answer]。"
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1].strip()
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_answer = identity['correct_answer']
        return solution == correct_answer

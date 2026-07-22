"""# 

### 谜题描述
Little Petya very much likes playing with little Masha. Recently he has received a game called \"Zero-One\" as a gift from his mother. Petya immediately offered Masha to play the game with him.

Before the very beginning of the game several cards are lain out on a table in one line from the left to the right. Each card contains a digit: 0 or 1. Players move in turns and Masha moves first. During each move a player should remove a card from the table and shift all other cards so as to close the gap left by the removed card. For example, if before somebody's move the cards on the table formed a sequence 01010101, then after the fourth card is removed (the cards are numbered starting from 1), the sequence will look like that: 0100101. 

The game ends when exactly two cards are left on the table. The digits on these cards determine the number in binary notation: the most significant bit is located to the left. Masha's aim is to minimize the number and Petya's aim is to maximize it.

An unpleasant accident occurred before the game started. The kids spilled juice on some of the cards and the digits on the cards got blurred. Each one of the spoiled cards could have either 0 or 1 written on it. Consider all possible variants of initial arrangement of the digits (before the juice spilling). For each variant, let's find which two cards are left by the end of the game, assuming that both Petya and Masha play optimally. An ordered pair of digits written on those two cards is called an outcome. Your task is to find the set of outcomes for all variants of initial digits arrangement.

Input

The first line contains a sequence of characters each of which can either be a \"0\", a \"1\" or a \"?\". This sequence determines the initial arrangement of cards on the table from the left to the right. The characters \"?\" mean that the given card was spoiled before the game. The sequence's length ranges from 2 to 105, inclusive.

Output

Print the set of outcomes for all possible initial digits arrangements. Print each possible outcome on a single line. Each outcome should be represented by two characters: the digits written on the cards that were left by the end of the game. The outcomes should be sorted lexicographically in ascending order (see the first sample).

Examples

Input

????


Output

00
01
10
11


Input

1010


Output

10


Input

1?1


Output

01
11

Note

In the first sample all 16 variants of numbers arrangement are possible. For the variant 0000 the outcome is 00. For the variant 1111 the outcome is 11. For the variant 0011 the outcome is 01. For the variant 1100 the outcome is 10. Regardless of outcomes for all other variants the set which we are looking for will contain all 4 possible outcomes.

In the third sample only 2 variants of numbers arrangement are possible: 111 and 101. For the variant 111 the outcome is 11. For the variant 101 the outcome is 01, because on the first turn Masha can remove the first card from the left after which the game will end.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# -*- coding: utf-8 -*-
import sys

if(__name__=='__main__'):
	poker=raw_input()
	n0=0;n1=0;
	L=len(poker)
	for i in xrange(L):
		if(poker[i]=='1'):
			n1+=1
		elif(poker[i]=='0'):
			n0+=1
	x=L-n0-n1
	
	if(n1<=(L-1)/2): #如果1的个数少于一半
		print(\"00\")
	if(n0<=(L/2) and L/2<=n0+x):
		if(poker[L-1]=='1' or (poker[L-1]=='?' and n1+1 <= (L+1)/2)):
			print(\"01\")
		if(poker[L-1]=='0' or (poker[L-1]=='?' and n0+1 <= L/2)):
			print(\"10\") 
	
	if(n0<=(L-2)/2):
		print(\"11\");
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Czeroonebootcamp(Basebootcamp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_length = kwargs.get('min_length', 2)
        self.max_length = kwargs.get('max_length', 10)
        self.prob_0 = kwargs.get('prob_0', 0.3)
        self.prob_1 = kwargs.get('prob_1', 0.3)
        self.prob_q = kwargs.get('prob_q', 0.4)
    
    def case_generator(self):
        length = random.randint(self.min_length, self.max_length)
        chars = []
        total = self.prob_0 + self.prob_1 + self.prob_q
        for _ in range(length):
            if total == 0:
                r = random.random() * 3
                idx = int(r % 3)
                chars.append(['0', '1', '?'][idx])
            else:
                r = random.uniform(0, total)
                if r < self.prob_0:
                    chars.append('0')
                elif r < self.prob_0 + self.prob_1:
                    chars.append('1')
                else:
                    chars.append('?')
        return {'input': ''.join(chars)}
    
    @staticmethod
    def prompt_func(question_case):
        input_str = question_case['input']
        return f"""你是解谜专家，需要解决一个名为“Zero-One”游戏的谜题。游戏规则如下：

两位玩家Masha和Petya轮流在初始的卡片序列中移除卡片。Masha先手。卡片排成一行，每次移除一张后剩余卡片左移补齐。游戏结束时剩下两张卡片组成二进制数（左侧为高位），Masha希望数值最小化，Petya希望最大化。

当前卡片序列包含模糊卡片(?可替换为0或1)，请计算所有可能的初始数值组合下游戏结束时的最终结果集合。

输入序列：{input_str}

请按字典序输出所有可能结果，每行一个，并包裹在[answer]和[/answer]标签之间，例如：
[answer]
00
01
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n')]
        valid = []
        for line in lines:
            if line in {'00', '01', '10', '11'}:
                valid.append(line)
        return valid if valid else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):  # 修正缩进
        if not solution:
            return False
        expected = cls.compute_valid_outcomes(identity['input'])
        return sorted(solution) == sorted(expected)
    
    @classmethod
    def compute_valid_outcomes(cls, input_str):  # 修正缩进
        n0, n1, L = 0, 0, len(input_str)
        for c in input_str:
            n0 += (c == '0')
            n1 += (c == '1')
        x = L - n0 - n1
        outcomes = set()
        
        if n1 <= (L-1) // 2:
            outcomes.add('00')
        
        L_half = L // 2
        if n0 <= L_half and L_half <= (n0 + x):
            last_char = input_str[-1] if L > 0 else ''
            if last_char == '1' or (last_char == '?' and (n1 + 1) <= (L + 1) // 2):
                outcomes.add('01')
            if last_char == '0' or (last_char == '?' and (n0 + 1) <= L_half):
                outcomes.add('10')
        
        if n0 <= (L-2) // 2:
            outcomes.add('11')
        
        return sorted(outcomes)

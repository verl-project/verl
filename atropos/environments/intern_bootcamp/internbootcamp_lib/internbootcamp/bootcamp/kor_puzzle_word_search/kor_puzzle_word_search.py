"""# 谜题训练场开发任务

## 任务概述
你是一位资深程序员，我需要你帮我实现一个特定谜题的训练场环境类。这个类继承自`Basebootcamp`，用于生成谜题实例并验证解答。

## 背景说明
我正在开发一系列谜题训练场，每个训练场对应一个特定类型的谜题。训练场类命名为`{PuzzleName}bootcamp`，其中`PuzzleName`是谜题的名称。

每个训练场类主要提供两个核心功能：
1. 生成该谜题类型的问题实例
2. 验证用户对问题的回答是否正确

## 技术接口规范

### 类方法实现要求

```python
from bootcamp import Basebootcamp

class {PuzzleName}bootcamp(Basebootcamp):
    def __init__(self, **params):
        \"\"\"
        请你自定义params，以保存该puzzle相关的参数，例如网格大小等，参数配有默认值
        \"\"\"
        pass
    
    def case_generator(self):
        \"\"\"
        生成谜题实例，提示：为保证谜题有解，可以先生成结果再对结果处理得到谜题
        返回：一个可JSON序列化的字典（避免包含set等无法通过json.dumps处理的数据结构）
        \"\"\"
        pass
    
    @staticmethod
    def prompt_func(question_case) -> str:
        \"\"\"
        将case_generator生成的谜题实例转换为文本形式的问题，问题中包含问题背景、对谜题规则的介绍、具体要解决的谜题实例、期望最终答案的格式，
        例如：你是xxxx，请你解答yyyy，规则如下：yyyy，最终答案放置在：zzzzz
        注意：请参照提供的谜题描述进行复述，规则应当描述详细，包括任务背景、具体任务操作规则、对题目格式和答案格式的含义介绍等，

        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 格式化的问题字符串
            
        注意:
            1. 需考虑问题的格式，以便后续能正确提取
            2. 问题描述中应包含期望的答案格式说明，以便后续能正确提取，为了避免抽取时匹配出干扰项，请要求模型将答案放在特定标签（如双括号）内，例如[[your answer here]]
        \"\"\"
        pass
    
    @staticmethod
    def extract_output(output):
        \"\"\"
        从LLM的回复中提取符合格式要求的答案，如有多个，请抽取最后一个，避免使用re.search等只抽取第一个结果的方式。
        
        参数:
            output: LLM的完整输出（包含原始问题和回答）
            
        返回:
            提取的答案，若未找到符合格式的答案则返回None
        \"\"\"
        pass
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        \"\"\"
        验证提取的答案是否正确，注意一个问题可以能有多个解，按照谜题规则进行检验，不要直接匹配可能的答案。
        
        参数:
            solution: extract_output提取的答案
            identity: case_generator生成的谜题实例
            
        返回:
            bool: 答案是否正确
        \"\"\"
        pass
```

### 验证评分方法（基类已实现）

```python
@classmethod
def verify_score(cls, model_output, identity:dict, format_score=0.1) -> float:
    \"\"\"
    验证输出结果并评分。
    
    参数:
        model_output: 模型的完整输出
        identity: 谜题实例（由case_generator生成）
        format_score: 答案格式正确时的基础分数
    
    返回:
        float: 评分结果（0-1之间）
    \"\"\"
    score = 0. 
    try:
        extract_solution = cls.extract_output(model_output)
        if extract_solution is None:
            return score
        else:
            score = format_score # 格式正确时的基础分数
        if cls._verify_correction(extract_solution, identity):
            score = 1.  # 答案完全正确时的满分
    except Exception as e:
        # 处理异常情况
        pass
    return score
```

### 使用示例

```python
# 初始化谜题训练场
bootcamp = Puzzlebootcamp()

# 生成谜题实例
case = bootcamp.case_generator()

# 将谜题转换为文本问题
prompt = Puzzlebootcamp.prompt_func(case)

# 获取LLM对问题的解答
response = get_response(prompt, \"LLM\")

# 从完整对话中提取答案
extracted_output = Puzzlebootcamp.extract_output(prompt + response)

# 验证答案并评分
score = Puzzlebootcamp.verify_score(extracted_output, case)
```

## 你的任务
请根据以下谜题描述（谜题描述可能不完整，请先结合你的知识澄清规则），实现一个完整的谜题训练场类：

### 谜题描述
1.The game is played on a square grid of letters.
2.The player is given a set of words to find on the grid.
3.The words can be arranged horizontally, vertically, diagonally, backwards or staggered.
4.Find all hidden words and label them.Example questions are as follows:

<example 0>
I  I  S  P  L  P  W  R  S  B  Z  K  O

L  H  A  Y  D  N  V  Q  A  S  K  Q  E

T  Y  F  K  M  T  C  I  I  H  I  Z  L

I  K  A  S  N  E  S  M  V  W  E  E  P

N  S  Z  N  N  E  E  T  Q  A  L  V  U

C  V  N  I  A  R  Z  K  G  G  L  V  C

H  O  M  V  M  A  B  A  A  N  V  D  C

O  K  J  A  U  V  G  R  I  E  G  M  I

P  I  W  R  H  E  K  O  A  R  S  D  N

I  A  N  T  C  L  D  V  J  H  R  I  I

N  H  O  S  S  L  E  D  N  E  M  Z  Z

H  C  A  B  V  K  R  V  R  A  D  S  R

X  T  R  A  Z  O  M  P  F  I  R  V  A

Word List:
PUCCINI CHOPIN RAVEL.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 0>

<example 1>
F  J  G  N  I  W  S  S  H  I  M

L  T  O  L  I  P  O  T  U  A  I

E  S  T  N  A  R  F  P  D  S  S

E  T  D  R  A  O  B  H  S  A  D

H  O  W  U  N  P  O  T  P  M  T

W  E  Z  D  A  E  H  S  A  C  T

Q  T  X  D  D  L  N  R  L  I  M

G  M  Y  E  L  L  A  G  F  B  L

U  W  T  R  I  E  C  F  I  W  Q

Y  A  R  D  U  R  H  T  E  N  W

C  B  N  R  O  T  A  V  E  L  E


Word List:
AUTOPILOT PROPELLER DASHBOARD RUDDER ELEVATOR.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 1>

<example 2>
J  R  E  V  U  O  C  N  A  V  J

W  E  U  V  A  L  G  H  Z  T  J

F  I  L  E  A  J  Y  T  E  I  G

W  T  I  R  A  L  E  I  G  H  S

L  R  K  R  T  E  L  M  V  C  L

A  A  L  A  D  O  J  S  O  B  H

R  C  A  Z  J  C  B  O  I  Q  X

Y  V  R  Z  U  E  K  A  R  D  V

B  Z  B  A  U  M  F  T  C  I  W

K  G  A  N  A  L  L  E  G  A  M

J  W  C  O  L  U  M  B  U  S  W


Word List:
CABOT DRAKE CABRAL.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 2>

<example 3>
S  H  K  G  P  N  O  Z  A  O  X

N  Q  R  S  U  Q  I  T  B  Q  Q

E  H  O  M  E  P  L  A  T  E  Q

D  E  H  Y  N  N  O  R  W  R  W

B  L  S  L  I  D  E  B  C  I  E

U  M  E  S  L  T  W  A  L  K  B

L  E  X  I  T  U  T  S  M  R  S

L  T  M  I  F  C  T  E  A  M  Y

P  D  H  A  H  T  G  N  Z  A  L

E  O  R  E  G  G  U  L  S  W  P
 
N  U  R  S  F  V  K  O  H  T  J

Word List:
BASE NOHITTER BULLPEN OUTFIELD CATCHER.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 3>

<example 4>
LIARDERDKXV
TORKNYPXAFM 
HCCIUZGNATH 
YUMOSASAURJ 
LOIMOHTANTL 
ATLASBEARXQ 
CSYMOCILETU 
IAKMDXFZKRA 
NODOTPYLGXG 
EJDTEBGJJAG
MOWHCORUAIA

Word List:
ATLASBEAR MAMMOTH AUROCH MOA DODO MOSASAUR GLYPTODON.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 4>

<example 5>
YJLBAFYTTCL 
SKYSQVONION 
UASPARAGUSO
MLMCELMRLUB
BEETPVKWKSK
XNVGKOCDPOJ
NNGZCGCICYT 
GERIIDNHMOC
RFQOHAGPAWR
SLCUCUMBERG
EKOHCITRARD

Word List:
ARTICHOKE EGGPLANT ASPARAGUS FENNEL BEET KALE CHARD OKRA CHICKPEA ONION CORN SPINACH CUCUMBER YAM.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 5>

<example 6>
TKRIRRXYLZFQPS
WASHERTEHCTAH
NBPBLTKVNQGSA 
NPBEUAMEGROAM
UWORRABLEEHWM
FKLWLAORLPOOE 
WKTLEVUIYAMOR 
EAEWERFQIPPDC 
YTXSSFSPSDYPA 
VURUDEGAINMPR 
PGDHCNERWAHCB 
ORTCFXLMLSQFA 
RYXUPEVCWUTYT

Word List:
BOLT RAKE CLAMP RULER FILE SANDPAPER GLOVES SQUARE HAMMER TAPE HATCHET WASHER LEVEL WHEELBARROW MALLET WOOD POWERSAW WRENCH.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 6>

<example 7>
LECTULSAXNR
XPOEELPASOV
MZBMNMRLQMC
MLOSANGELES
EIDTSVSYSBX
MWASHINGTON
PULMVZOFUSV
HALNITSUATD
IIAXLTUCSON
SLSJLFRESNO
UTVDENVERVK

Word List:
AUSTIN MEMPHIS BOSTON MIAMI DALLAS NASHVILLE DENVER TAMPA ELPASO TUCSON FRESNO TULSA LOSANGELES WASHINGTON.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 7>

<example 8>
SDHTWZEGCKX
WSOODNICKEL 
AFAAXKNNWXN
IMERCURYCNR
RLFEBCWRIME
OMZTXNQCUCV
NMMXSIHIXOL
PEWTERLDUPI
FQEYOLAJGPS
REJMASOLDER
LXEGOLDFPRS

Word List:
BRASS NICHROME COPPER NICKEL GALLIUM PEWTER GOLD SILVER IRON SOLDER LEAD STEEL MERCURY ZING.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 8>

<example 9>
BASTIONGVFIRL
UAWLANACDPOKS
NKTEFSPKFSFTJ
KFGDLOKHEZOVF
EFAATTRENCHUZ
RKBTUNSTKXJLS
EGDIRBWARDNRL
WONCRADOCZEAQ
OBAEEEVMXDPMD
TUSGTHGEONKPS
EZTFLWIULYEAI
FSIAXOBLLIPRI
HTSOPTUOPHNTL

Word List:
BASTION PILLBOX BUNKER RAMPART CANAL RAVELIN CASTLE REDOUBT CITADEL SANDBAG DRAWBRIDGE STOCKADE FORT TOWER MOAT TRENCH OUTPOST TURRET.
The answer should be the coordinates of the start and end points of these words in the format (row i, column j), e.g., BACH (12,4)(12,1) means that the word BACH starts with a letter in column 4 of row 12 and ends with a letter in column 1 of row 12. The output order should be the same as the order in the word list. Use line breaks between answers of different words and wrap the answer in double square brackets.
e.g.
Suppose the word list is BACH MOZART BRAHMS
The answer should be:
[[BACH (12,4)(12,1) 
MOZART(13,7)(13,2) 
BRAHMS (7,7)(12,12)]]
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorPuzzleWordSearchbootcamp(Basebootcamp):
    def __init__(self, grid_size=13, word_list=None):
        super().__init__()
        self.grid_size = grid_size
        self.word_list = word_list or self._default_word_list()

    def _default_word_list(self):
        return ["SAMPLE", "WORDS", "PUZZLE"]

    def case_generator(self):
        grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        directions = [
            (0, 1), (1, 0), (1, 1), (1, -1),
            (0, -1), (-1, 0), (-1, -1), (-1, 1)
        ]

        for word in self.word_list:
            word = word.upper()
            placed = False
            for _ in range(100):
                dr, dc = random.choice(directions)
                reverse = random.choice([True, False])
                current_word = word[::-1] if reverse else word
                length = len(current_word)

                max_row = self.grid_size - 1 - (length-1)*max(0, dr)
                min_row = (length-1)*abs(min(0, dr))
                max_col = self.grid_size - 1 - (length-1)*max(0, dc)
                min_col = (length-1)*abs(min(0, dc))

                if max_row < min_row or max_col < min_col:
                    continue

                row = random.randint(min_row, max_row)
                col = random.randint(min_col, max_col)

                valid = True
                positions = []
                for i in range(length):
                    r = row + i*dr
                    c = col + i*dc
                    if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                        valid = False
                        break
                    existing = grid[r][c]
                    if existing and existing != current_word[i]:
                        valid = False
                        break
                    positions.append((r, c))
                
                if valid:
                    for i, (r, c) in enumerate(positions):
                        grid[r][c] = current_word[i]
                    placed = True
                    break
            if not placed:
                raise ValueError(f"Failed to place word: {word}")

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i][j] is None:
                    grid[i][j] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        return {
            "grid": [row.copy() for row in grid],
            "word_list": self.word_list.copy()
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        grid = question_case["grid"]
        word_list = question_case["word_list"]
        grid_lines = ["  ".join(row) for row in grid]
        grid_str = "\n".join(grid_lines)
        word_str = " ".join(word_list) + "."
        
        return f"""You are participating in a word search puzzle. Find all hidden words in the following grid. Words can be arranged horizontally, vertically, diagonally, forwards, or backwards.

Grid (rows and columns are 0-indexed from top-left):
{grid_str}

Word List:
{word_str}

Format your answer with each word's start and end coordinates like this:
WORD (start_row,start_col)(end_row,end_col)

List answers in the order of the word list, each on a new line. Enclose your final answer within double square brackets. Example:
[[EXAMPLE (0,0)(3,0)
WORDS (2,4)(2,8)]]
"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        solution = []
        for line in content.split('\n'):
            line = line.strip()
            match = re.match(r'^(\w+)\s*\((\d+),(\d+)\)\((\d+),(\d+)\)$', line)
            if match:
                solution.append((
                    match.group(1).upper(),
                    (int(match.group(2)), int(match.group(3))),
                    (int(match.group(4)), int(match.group(5)))
                ))
        return solution if solution else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        grid = identity["grid"]
        word_list = [w.upper() for w in identity["word_list"]]
        
        if len(solution) != len(word_list):
            return False
        
        for i, (word, start, end) in enumerate(solution):
            target_word = word_list[i]
            if word.upper() != target_word:
                return False

            sr, sc = start
            er, ec = end
            length = len(target_word)

            dy = er - sr
            dx = ec - sc

            try:
                step_y = dy // (length-1) if length > 1 else dy
                step_x = dx // (length-1) if length > 1 else dx
            except ZeroDivisionError:
                return False

            if length > 1 and (abs(step_x) not in (0,1) or abs(step_y) not in (0,1)):
                return False

            formed = []
            for j in range(length):
                r = sr + j*step_y
                c = sc + j*step_x
                if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
                    return False
                formed.append(grid[r][c].upper())

            formed_str = ''.join(formed)
            if formed_str not in (target_word, target_word[::-1]):
                return False
        
        return True

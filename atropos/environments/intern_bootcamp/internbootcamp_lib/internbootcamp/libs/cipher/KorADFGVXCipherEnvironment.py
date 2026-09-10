from .BaseCipherEnvironment import BaseCipherEnvironment

from random import randint

ALPHABET_CZ = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
ALPHABET_EN = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
ALPHABET_ALL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

class Matrix:
    def __init__(self, matrixType=""):
        self.matrix = []
        self.setType(matrixType)

    def setType(self, matrixType):
        self.type = matrixType
        if self.type == "cz":
            self.length = 5
            self.alphabet = ALPHABET_CZ
        elif self.type == "en":
            self.length = 5
            self.alphabet = ALPHABET_EN
        else:
            self.length = 6
            self.alphabet = ALPHABET_ALL
        self.clean()
        self.fill()

    def fill(self):
        remains = self.alphabet
        for i in range(self.length):
            for j in range(self.length):
                self.matrix[i][j] = remains[randint(0, len(remains) - 1)]
                remains = remains.replace(self.matrix[i][j], '')

    def clean(self):
        self.matrix = [['' for i in range(self.length)] for j in range(self.length)]

    def find(self, letter):
        for i in range(self.length):
            for j in range(self.length):
                if self.matrix[i][j] == letter:
                    return (i, j)

matrix = Matrix()
matrix.matrix = [['R', 'U', 'A', '0', 'Q', 'B'], 
                ['D', '2', 'W', 'K', 'S', '1'], 
                ['H', '4', '5', 'F', 'T', 'Z'], 
                ['Y', 'C', 'G', 'X', '7', 'L'], 
                ['9', '8', 'I', '3', 'P', 'N'], 
                ['6', 'J', 'V', 'O', 'E', 'M']]


class KorADFGVXCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
    
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule19_ADFGVX'

    def encode(self, text, **kwargs):
        # 将输入转换为大写字母，并移除非字母数字字符
        text = ''.join([char.upper() for char in text if char.isalnum()])
        print(f"处理后的输入文本: {text}")
        
        line = "ADFGVX"
        encrypted = ""
        print("\n加密过程:")
        
        # 第一步：获取每个字符的行列位置并转换
        print("1. 获取每个字符的行列位置并转换为加密字符:")
        binary_pairs = []
        for i in text:
            row, col = matrix.find(i)
            row_char = line[row]
            col_char = line[col]
            binary_pairs.append((row_char, col_char))
            print(f"字符 {i} 在矩阵中的位置是 ({row},{col})，转换为加密二元组 ({row_char},{col_char})")
        
        # 第二步：合并所有行和列
        print("\n2. 合并所有加密二元组的行和列:")
        for pair in binary_pairs:
            encrypted += pair[0]
        for pair in binary_pairs:
            encrypted += pair[1]
        print(f"先读取所有行: {''.join([pair[0] for pair in binary_pairs])}")
        print(f"再读取所有列: {''.join([pair[1] for pair in binary_pairs])}")
        print(f"最终密文: {encrypted}")
        
        return encrypted

    def decode(self, text, **kwargs):
        line = "ADFGVX"
        print(f"输入密文: {text}")
        
        print("\n解密过程:")
        # 第一步：将密文分成两半
        half_length = len(text) // 2
        first_half = text[:half_length]
        second_half = text[half_length:]
        print(f"1. 将密文分成两半:")
        print(f"前半部分: {first_half}")
        print(f"后半部分: {second_half}")
        
        # 第二步：配对解密
        print("\n2. 将前后半部分配对并解密:")
        decryptedText = ""
        for i, (first, second) in enumerate(zip(first_half, second_half)):
            row = line.index(first)
            col = line.index(second)
            char = matrix.matrix[row][col]
            decryptedText += char
            print(f"配对 ({first},{second}) 对应矩阵位置 ({row},{col})，解密为字符: {char}")
        
        print(f"\n最终明文: {decryptedText}")
        return decryptedText

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 不含标点和空格的大写字母字符串
- 输出:
    - 密文: 不含标点和空格的大写字母字符串
- 准备:
    - 6x6矩阵 (矩阵中的行和列从0开始计数)
        [['R', 'U', 'A', '0', 'Q', 'B'], 
        ['D', '2', 'W', 'K', 'S', '1'], 
        ['H', '4', '5', 'F', 'T', 'Z'], 
        ['Y', 'C', 'G', 'X', '7', 'L'], 
        ['9', '8', 'I', '3', 'P', 'N'], 
        ['6', 'J', 'V', 'O', 'E', 'M']]
    - 加密字符集: "ADFGVX"
- 加密步骤:
    - 对明文中的每个字符:
        - 在6X6矩阵中找到该字符的行数和列数。例如，A的行数为0，列数为2。
        - 加密字符集中的字符位置标记为0-6。使用加密字符集中对应位置的字符替换行数和列数得到加密二元组。
            - 例如，A的行数为0对应加密字符集中的A，列数为2对应加密字符集中的F，所以A的加密二元组为(A,F)。
        - 读取所有加密二元组的行，然后读取所有加密二元组的列得到最终密文。
            - 例如，加密二元组为(A,F)(X,V)，最后读取为AXFV，所以最终密文为AXFV。"""

    def get_decode_rule(self, ):
        return """解密规则:
- 输入:
    - 密文: 不含标点和空格的大写字母字符串
- 输出:
    - 明文: 不含标点和空格的大写字母字符串
- 准备:
    - 6x6矩阵 (与加密相同)
    - 加密字符集 (与加密相同)
- 解密步骤:
    - 将密文分成两半
    - 每次从上半部分和下半部分各取一个字母作为解密二元组:
        - 加密字符集中的字符位置标记为0-6，使用加密字符集中对应字符的位置解密出解密二元组代表的行数和列数。
            - 例如，解密二元组为加密二元组(A,F)，A的位置为0，F的位置为2，所以行数为0，列数为2，得到(0,2)。
        - 使用得到的行数和列数在6x6矩阵中找到对应位置的字母作为解密后的字符。
            - 例如，(0,2)位置的字符为A，所以AF解密后的字符为A。
    - 连接所有解密后的字符得到最终明文。"""

from .BaseCipherEnvironment import BaseCipherEnvironment

polybius_square = [
    ['R', 'T', 'X', 'F', 'S'],
    ['W', 'C', 'M', 'V', 'H'],
    ['Z', 'J', 'A', 'P', 'B'],
    ['L', 'Q', 'Y', 'G', 'K'],
    ['N', 'E', 'U', 'D', 'I']
]

def find_position(char):
    for i in range(len(polybius_square)):
        for j in range(len(polybius_square[i])):
            if polybius_square[i][j] == char:
                return i + 1, j + 1
    return None


class KorPolybiusSquareCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "Polybius Square Cipher from KOR-bench"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule4_PolybiusSquareCipher"
    
    def encode(self, text, **kwargs):
        print(f"开始加密文本: {text}")
        # 转换为大写并只保留字母
        text = ''.join([c.upper() for c in text if c.isalpha()])
        print(f"预处理后的文本: {text}")
        
        encrypted_text = ""
        print("开始逐字符加密:")
        for char in text:
            print(f"处理字符: {char}")
            if char == 'O':
                encrypted_text += '66'
                print(f"字符 {char} 不在Polybius方阵中，替换为: 66")
            else:
                row, col = find_position(char)
                if row and col:
                    encrypted_text += f"{row}{col}"
                    print(f"字符 {char} 在方阵中的位置是: 第{row}行第{col}列，替换为: {row}{col}")
        
        print(f"最终加密结果: {encrypted_text}")
        return encrypted_text

    def decode(self, text, **kwargs):
        print(f"开始解密文本: {text}")
        decrypted_text = ""
        i = 0
        print("开始逐对数字解密:")
        while i < len(text):
            if text[i:i+2] == '66':
                decrypted_text += 'O'
                print(f"遇到数字对: 66，解密为: O")
                i += 2
            elif text[i].isdigit():
                row = int(text[i])
                col = int(text[i+1])
                char = polybius_square[row-1][col-1]
                decrypted_text += char
                print(f"遇到数字对: {row}{col}，对应方阵位置: 第{row}行第{col}列，解密为: {char}")
                i += 2
        
        print(f"最终解密结果: {decrypted_text}")
        return decrypted_text


    def get_decode_rule(self, ):
        decode_rule = """
解密规则:
- 输入:
    - 密文: 字符串
- 输出:
    - 明文: 大写字母字符串
- 准备:
    - Polybius方阵(与加密相同)
- 解密步骤(与加密步骤相反):
    - 对于密文中的每对数字CrCc:
        - 根据CrCc表示的行列号在Polybius方阵中找到对应字母
        - 如果CrCc=66，替换为"O"
    """
        return decode_rule

    def get_encode_rule(self, ):
        encode_rule = """
加密规则:
- 输入:
    - 明文: 大写字母字符串，不含标点和空格
- 输出:
    - 密文: 大写字母字符串
- 准备:
    - Polybius方阵:
    1  2  3  4  5
1   R  T  X  F  S
2   W  C  M  V  H
3   Z  J  A  P  B
4   L  Q  Y  G  K
5   N  E  U  D  I

- 加密步骤:
    - 对于每个明文字符p:
        - 如果p是存在于Polybius方阵中的大写字母:
            - 用字符在方阵中的行号和列号(都从1开始计数)替换该字符
        - 特别地，字母O不存在于方阵中，用66替换
    """
        return encode_rule

        
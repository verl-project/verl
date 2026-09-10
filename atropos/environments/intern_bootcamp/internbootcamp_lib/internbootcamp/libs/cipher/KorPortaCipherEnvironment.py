from .BaseCipherEnvironment import BaseCipherEnvironment

class PortaCipher:
    def __init__(self, key):
        self.key = key.upper()
        self.alphabets = [
            'NOPQRSTUVWXYZABCDEFGHIJKLM',
            'ZNOPQRSTUVWXYBCDEFGHIJKLMA',
            'YZNOPQRSTUVWXCDEFGHIJKLMAB',
            'XYZNOPQRSTUVWDEFGHIJKLMABC',
            'WXYZNOPQRSTUVEFGHIJKLMABCD',
            'VWXYZNOPQRSTUFGHIJKLMABCDE',
            'UVWXYZNOPQRSTGHIJKLMABCDEF',
            'TUVWXYZNOPQRSHIJKLMABCDEFG',
            'STUVWXYZNOPQRIJKLMABCDEFGH',
            'RSTUVWXYZNOPQJKLMABCDEFGHI',
            'QRSTUVWXYZNOPKLMABCDEFGHIJ',
            'PQRSTUVWXYZNOLMABCDEFGHIJK',
            'OPQRSTUVWXYZNMABCDEFGHIJKL'
        ]
        self.char_to_alphabet_index = {chr(i + ord('A')): i // 2 for i in range(26)}

    def encrypt_char(self, char, key_char):
        index = self.char_to_alphabet_index[key_char]
        return self.alphabets[index][ord(char) - ord('A')]

    def decrypt_char(self, char, key_char):
        index = self.char_to_alphabet_index[key_char]
        return chr(self.alphabets[index].index(char) + ord('A'))

    def encrypt(self, plaintext):
        plaintext = plaintext.upper()
        ciphertext = []
        for i, char in enumerate(plaintext):
            if char.isalpha():
                key_char = self.key[i % len(self.key)]
                ciphertext.append(self.encrypt_char(char, key_char))
            else:
                ciphertext.append(char)
        return ''.join(ciphertext)

    def decrypt(self, ciphertext):
        ciphertext = ciphertext.upper()
        plaintext = []
        for i, char in enumerate(ciphertext):
            if char.isalpha():
                key_char = self.key[i % len(self.key)]
                plaintext.append(self.decrypt_char(char, key_char))
            else:
                plaintext.append(char)
        return ''.join(plaintext)


class KorPortaCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = 'Portacipher from kor'
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule8_PortaCipher'
    
    def encode(self, text, key):
        print(f"开始加密文本: {text}")
        print(f"使用密钥: {key}")
        
        # 预处理文本，只保留字母并转为大写
        processed_text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"预处理后的文本: {processed_text}")
        
        porta = PortaCipher(key)
        encoded_text = porta.encrypt(processed_text)
        
        print("加密步骤:")
        for i, (p, c) in enumerate(zip(processed_text, encoded_text)):
            key_char = key[i % len(key)].upper()
            print(f"  第{i+1}个字符 {p} 使用密钥字符 {key_char}:")
            print(f"    - 查找密钥字符 {key_char} 对应的替换表")
            print(f"    - 将明文字符 {p} 替换为密文字符 {c}")
        
        print(f"最终加密结果: {encoded_text}")
        return encoded_text

    def decode(self, text, key):
        print(f"开始解密文本: {text}")
        print(f"使用密钥: {key}")
        
        porta = PortaCipher(key)
        decoded_text = porta.decrypt(text)
        
        print("解密步骤:")
        for i, (c, p) in enumerate(zip(text, decoded_text)):
            key_char = key[i % len(key)].upper()
            print(f"  第{i+1}个字符 {c} 使用密钥字符 {key_char}:")
            print(f"    - 查找密钥字符 {key_char} 对应的替换表")
            print(f"    - 将密文字符 {c} 还原为明文字符 {p}")
        
        print(f"最终解密结果: {decoded_text}")
        return decoded_text

    def get_encode_rule(self, ):
        return """加密规则:
输入:
    - 明文: 大写字母字符串，不含标点和空格
    - 密钥: 用于选择替换表的字符串
输出:
    - 密文: 大写字母字符串
准备工作:
    - 密码替换表:
        使用以下13个密码替换表，每个表对应两个字母:
        AB: NOPQRSTUVWXYZABCDEFGHIJKLM
        CD: ZNOPQRSTUVWXYBCDEFGHIJKLMA
        EF: YZNOPQRSTUVWXCDEFGHIJKLMAB
        GH: XYZNOPQRSTUVWDEFGHIJKLMABC
        IJ: WXYZNOPQRSTUVEFGHIJKLMABCD
        KL: VWXYZNOPQRSTUFGHIJKLMABCDE
        MN: UVWXYZNOPQRSTGHIJKLMABCDEF
        OP: TUVWXYZNOPQRSHIJKLMABCDEFG
        QR: STUVWXYZNOPQRIJKLMABCDEFGH
        ST: RSTUVWXYZNOPQJKLMABCDEFGHI
        UV: QRSTUVWXYZNOPKLMABCDEFGHIJ
        WX: PQRSTUVWXYZNOLMABCDEFGHIJK
        YZ: OPQRSTUVWXYZNMABCDEFGHIJKL
    - 标准字母表:
        ABCDEFGHIJKLMNOPQRSTUVWXYZ
加密步骤:
    - 将密钥中的每个字母与明文中的每个字母配对。如果密钥比明文短，重复使用密钥
    - 对于每个明文字符p:
        - 根据与之配对的密钥字母找到对应的密码替换表
        - 在标准字母表中找到p的位置，用密码替换表中相同位置的字母替换它"""

    def get_decode_rule(self, ):
        return """解密规则:
输入:
    - 密文: 大写字母字符串
    - 密钥: 用于选择替换表的字符串
输出:
    - 明文: 大写字母字符串
准备工作:
    - 密码替换表: (与加密相同)
    - 标准字母表: (与加密相同)
解密步骤:
    - 将密钥中的每个字母与密文中的每个字母配对。如果密钥比密文短，重复使用密钥
    - 对于每个密文字符c:
        - 根据与之配对的密钥字母找到对应的密码替换表
        - 在密码替换表中找到c的位置，用标准字母表中相同位置的字母还原它"""

        
from .BaseCipherEnvironment import BaseCipherEnvironment

class KorPigpenMasonicCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "Pigpen Masonic Cipher from Kor-bench"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule2_PigpenMasonicCipher"
    
    def encode(self, text, **kwargs):
        print("开始加密过程...")
        print(f"原始输入文本: {text}")
        
        # 将文本转换为大写并移除非字母字符
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        # 初始化加密表
        encryption_table = {
            'A': '!', 'B': '@', 'C': '#', 'D': '$',
            'E': '%', 'F': '^', 'G': '&', 'H': '*',
            'I': '(', 'J': ')', 'K': '_', 'L': '+',
            'M': '=', 'N': '~', 'O': '?', 'P': '/',
            'Q': '0', 'R': ':', 'S': ';', 'T': '<',
            'U': '>', 'V': '1', 'W': '2', 'X': '3',
            'Y': '4', 'Z': '5'
        }
        
        encrypted_text = ""
        print("\n逐字符加密过程:")
        for char in text:
            if char in encryption_table:
                encrypted_char = encryption_table[char]
                print(f"字符 {char} 被加密为 {encrypted_char}")
                encrypted_text += encrypted_char
            else:
                encrypted_text += char
                
        print(f"\n最终加密结果: {encrypted_text}")
        return encrypted_text

    def decode(self, text, **kwargs):
        print("开始解密过程...")
        print(f"加密文本: {text}")
        
        # 初始化解密表
        encryption_table = {
            'A': '!', 'B': '@', 'C': '#', 'D': '$',
            'E': '%', 'F': '^', 'G': '&', 'H': '*',
            'I': '(', 'J': ')', 'K': '_', 'L': '+',
            'M': '=', 'N': '~', 'O': '?', 'P': '/',
            'Q': '0', 'R': ':', 'S': ';', 'T': '<',
            'U': '>', 'V': '1', 'W': '2', 'X': '3',
            'Y': '4', 'Z': '5'
        }
        decryption_table = {v: k for k, v in encryption_table.items()}
        
        decrypted_text = ""
        print("\n逐字符解密过程:")
        for char in text:
            if char in decryption_table:
                decrypted_char = decryption_table[char]
                print(f"符号 {char} 被解密为 {decrypted_char}")
                decrypted_text += decrypted_char
            else:
                decrypted_text += char
                
        print(f"\n最终解密结果: {decrypted_text}")
        return decrypted_text

    def get_encode_rule(self, ):
        encode_rule = """
加密规则:

输入:
    - 明文: 仅包含大写字母的字符串，不含标点符号和空格。
输出:
    - 密文: 大写字母字符串。
准备:
    - 加密表 = {
    'A': '!', 'B': '@', 'C': '#', 'D': '$',
    'E': '%', 'F': '^', 'G': '&', 'H': '*',
    'I': '(', 'J': ')', 'K': '_', 'L': '+',
    'M': '=', 'N': '~', 'O': '?', 'P': '/',
    'Q': '0', 'R': ':', 'S': ';', 'T': '<',
    'U': '>', 'V': '1', 'W': '2', 'X': '3',
    'Y': '4', 'Z': '5'
    }
加密步骤:
    - 对于每个给定的明文字符 p:
        - 如果 p 是大写字母且存在于加密表中:
            - 用加密表中对应的符号替换 p。
    """
        return encode_rule

    def get_decode_rule(self, ):
        decode_rule = """
解密规则:

输入:
    - 密文: 大写字母字符串。
输出:
    - 明文: 大写字母字符串。
准备:
    - 加密表 = {
    'A': '!', 'B': '@', 'C': '#', 'D': '$',
    'E': '%', 'F': '^', 'G': '&', 'H': '*',
    'I': '(', 'J': ')', 'K': '_', 'L': '+',
    'M': '=', 'N': '~', 'O': '?', 'P': '/',
    'Q': '0', 'R': ':', 'S': ';', 'T': '<',
    'U': '>', 'V': '1', 'W': '2', 'X': '3',
    'Y': '4', 'Z': '5'
    }
解密步骤 (与加密步骤完全相反):
    - 对于每个给定的密文字符 c:
        - 如果 c 是加密表中的符号且存在于加密表中:
            - 用加密表中对应的大写字母替换 c。
    """
        return decode_rule

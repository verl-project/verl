from .BaseCipherEnvironment import BaseCipherEnvironment

import re

multitap_encode = {
    'A': '2^1', 'B': '2^2', 'C': '2^3',
    'D': '3^1', 'E': '3^2', 'F': '3^3',
    'G': '4^1', 'H': '4^2', 'I': '4^3',
    'J': '5^1', 'K': '5^2', 'L': '5^3',
    'M': '6^1', 'N': '6^2', 'O': '6^3',
    'P': '7^1', 'Q': '7^2', 'R': '7^3', 'S': '7^4',
    'T': '8^1', 'U': '8^2', 'V': '8^3',
    'W': '9^1', 'X': '9^2', 'Y': '9^3', 'Z': '9^4',
}

multitap_decode = {v: k for k, v in multitap_encode.items()}


class KorMultiTapPhoneCodeEnvironment(BaseCipherEnvironment):
    def __init__(self, problem_description='', *args, **kwargs):
        super().__init__(problem_description, *args, **kwargs)
    
    @property
    def cipher_name(self) -> str:
        return "Kor_rule3_MultiTapPhoneCode"
    
    def encode(self, text, **kwargs):
        print(f"原始文本: {text}")
        # 转换为大写字母并移除非字母字符
        text = ''.join(char.upper() for char in text if char.isalpha())
        print(f"处理后的文本(仅大写字母): {text}")
        
        print("开始逐字符加密:")
        encoded_text = ''
        for char in text:
            if char in multitap_encode:
                code = multitap_encode[char]
                print(f"字符 {char} 对应的多击编码是: {code}")
                encoded_text += code
        
        print(f"最终加密结果: {encoded_text}")
        return encoded_text

    def decode(self, text, **kwargs):
        print(f"加密文本: {text}")
        print("开始解密:")
        
        decoded_text = ''
        matches = re.findall(r'\d\^\d|\d', text)
        for match in matches:
            if match in multitap_decode:
                char = multitap_decode[match]
                print(f"多击编码 {match} 对应的字符是: {char}")
                decoded_text += char
        
        print(f"解密结果: {decoded_text}")
        return decoded_text

    def get_encode_rule(self,):
        encode_rule = """
加密规则:
- 输入:
    - 明文: 仅包含大写字母的字符串，不含标点和空格。
- 输出:
    - 密文: 不含标点的字符串。
- 准备:
    - 多击编码表
        | 字母 | 多击编码 |
        | A | 2^1 |
        | B | 2^2 |
        | C | 2^3 |
        | D | 3^1 |
        | E | 3^2 |
        | F | 3^3 |
        | G | 4^1 |
        | H | 4^2 |
        | I | 4^3 |
        | J | 5^1 |
        | K | 5^2 |
        | L | 5^3 |
        | M | 6^1 |
        | N | 6^2 |
        | O | 6^3 |
        | P | 7^1 |
        | Q | 7^2 |
        | R | 7^3 |
        | S | 7^4 |
        | T | 8^1 |
        | U | 8^2 |
        | V | 8^3 |
        | W | 9^1 |
        | X | 9^2 |
        | Y | 9^3 |
        | Z | 9^4 |
- 加密步骤:
    - 对于每个明文字符p:
        - 如果p是大写字母且存在于多击编码表中:
            - 用多击编码表中对应的多击编码替换p。
"""
        return encode_rule

    def get_decode_rule(self):
        decode_rule = """
解密规则:
- 输入:
    - 密文: 不含标点的字符串。
- 输出:
    - 明文: 大写字母字符串。
- 准备: 多击编码表(与加密相同)
- 解密步骤(与加密步骤相反):
    - 对于每个密文中的多击编码c:
        - 如果c是多击编码表中的编码:
            - 用多击编码表中对应的大写字母替换c。
"""
        return decode_rule

        
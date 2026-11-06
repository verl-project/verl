from .BaseCipherEnvironment import BaseCipherEnvironment

def prepare_alphabet(keyword):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    reversed_alphabet = alphabet[::-1]
    cleaned_keyword = "".join(dict.fromkeys(keyword))
    substitution_alphabet = cleaned_keyword + "".join([char for char in alphabet if char not in cleaned_keyword])
    return alphabet, reversed_alphabet, substitution_alphabet


class KorInverseShiftSubstitutionCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule1_InverseShiftSubstitutionCipher"
    
        
    def encode(self, text, **kwargs):
        keyword = "RFDDJUUH"
        n = 4
        
        # 处理输入文本,只保留字母并转大写
        text = ''.join([c.upper() for c in text if c.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        alphabet, reversed_alphabet, substitution_alphabet = prepare_alphabet(keyword)
        print(f"标准字母表: {alphabet}")
        print(f"反转字母表: {reversed_alphabet}")
        print(f"替换字母表: {substitution_alphabet}")
        
        ciphertext = []
        for char in text:
            print(f"\n加密字符 {char}:")
            # 步骤1: 反转映射
            reverse_char = reversed_alphabet[alphabet.index(char)]
            print(f"1. 在标准字母表中找到位置并用反转字母表对应位置字母替换: {char} -> {reverse_char}")
            
            # 步骤2: 向前移动4位
            index = (ord(reverse_char) - ord('A') + n) % 26
            shifted_char = alphabet[index]
            print(f"2. 将得到的字母向前移动4位: {reverse_char} -> {shifted_char}")
            
            # 步骤3: 替换字母表映射
            encrypted_char = substitution_alphabet[index]
            print(f"3. 在标准字母表中找到位置并用替换字母表对应位置字母替换: {shifted_char} -> {encrypted_char}")
            
            ciphertext.append(encrypted_char)
        
        result = "".join(ciphertext)
        print(f"\n最终加密结果: {result}")
        return result

    def decode(self, text, **kwargs):
        keyword = "RFDDJUUH"
        n = 4
        
        alphabet, reversed_alphabet, substitution_alphabet = prepare_alphabet(keyword)
        print(f"标准字母表: {alphabet}")
        print(f"反转字母表: {reversed_alphabet}")
        print(f"替换字母表: {substitution_alphabet}")
        
        plaintext = []
        for char in text:
            print(f"\n解密字符 {char}:")
            # 步骤1: 在替换字母表中找到对应标准字母表字母
            index = substitution_alphabet.index(char)
            standard_char = alphabet[index]
            print(f"1. 在替换字母表中找到位置并用标准字母表对应位置字母替换: {char} -> {standard_char}")
            
            # 步骤2: 向后移动4位
            decrypted_index = (index - n) % 26
            shifted_char = chr(ord('A') + decrypted_index)
            print(f"2. 将得到的字母向后移动4位: {standard_char} -> {shifted_char}")
            
            # 步骤3: 反转映射还原
            decrypted_char = alphabet[reversed_alphabet.index(shifted_char)]
            print(f"3. 在反转字母表中找到位置并用标准字母表对应位置字母替换: {shifted_char} -> {decrypted_char}")
            
            plaintext.append(decrypted_char)
        
        result = "".join(plaintext)
        print(f"\n最终解密结果: {result}")
        return result

    def get_encode_rule(self,):
        return """加密规则:
- 输入:
    - 明文: 仅包含大写字母的字符串，不含标点和空格
- 输出:
    - 密文: 大写字母字符串
- 准备:
    - 标准字母表: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    - 反转字母表: "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    - 替换字母表: "RFDJUHABCEGIKLMNOPQSTVWXYZ"
- 加密步骤:
    - 对明文中的每个字母p:
    - (1) 使用反转字母表进行反向映射。在标准字母表中找到其位置，并用反转字母表中对应位置的字母替换。例如，A映射为Z，B映射为Y。
    - (2) 将步骤(1)得到的字母在标准字母表顺序中向前移动4位。例如，如果p=A，经过步骤(1)映射为Z，然后Z在标准字母表中向前移动4位得到D。
    - (3) 将步骤(2)得到的字母，在标准字母表中找到其位置，用替换字母表中对应位置的字母替换，得到最终的密文字母。例如，如果经过步骤(2)得到字母D，则映射为J。"""

    def get_decode_rule(self):
        return """解密规则:
- 输入:
    - 密文: 大写字母字符串
- 输出:
    - 明文: 大写字母字符串
- 准备:
    - 标准字母表: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    - 反转字母表: "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    - 替换字母表: "RFDJUHABCEGIKLMNOPQSTVWXYZ"
- 解密步骤(与加密步骤完全相反):
    - (1) 对密文中的每个字母c，在替换字母表中找到其位置，用标准字母表中对应位置的字母替换。
    - (2) 将步骤(1)得到的字母按标准字母表顺序向后移动4位。
    - (3) 将步骤(2)得到的字母，在反转字母表中找到其位置，然后用标准字母表中对应位置的字母替换。例如，Z映射为A，Y映射为B。"""

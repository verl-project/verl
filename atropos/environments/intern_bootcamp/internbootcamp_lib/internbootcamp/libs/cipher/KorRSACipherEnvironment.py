from .BaseCipherEnvironment import BaseCipherEnvironment

class KorRSACipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "RSA Cipher from kor-bench"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule23_RSACipher"
    
    def encode(self, text, **kwargs):
        print("开始加密过程...")
        # 将输入转换为大写字母，去除标点和空格
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        # 设置加密参数
        e = 263
        n = 299
        print(f"使用参数: e={e}, n={n}")
        
        ciphertext = []
        print("\n逐字符加密:")
        for char in text:
            x = ord(char)
            print(f"字符 {char} 的ASCII码为 {x}")
            y = pow(x, e, n)
            print(f"计算 {x}^{e} mod {n} = {y}")
            ciphertext.append(str(y))
        
        encode_text = ','.join(ciphertext)
        print(f"\n最终加密结果: {encode_text}")
        return encode_text

    def decode(self, text, **kwargs):
        print("开始解密过程...")
        print(f"收到的加密文本: {text}")
        
        # 设置解密参数
        e = 263
        n = 299
        print(f"使用参数: e={e}, n={n}")
        
        # 分割密文
        numbers = text.split(',')
        print(f"分割后的数字序列: {numbers}")
        
        plaintext = []
        print("\n逐数字解密:")
        for num in numbers:
            c = int(num)
            z = pow(c, e, n)
            print(f"计算 {c}^{e} mod {n} = {z}")
            char = chr(z)
            print(f"对应的字符为: {char}")
            plaintext.append(char)
        
        decode_text = ''.join(plaintext)
        print(f"\n最终解密结果: {decode_text}")
        return decode_text

    def get_encode_rule(self, ):
        encode_rule = """
加密规则:
- 输入:
    - 明文: 仅包含大写字母的字符串，不含标点和空格
- 输出:
    - 密文: 由逗号分隔的数字序列，例如"y1,y2,..."
- 准备:
    - e: 263
    - n: 299
- 加密步骤:
    - 对明文中的每个字母p:
        - 获取字母p对应的ASCII码的十进制数x
        - 计算x^e mod n作为该字母的密文数字y，这里^表示乘法运算
    - 最后，将所有y用逗号连接形成最终密文
"""
        return encode_rule

    def get_decode_rule(self, ):
        decode_rule = """
解密规则:
- 输入:
    - 密文: 由逗号分隔的数字序列，例如"y1,y2,..."
- 输出:
    - 明文: 仅包含大写字母的字符串，不含标点和空格
- 准备:
    - e: 263
    - n: 299
- 解密步骤:
    - 对密文中的每个数字c:
        - 计算z = c^e mod n，这里^表示乘法运算
        - 根据z的十进制值，使用ASCII码找到对应的字母作为明文字母p
    - 最后，将所有p连接得到最终明文
"""
        return decode_rule

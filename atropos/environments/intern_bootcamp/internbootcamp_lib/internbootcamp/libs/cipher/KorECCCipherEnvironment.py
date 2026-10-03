from .BaseCipherEnvironment import BaseCipherEnvironment

def get_inverse(mu, p):
    for i in range(1, p):
        if (i*mu)%p == 1:
            return i
    return -1

def get_gcd(zi, mu):
    if mu:
        return get_gcd(mu, zi%mu)
    else:
        return zi

def get_np(x1, y1, x2, y2, a, p):
    flag = 1  
    if x1 == x2 and y1 == y2:
        zi = 3 * (x1 ** 2) + a  
        mu = 2 * y1    
    else:
        zi = y2 - y1
        mu = x2 - x1
        if zi* mu < 0:
            flag = 0        
            zi = abs(zi)
            mu = abs(mu)
    gcd_value = get_gcd(zi, mu)     
    zi = zi // gcd_value            
    mu = mu // gcd_value
    inverse_value = get_inverse(mu, p)
    k = (zi * inverse_value)
    if flag == 0:                   
        k = -k
    k = k % p
    x3 = (k ** 2 - x1 - x2) % p
    y3 = (k * (x1 - x3) - y1) % p
    return x3,y3

def get_rank(x0, y0, a, b, p):
    x1 = x0            
    y1 = (-1*y0)%p     
    tempX = x0
    tempY = y0
    n = 1
    while True:
        n += 1
        p_x,p_y = get_np(tempX, tempY, x0, y0, a, p)
        if p_x == x1 and p_y == y1:
            return n+1
        tempX = p_x
        tempY = p_y


class KorECCCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule24_ECCCipher'

    def encode(self, text, **kwargs):
        print("开始加密过程...")
        # 将输入转换为大写字母，去除标点和空格
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        k_q_x = 12
        print(f"使用预设的k_q_x值: {k_q_x}")
        
        cipher = []
        for char in text:
            print(f"\n处理字符: {char}")
            ascii_value = ord(char)
            print(f"对应的ASCII值: {ascii_value}")
            encrypted_value = ascii_value * k_q_x
            print(f"加密后的值: {encrypted_value}")
            cipher.append(str(encrypted_value))
        
        result = ','.join(cipher)
        print(f"\n最终加密结果: {result}")
        return result

    def decode(self, text, **kwargs):
        print("开始解密过程...")
        print(f"收到的加密文本: {text}")
        
        k_q_x = 12
        print(f"使用预设的k_q_x值: {k_q_x}")
        
        numbers = text.split(',')
        plain_text = ""
        
        for num in numbers:
            print(f"\n处理数字: {num}")
            decrypted_value = int(num) // k_q_x
            print(f"除以k_q_x后的值: {decrypted_value}")
            char = chr(decrypted_value)
            print(f"对应的字符: {char}")
            plain_text += char
        
        print(f"\n最终解密结果: {plain_text}")
        return plain_text

    def get_encode_rule(self):
        return """加密规则:
- 输入:
    - 明文: 不含标点和空格的大写字母字符串
- 输出:
    - 密文: 由逗号分隔的数字序列，例如"y1,y2,..."
- 准备:
    - k_q_x值: 12
- 加密步骤:
    - 对明文中的每个字母p:
        - 获取字母p对应的ASCII码值x
        - 计算x * k_q_x作为该字母对应的密文数字y
    - 最后将所有y值用逗号连接得到最终密文"""

    def get_decode_rule(self):
        return """解密规则:
- 输入:
    - 密文: 由逗号分隔的数字序列，例如"y1,y2,..."
- 输出:
    - 明文: 不含标点和空格的大写字母字符串
- 准备:
    - k_q_x值: 12
- 解密步骤:
    - 对密文中的每个数字c:
        - 计算z = c // k_q_x，其中//表示整数除法运算，返回商的整数部分
        - 根据z对应的ASCII码值找到对应的字母作为明文字母p
    - 最后将所有p连接得到最终明文"""

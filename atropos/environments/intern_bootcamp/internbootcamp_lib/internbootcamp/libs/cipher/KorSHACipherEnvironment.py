from .BaseCipherEnvironment import BaseCipherEnvironment

import hashlib

def generate_key():
    secret = "SECRET_KEY"
    sha256 = hashlib.sha256()
    sha256.update(secret.encode('utf-8'))
    return sha256.digest()


class KorSHACipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "SHA Cipher from kor-bench"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule25_SHACipher"
    
    def encode(self, text, **kwargs):
        print("开始加密过程...")
        print(f"原始输入文本: {text}")
        
        # 处理输入文本，只保留字母并转换为大写
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        # 生成密钥
        key = generate_key()
        print(f"生成的SHA-256密钥: {key.hex()}")
        
        # 将文本转换为字节序列
        plaintext_bytes = text.encode('utf-8')
        print(f"文本转换为字节序列: {plaintext_bytes}")
        
        # 使用XOR运算加密
        print("开始XOR加密...")
        ciphertext_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(plaintext_bytes)])
        print(f"加密后的字节序列: {ciphertext_bytes}")
        
        # 转换为十六进制字符串
        result = ciphertext_bytes.hex()
        print(f"最终加密结果(十六进制): {result}")
        
        return result

    def decode(self,text, **kwargs):
        print("开始解密过程...")
        print(f"加密的十六进制文本: {text}")
        
        # 生成密钥
        key = generate_key()
        print(f"生成的SHA-256密钥: {key.hex()}")
        
        # 将十六进制转换为字节序列
        ciphertext_bytes = bytes.fromhex(text)
        print(f"十六进制转换为字节序列: {ciphertext_bytes}")
        
        # 使用XOR运算解密
        print("开始XOR解密...")
        plaintext_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(ciphertext_bytes)])
        print(f"解密后的字节序列: {plaintext_bytes}")
        
        # 转换为文本
        result = plaintext_bytes.decode('utf-8')
        print(f"最终解密结果: {result}")
        
        return result

    def get_encode_rule(self,):
        return """加密规则:
    - 输入:
        - 明文: 仅包含大写字母的字符串，不含标点符号和空格
    - 输出:
        - 密文: 十六进制字符串（包含小写字母a-e）
    - 准备:
        - 密钥(SHA哈希值)
            - 对"SECRET_KEY"执行SHA-256运算，得到'73ef2a4edd7a7fbf07fd5f6faf99674dc0c25a025fd74c221f4c35849e5c0fb3'
    - 加密步骤:
        - 将明文字符串转换为字节序列（ASCII编码）
        - 使用密钥对每个字节进行异或（XOR）运算加密。重复使用密钥使其长度与明文字节数相同
        - 将加密后的字节序列转换为十六进制字符串作为密文输出"""

    def get_decode_rule(self,):
        return """解密规则:
    - 输入:
        - 密文: 十六进制字符串（包含小写字母a-e）
    - 输出:
        - 明文: 仅包含大写字母的字符串，不含标点符号和空格
    - 准备:
        - 密钥（与加密相同，是通过SHA-256获得的哈希值）
    - 解密步骤:
        - 将密文字符串转换为字节序列
        - 使用密钥对每个字节进行异或（XOR）运算解密（解密过程与加密过程相同）
        - 将解密后的字节序列转换为明文字符串"""

        
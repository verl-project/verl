from .BaseCipherEnvironment import BaseCipherEnvironment

S_BOX = {
    0x00: 0x0F, 0x01: 0x0A, 0x02: 0x07, 0x03: 0x05,
    0x04: 0x09, 0x05: 0x03, 0x06: 0x0D, 0x07: 0x00,
    0x08: 0x0E, 0x09: 0x08, 0x0A: 0x04, 0x0B: 0x06,
    0x0C: 0x01, 0x0D: 0x02, 0x0E: 0x0B, 0x0F: 0x0C
}

INV_S_BOX = {
    0x0F: 0x00, 0x0A: 0x01, 0x07: 0x02, 0x05: 0x03,
    0x09: 0x04, 0x03: 0x05, 0x0D: 0x06, 0x00: 0x07,
    0x0E: 0x08, 0x08: 0x09, 0x04: 0x0A, 0x06: 0x0B,
    0x01: 0x0C, 0x02: 0x0D, 0x0B: 0x0E, 0x0C: 0x0F
}

KEY = b'1234567890ABCDEF'

def xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


def substitute(bytes_data, box):
    result = bytearray()
    for byte in bytes_data:
        high_nibble = (byte >> 4) & 0x0F
        low_nibble = byte & 0x0F
        substituted_high = box[high_nibble]
        substituted_low = box[low_nibble]
        combined_byte = (substituted_high << 4) | substituted_low
        result.append(combined_byte)
    return bytes(result)

def simple_permute(bytes_data):
    return bytes(((b << 1) & 0xFF) | ((b >> 7) & 0x01) for b in bytes_data)

def inverse_permute(bytes_data):
        return bytes(((b >> 1) & 0xFF) | ((b << 7) & 0x80) for b in bytes_data)


class KorSBOXCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "SBOX Cipher from kor-bench"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule22_SBOXCipher"
    
    def encode(self, text, **kwargs):
        print("开始加密过程...")
        # 将输入转换为大写字母和空格
        text = ''.join(char.upper() for char in text if char.isalnum() or char.isspace())
        print(f"1. 规范化输入文本: {text}")
        
        # 使用给定的加密代码进行加密
        print("2. 开始分块处理...")
        blocks = [text[i:i+8] for i in range(0, len(text), 8)]
        encrypted_blocks = []
        
        for i, block in enumerate(blocks):
            print(f"\n处理第{i+1}个块: {block}")
            block = block.ljust(8, '\x00')
            print(f"  - 填充后的块: {block}")
            
            state = xor_bytes(block.encode('ascii'), KEY)
            print(f"  - 与密钥XOR后: {state.hex().upper()}")
            
            state = substitute(state, S_BOX)
            print(f"  - S盒替换后: {state.hex().upper()}")
            
            state = simple_permute(state)
            print(f"  - 左移置换后: {state.hex().upper()}")
            
            state = xor_bytes(state, KEY)
            print(f"  - 最终与密钥XOR后: {state.hex().upper()}")
            
            encrypted_blocks.append(''.join(f'{b:02X}' for b in state))
        
        result = ''.join(encrypted_blocks)
        print(f"\n最终加密结果: {result}")
        return result

    def decode(self, text, **kwargs):
        print("开始解密过程...")
        print(f"1. 接收到的密文: {text}")
        
        print("2. 开始分块处理...")
        blocks = [text[i:i+16] for i in range(0, len(text), 16)]
        decrypted_blocks = []
        
        for i, block in enumerate(blocks):
            print(f"\n处理第{i+1}个块: {block}")
            
            state = bytes.fromhex(block)
            print(f"  - 转换为字节: {state.hex().upper()}")
            
            state = xor_bytes(state, KEY)
            print(f"  - 与密钥XOR后: {state.hex().upper()}")
            
            state = inverse_permute(state)
            print(f"  - 右移置换后: {state.hex().upper()}")
            
            state = substitute(state, INV_S_BOX)
            print(f"  - 逆S盒替换后: {state.hex().upper()}")
            
            state = xor_bytes(state, KEY)
            print(f"  - 最终与密钥XOR后: {state.hex().upper()}")
            
            decrypted_blocks.append(state.decode('ascii').rstrip('\x00'))
        
        result = ''.join(decrypted_blocks)
        print(f"\n最终解密结果: {result}")
        return result

    def get_encode_rule(self, ):
        return """加密规则:
    - 输入:
        - 明文: 大写字母和空格组成的字符串
    - 输出:
        - 密文: 表示加密数据的十六进制字符串(其中A-E需要大写)
    - 准备:
        - 固定密钥
            - KEY = b'1234567890ABCDEF'
        - S盒
            - S_BOX = {
            0x00: 0x0F, 0x01: 0x0A, 0x02: 0x07, 0x03: 0x05,
            0x04: 0x09, 0x05: 0x03, 0x06: 0x0D, 0x07: 0x00,
            0x08: 0x0E, 0x09: 0x08, 0x0A: 0x04, 0x0B: 0x06,
            0x0C: 0x01, 0x0D: 0x02, 0x0E: 0x0B, 0x0F: 0x0C
            }
    - 加密步骤:
        1. 填充: 如果明文长度不是8字节的倍数，用\\x00(空字符)填充使其长度成为8字节的倍数
        2. 分块: 将填充后的明文分成8字节的块
        3. 块加密:
            - 转换为字节: 使用ASCII编码将每个块转换为字节
            - 与密钥XOR: 字节块与固定密钥进行XOR运算
            - 替换: 使用S盒替换每个字节的高4位和低4位并拼接
            - 置换: 通过将每个字节左移1位进行简单置换
            - 与密钥XOR: 置换后的字节块再次与固定密钥进行XOR运算
        4. 十六进制编码: 将加密后的字节块转换为十六进制字符串
        5. 拼接: 将所有加密块的十六进制字符串拼接形成最终密文"""

    def get_decode_rule(self, ):
        return """解密规则:
    - 输入:
        - 密文: 表示加密数据的十六进制字符串(其中A-E需要大写)
    - 输出:
        - 明文: 大写字母和空格组成的字符串
    - 准备:
        - 固定密钥
            - KEY = b'1234567890ABCDEF'
        - 逆S盒
            - INV_S_BOX = {
            0x0F: 0x00, 0x0A: 0x01, 0x07: 0x02, 0x05: 0x03,
            0x09: 0x04, 0x03: 0x05, 0x0D: 0x06, 0x00: 0x07,
            0x0E: 0x08, 0x08: 0x09, 0x04: 0x0A, 0x06: 0x0B,
            0x01: 0x0C, 0x02: 0x0D, 0x0B: 0x0E, 0x0C: 0x0F
            }
    - 解密步骤:
        1. 分块: 将密文分成16字符(8字节)的块
        2. 块解密:
            - 转换为字节: 将每个块从十六进制字符串转换为字节
            - 与密钥XOR: 字节块与固定密钥进行XOR运算
            - 逆置换: 通过将每个字节右移1位进行逆置换
            - 替换: 使用逆S盒替换块中字节的高4位和低4位并拼接
            - 与密钥XOR: 字节块再次与固定密钥进行XOR运算
        3. 转换为文本: 使用ASCII解码将解密后的字节块转换回文本
        4. 移除填充: 从解密后的明文末尾移除填充字符(\\x00)
        5. 拼接: 将所有解密块拼接形成最终明文"""

    
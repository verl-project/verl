from .BaseCipherEnvironment import BaseCipherEnvironment

key = '10101010'
permutation = (2, 0, 3, 1, 4, 6, 5, 7)
inverse_permutation = (1, 3, 0, 2, 4, 6, 5, 7)

def xor(bits1, bits2):
    return ''.join(['1' if b1 != b2 else '0' for b1, b2 in zip(bits1, bits2)])

def permute(bits, perm_table):
    return ''.join([bits[i] for i in perm_table])


class KorXORCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "XOR Cipher from kor-bench"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule21_XORCipher"

    def encode(self, text, **kwargs):
        # 移除非字母字符并转为大写
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"1. 输入文本处理为: {text}")
        
        encrypted_bits = []
        for char in text:
            print(f"\n处理字符: {char}")
            binary_plaintext = format(ord(char), '08b')
            print(f"2. 转换为8位二进制: {binary_plaintext}")
            
            xor_result = xor(binary_plaintext, key * (len(binary_plaintext) // len(key) + 1))[:len(binary_plaintext)]
            print(f"3. 与密钥进行XOR运算: {xor_result}")
            
            permuted = permute(xor_result, permutation)
            print(f"4. 应用置换表: {permuted}")
            
            encrypted_bits.append(permuted)
        
        encrypted_binary_string = ''.join(encrypted_bits)
        print(f"\n最终加密结果: {encrypted_binary_string}")
        return encrypted_binary_string

    def decode(self, text, **kwargs):
        print(f"1. 接收到的加密文本: {text}")
        
        decrypted_chars = []
        num_chars = len(text) // 8
        for i in range(num_chars):
            binary_chunk = text[i*8:(i+1)*8]
            print(f"\n处理8位二进制块: {binary_chunk}")
            
            permuted_bits = permute(binary_chunk, inverse_permutation)
            print(f"2. 应用逆置换: {permuted_bits}")
            
            xor_result = xor(permuted_bits, key * (len(permuted_bits) // len(key) + 1))[:len(permuted_bits)]
            print(f"3. 与密钥进行XOR运算: {xor_result}")
            
            decrypted_char = chr(int(xor_result, 2))
            print(f"4. 转换为字符: {decrypted_char}")
            
            decrypted_chars.append(decrypted_char)
        
        result = ''.join(decrypted_chars)
        print(f"\n最终解密结果: {result}")
        return result

    def get_encode_rule(self, ):
        encode_rule = """
        加密规则:
        - 输入:
            - 明文: 仅包含大写字母(A-Z)的字符串，不含标点和空格
        - 输出:
            - 密文: 仅包含0和1的二进制字符串
        - 准备:
            - 固定密钥: 8位二进制字符串(例如'10101010')
            - 置换表:
                - 置换表: (2, 0, 3, 1, 4, 6, 5, 7)
                - 逆置换表: (1, 3, 0, 2, 4, 6, 5, 7)
        - 加密步骤:
            1. 将每个字符转换为二进制:
                - 将每个字符转换为ASCII值
                - 将ASCII值转换为8位二进制字符串
            2. XOR运算:
                - 对字符的8位二进制表示与固定密钥进行XOR运算
                - 如果需要，重复密钥以匹配二进制表示的长度
            3. 置换:
                - 对XOR结果应用置换表得到每个字符的最终加密二进制字符串
            4. 合并二进制字符串:
                - 将所有字符的二进制字符串连接形成最终密文
        """
        return encode_rule

    def get_decode_rule(self, ):
        decode_rule = """
        解密规则:
        - 输入:
            - 密文: 仅包含0和1的二进制字符串
        - 输出:
            - 明文: 仅包含大写字母(A-Z)的字符串，不含标点和空格
        - 准备:
            - 固定密钥: 与加密使用的相同8位二进制字符串(例如'10101010')
            - 置换表:
                - 置换表: (2, 0, 3, 1, 4, 6, 5, 7)
                - 逆置换表: (1, 3, 0, 2, 4, 6, 5, 7)
        - 解密步骤:
            1. 将密文分块:
                - 将二进制密文分成8位一组，每组代表一个加密字符
            2. 逆置换:
                - 对每个8位块应用逆置换表以还原加密时的置换
            3. XOR运算:
                - 对置换后的二进制块与固定密钥进行XOR运算
            4. 二进制转字符:
                - 将得到的二进制字符串转换为十进制值
                - 将十进制值转换为对应的ASCII字符
            5. 合并字符:
                - 将每个二进制块得到的字符连接形成最终明文
        """
        return decode_rule


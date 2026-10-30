from .BaseCipherEnvironment import BaseCipherEnvironment

class AlbertiCipher:
    def __init__(self, period, increment):
        self.outer_disk = "QWERTYUIOPASDFGHJZXCVBNMKL"
        self.inner_disk = "JKLZXCVBNMASDFGHJQWERTYUIO"
        self.initial_offset = 0
        self.period = period
        self.increment = increment
        self.reset_disks()

    def reset_disks(self):
        self.current_inner_disk = self.inner_disk[self.initial_offset:] + self.inner_disk[:self.initial_offset]

    def encrypt_char(self, char):
        if char in self.outer_disk:
            index = self.outer_disk.index(char)
            return self.current_inner_disk[index]
        else:
            return char

    def decrypt_char(self, char):
        if char in self.current_inner_disk:
            index = self.current_inner_disk.index(char)
            return self.outer_disk[index]
        else:
            return char

    def rotate_disk(self, increment):
        self.current_inner_disk = self.current_inner_disk[increment:] + self.current_inner_disk[:increment]

    def encrypt(self, plaintext):
        self.reset_disks()
        ciphertext = []
        for i, char in enumerate(plaintext):
            ciphertext.append(self.encrypt_char(char))
            if (i + 1) % self.period == 0:
                self.rotate_disk(self.increment)
        return ''.join(ciphertext)

    def decrypt(self, ciphertext):
        self.reset_disks()
        plaintext = []
        for i, char in enumerate(ciphertext):
            plaintext.append(self.decrypt_char(char))
            if (i + 1) % self.period == 0:
                self.rotate_disk(self.increment)
        return ''.join(plaintext)


class KorAlbertiCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule9_AlertiCipher'

    def encode(self,text, period=5, increment=4):
        # 将输入转换为大写字母，去除标点和空格
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        cipher = AlbertiCipher(period, increment)
        print(f"初始化Alberti密码盘:")
        print(f"外圈: {cipher.outer_disk}")
        print(f"内圈: {cipher.inner_disk}")
        print(f"周期: {period} (每处理{period}个字符后旋转内圈)")
        print(f"增量: {increment} (每次旋转{increment}个位置)")
        
        encoded = cipher.encrypt(text)
        print("\n加密过程:")
        for i, (p, c) in enumerate(zip(text, encoded)):
            print(f"字符 {p} 在外圈位置 {cipher.outer_disk.index(p)} 对应内圈字符 {c}")
            if (i + 1) % period == 0:
                print(f"已处理{period}个字符，内圈向右旋转{increment}个位置")
        
        print(f"\n最终加密结果: {encoded}")
        return encoded

    def decode(self, text, period=5, increment=4):
        cipher = AlbertiCipher(period, increment)
        print(f"初始化Alberti密码盘:")
        print(f"外圈: {cipher.outer_disk}")
        print(f"内圈: {cipher.inner_disk}")
        print(f"周期: {period} (每处理{period}个字符后旋转内圈)")
        print(f"增量: {increment} (每次旋转{increment}个位置)")
        
        decoded = cipher.decrypt(text)
        print("\n解密过程:")
        for i, (c, p) in enumerate(zip(text, decoded)):
            print(f"字符 {c} 在内圈位置 {cipher.current_inner_disk.index(c)} 对应外圈字符 {p}")
            if (i + 1) % period == 0:
                print(f"已处理{period}个字符，内圈向右旋转{increment}个位置")
        
        print(f"\n最终解密结果: {decoded}")
        return decoded

    def get_encode_rule(self):
        encode_rule = """
加密规则:

输入:
- 明文: 大写字母字符串，不含标点和空格
- period: 定义内圈多久旋转一次。周期性表示在加密过程中每处理指定数量的字符后，内圈将根据增量值旋转一次
- increment: 定义内圈每次旋转的字符数。在每个周期结束时，内圈将根据增量值向右旋转相应数量的字符

输出:
- 密文: 大写字母字符串

准备:
- outer_disk = "QWERTYUIOPASDFGHJZXCVBNMKL"
- inner_disk = "JKLZXCVBNMASDFGHJQWERTYUIO"

加密步骤:
- 对明文中的每个字符p:
    - 在外圈找到该字符
    - 用内圈对应位置的字符替换它
    - 每加密period个字符后，将内圈向右旋转increment个字符。例如，将'ZXCVBNMASDFGHJKLQWERTYUIOP'旋转4位得到'BNMASDFGHJKLQWERTYUIOPZXCV'
        """
        return encode_rule

    def get_decode_rule(self):
        decode_rule = """
解密规则:

输入:
- 密文: 大写字母字符串
- period (与加密相同)
- increment (与加密相同)

输出:
- 明文: 大写字母字符串

准备:
- outer_disk = "QWERTYUIOPASDFGHJZXCVBNMKL"
- inner_disk = "JKLZXCVBNMASDFGHJQWERTYUIO"

解密步骤 (与加密步骤完全相反):
- 对密文中的每个字符c:
    - 在内圈找到该字符
    - 用外圈对应位置的字符替换它
    - 每解密period个字符后，将内圈向右旋转increment个字符。例如，将'ZXCVBNMASDFGHJKLQWERTYUIOP'旋转4位得到'BNMASDFGHJKLQWERTYUIOPZXCV'
        """
        return decode_rule

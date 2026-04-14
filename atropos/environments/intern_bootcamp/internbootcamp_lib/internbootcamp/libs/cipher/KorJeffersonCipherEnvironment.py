from .BaseCipherEnvironment import BaseCipherEnvironment

class JeffersonCipher:
    def __init__(self):
        self.wheel_configuration = [
            "ABCEIGDJFVUYMHTQKZOLRXSPWN",
            "ACDEHFIJKTLMOUVYGZNPQXRWSB",
            "ADKOMJUBGEPHSCZINXFYQRTVWL",
            "AEDCBIFGJHLKMRUOQVPTNWYXZS",
            "AFNQUKDOPITJBRHCYSLWEMZVXG",
            "AGPOCIXLURNDYZHWBJSQFKVMET",
            "AHXJEZBNIKPVROGSYDULCFMQTW",
            "AIHPJOBWKCVFZLQERYNSUMGTDX",
            "AJDSKQOIVTZEFHGYUNLPMBXWCR",
            "AKELBDFJGHONMTPRQSVZUXYWIC",
            "ALTMSXVQPNOHUWDIZYCGKRFBEJ",
            "AMNFLHQGCUJTBYPZKXISRDVEWO",
            "ANCJILDHBMKGXUZTSWQYVORPFE",
            "AODWPKJVIUQHZCTXBLEGNYRSMF",
            "APBVHIYKSGUENTCXOWFQDRLJZM",
            "AQJNUBTGIMWZRVLXCSHDEOKFPY",
            "ARMYOFTHEUSZJXDPCWGQIBKLNV",
            "ASDMCNEQBOZPLGVJRKYTFUIWXH",
            "ATOJYLFXNGWHVCMIRBSEKUPDZQ",
            "AUTRZXQLYIOVBPESNHJWMDGFCK",
            "AVNKHRGOXEYBFSJMUDQCLZWTIP",
            "AWVSFDLIEBHKNRJQZGMXPUCOTY",
            "AXKWREVDTUFOYHMLSIQNJCPGBZ",
            "AYJPXMVKBQWUGLOSTECHNZFRID",
            "AZDNBUHYFWJLVGRCQMPSOEXTKI"
        ]
    
    def encrypt(self, message):
        encrypted_message = []
        wheel_position = 0
        
        for char in message:
            current_wheel = self.wheel_configuration[wheel_position]
            index = current_wheel.index(char)
            encrypted_char = current_wheel[(index + 1) % 26]
            encrypted_message.append(encrypted_char)
            wheel_position = (wheel_position + 1) % len(self.wheel_configuration)
        
        return ''.join(encrypted_message)
    
    def decrypt(self, encrypted_message):
        decrypted_message = []
        wheel_position = 0
        
        for char in encrypted_message:
            current_wheel = self.wheel_configuration[wheel_position]
            index = current_wheel.index(char)
            decrypted_char = current_wheel[(index - 1) % 26]
            decrypted_message.append(decrypted_char)
            wheel_position = (wheel_position + 1) % len(self.wheel_configuration)
        
        return ''.join(decrypted_message)


class KorJeffersonCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule10_JeffersonCipher'
    
    def encode(self, text, **kwargs):
        # 将输入转换为大写字母并移除非字母字符
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        cipher = JeffersonCipher()
        print("初始化Jefferson密码轮...")
        
        print("开始加密过程:")
        print("- 从第1个密码轮开始")
        encrypted = cipher.encrypt(text)
        print(f"- 对每个字符:")
        for i, (plain, cipher) in enumerate(zip(text, encrypted)):
            wheel_num = (i % 25) + 1
            print(f"  * 在第{wheel_num}个密码轮上，将字符 {plain} 替换为下一个字符 {cipher}")
        
        print(f"加密完成，结果: {encrypted}")
        return encrypted

    def decode(self, text, **kwargs):
        cipher = JeffersonCipher()
        print("初始化Jefferson密码轮...")
        
        print("开始解密过程:")
        print("- 从第1个密码轮开始")
        decrypted = cipher.decrypt(text)
        print(f"- 对每个字符:")
        for i, (cipher, plain) in enumerate(zip(text, decrypted)):
            wheel_num = (i % 25) + 1
            print(f"  * 在第{wheel_num}个密码轮上，将字符 {cipher} 替换为前一个字符 {plain}")
        
        print(f"解密完成，结果: {decrypted}")
        return decrypted

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 仅包含大写字母的字符串，不含标点和空格
- 输出:
    - 密文: 大写字母字符串
- 准备:
    - 25个密码轮，每个密码轮包含26个字母的不同排列
    [
            "ABCEIGDJFVUYMHTQKZOLRXSPWN",
            "ACDEHFIJKTLMOUVYGZNPQXRWSB",
            "ADKOMJUBGEPHSCZINXFYQRTVWL",
            "AEDCBIFGJHLKMRUOQVPTNWYXZS",
            "AFNQUKDOPITJBRHCYSLWEMZVXG",
            "AGPOCIXLURNDYZHWBJSQFKVMET",
            "AHXJEZBNIKPVROGSYDULCFMQTW",
            "AIHPJOBWKCVFZLQERYNSUMGTDX",
            "AJDSKQOIVTZEFHGYUNLPMBXWCR",
            "AKELBDFJGHONMTPRQSVZUXYWIC",
            "ALTMSXVQPNOHUWDIZYCGKRFBEJ",
            "AMNFLHQGCUJTBYPZKXISRDVEWO",
            "ANCJILDHBMKGXUZTSWQYVORPFE",
            "AODWPKJVIUQHZCTXBLEGNYRSMF",
            "APBVHIYKSGUENTCXOWFQDRLJZM",
            "AQJNUBTGIMWZRVLXCSHDEOKFPY",
            "ARMYOFTHEUSZJXDPCWGQIBKLNV",
            "ASDMCNEQBOZPLGVJRKYTFUIWXH",
            "ATOJYLFXNGWHVCMIRBSEKUPDZQ",
            "AUTRZXQLYIOVBPESNHJWMDGFCK",
            "AVNKHRGOXEYBFSJMUDQCLZWTIP",
            "AWVSFDLIEBHKNRJQZGMXPUCOTY",
            "AXKWREVDTUFOYHMLSIQNJCPGBZ",
            "AYJPXMVKBQWUGLOSTECHNZFRID",
            "AZDNBUHYFWJLVGRCQMPSOEXTKI"
        ]
- 加密步骤:
    - 初始选择第1个密码轮
    - 对明文中的每个字符p:
        - 在当前密码轮上找到字符p，用其后一个字符替换得到密文字符
        - 如果当前字符在密码轮末尾，则回到密码轮开头
        - 移动到下一个密码轮处理下一个字符，当到达最后一个密码轮时，返回第一个密码轮继续加密过程"""

    def get_decode_rule(self, ):
        return """解密规则:
- 输入:
    - 密文: 大写字母字符串
- 输出:
    - 明文: 大写字母字符串
- 准备:
    - 25个密码轮，与加密相同
- 解密步骤(与加密步骤相反):
    - 初始选择第1个密码轮
    - 对密文中的每个字符c:
        - 在当前密码轮上找到字符c，用其前一个字符替换得到明文字符
        - 如果当前字符在密码轮开头，则回到密码轮末尾
        - 移动到下一个密码轮处理下一个字符，当到达最后一个密码轮时，返回第一个密码轮继续解密过程"""

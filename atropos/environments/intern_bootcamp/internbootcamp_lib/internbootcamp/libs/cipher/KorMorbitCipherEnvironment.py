from .BaseCipherEnvironment import BaseCipherEnvironment

class MorseCodeCipher:
    def __init__(self, key):
        self.key = key
        self.index_mapping = self.generate_index_mapping()
        self.reverse_mapping = {v: k for k, v in self.index_mapping.items()}
    
    def generate_index_mapping(self):
        sorted_key = sorted(self.key)
        index_mapping = {
            '..': sorted_key.index(self.key[0]) + 1,
            '.-': sorted_key.index(self.key[1]) + 1,
            './': sorted_key.index(self.key[2]) + 1,
            '-.': sorted_key.index(self.key[3]) + 1,
            '--': sorted_key.index(self.key[4]) + 1,
            '-/': sorted_key.index(self.key[5]) + 1,
            '/.': sorted_key.index(self.key[6]) + 1,
            '/-': sorted_key.index(self.key[7]) + 1,
            '//': sorted_key.index(self.key[8]) + 1,
        }
        return index_mapping
    
    def encrypt(self, plaintext):
        morse_code = plaintext
        encrypted_numbers = []
        
        pairs = [morse_code[i:i+2] for i in range(0, len(morse_code), 2)]
        for pair in pairs:
            if len(pair) % 2 == 0:
                index = self.index_mapping[pair]
                encrypted_numbers.append(str(index))
            else:
                encrypted_numbers.append(pair)
        
        encrypted_message = ''.join(encrypted_numbers)
        return encrypted_message
    
    def decrypt(self, encrypted_message):
        decrypted_numbers = []
    
        for char in encrypted_message:
            if char.isdigit():
                index = int(char)
                decrypted_numbers.append(self.reverse_mapping[index])
            else:
                decrypted_numbers.append(char)
        
        decrypted_morse = ''.join(decrypted_numbers)
        return decrypted_morse
    
    def text_to_morse(self, text):
        morse_code = {
            'A': '.-',     'B': '-...',   'C': '-.-.',   'D': '-..',
            'E': '.',      'F': '..-.',   'G': '--.',    'H': '....',
            'I': '..',     'J': '.---',   'K': '-.-',    'L': '.-..',
            'M': '--',     'N': '-.',     'O': '---',    'P': '.--.',
            'Q': '--.-',   'R': '.-.',    'S': '...',    'T': '-',
            'U': '..-',    'V': '...-',   'W': '.--',    'X': '-..-',
            'Y': '-.--',   'Z': '--..',
        }
        
        morse_chars = []
        words = text.split(' ')
        
        for word in words:
            chars = []
            for char in word:
                if char.upper() in morse_code:
                    chars.append(morse_code[char.upper()])
            morse_chars.append('/'.join(chars))
        
        return '//'.join(morse_chars)
    
    def morse_to_text(self, morse_code):
        morse_code = morse_code.split('//')
        morse_to_char = {
            '.-': 'A',     '-...': 'B',   '-.-.': 'C',   '-..': 'D',
            '.': 'E',      '..-.': 'F',   '--.': 'G',    '....': 'H',
            '..': 'I',     '.---': 'J',   '-.-': 'K',    '.-..': 'L',
            '--': 'M',     '-.': 'N',     '---': 'O',    '.--.': 'P',
            '--.-': 'Q',   '.-.': 'R',    '...': 'S',    '-': 'T',
            '..-': 'U',    '...-': 'V',   '.--': 'W',    '-..-': 'X',
            '-.--': 'Y',   '--..': 'Z',
        }
        morse_chars = []
        for word in morse_code:
            chars = []
            for char in word.split('/'):
                chars.append(morse_to_char[char])
            morse_chars.append(''.join(chars))
        return ' '.join(morse_chars)


class KorMorbitCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule12_MorbitCipher'
    
    def encode(self, text, **kwargs):
        print("加密步骤开始:")
        # 将输入转换为大写字母，去除标点和空格
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"1. 将输入文本标准化为大写字母: {text}")
        
        # 初始化密码器
        cipher = MorseCodeCipher("123456789")
        
        # 转换为摩斯密码
        morse_code = cipher.text_to_morse(text)
        print(f"2. 将文本转换为摩斯密码: {morse_code}")
        
        # 加密
        encrypted = cipher.encrypt(morse_code)
        print(f"3. 根据数字索引映射表将摩斯密码对转换为数字: {encrypted}")
        
        return encrypted

    def decode(self, text, **kwargs):
        print("解密步骤开始:")
        print(f"1. 接收到的加密文本: {text}")
        
        # 初始化密码器
        cipher = MorseCodeCipher("123456789")
        
        # 解密为摩斯密码
        morse_code = cipher.decrypt(text)
        print(f"2. 将数字转换回摩斯密码: {morse_code}")
        
        # 转换为明文
        decrypted = cipher.morse_to_text(morse_code)
        print(f"3. 将摩斯密码转换为原文: {decrypted}")
        
        return decrypted

    def get_encode_rule(self, ):
        encode_rule = """
加密规则:
- 输入:
    - 明文: 大写字母字符串，不含标点和空格
- 输出:
    - 密文: 字符串
- 准备:
    - 数字索引映射表
        - '..' : 5
        - '.-' : 4
        - './' : 9
        - '-.' : 8
        - '--' : 6
        - '-/' : 7
        - '/.' : 3
        - '/-' : 1
        - '//' : 2
    - 摩斯密码表
        - A: '.-',     B: '-...',   C: '-.-.',   D: '-..',
        - E: '.',      F: '..-.',   G: '--.',    H: '....',
        - I: '..',     J: '.---',   K: '-.-',    L: '.-..',
        - M: '--',     N: '-.',     O: '---',    P: '.--.',
        - Q: '--.-',   R: '.-.',    S: '...',    T: '-',
        - U: '..-',    V: '...-',   W: '.--',    X: '-..-',
        - Y: '-.--',   Z: '--..'
- 加密步骤:
    - 根据摩斯密码表将明文中的每个字符转换为摩斯密码，用/分隔每个字符，例如AB对应'.-/-...'
    - 将摩斯密码分成两个字符一组。如果摩斯密码长度为奇数，最后一个字符不进行映射，直接添加到密文末尾
    - 根据数字索引映射表将每组字符转换为对应的数字字符串
    - 加密后的消息用字符串表示
"""
        return encode_rule

    def get_decode_rule(self, ):
        decode_rule = """
解密规则:
- 输入:
    - 密文: 数字字符串
- 输出:
    - 明文: 大写字母字符串
- 准备:
    - 数字索引映射表(与加密相同)
    - 摩斯密码表(与加密相同)
- 解密步骤(与加密步骤相反):
    - 根据数字索引映射表将密文中的每个数字转换为对应的字符对。如果密文末尾有非数字字符，则不处理。此时获得完整的摩斯密码
    - 通过/分隔摩斯密码获得每个字符的摩斯密码
    - 根据摩斯密码表将每个字符的摩斯密码转换为对应的明文字符
    - 最终明文字符为大写字符串
"""
        return decode_rule
    
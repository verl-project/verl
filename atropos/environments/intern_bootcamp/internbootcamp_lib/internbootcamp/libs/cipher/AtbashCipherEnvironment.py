from .BaseCipherEnvironment import BaseCipherEnvironment

class AtbashCipher:
    """实现 Atbash 密码的类，用于字符串转换。

    Atbash 密码是一种简单的替换密码，其中每个字母都被其在字母表中的反向对应字母所替换。
    """

    def __init__(self):
        """初始化 AtbashCipher 对象。"""
        pass

    def transform(self, text: str) -> str:
        """使用 Atbash 密码转换输入文本。

        Args:
            text (str): 要转换的输入文本。

        Returns:
            str: 应用 Atbash 密码后转换的文本。
        """
        result = []  # 存储转换后的字符
        words = text.split()  # 将文本拆分成单词

        # 逐个处理每个单词
        for word in words:
            print(f"转换单词: {word}")
            transformed_word = []  # 存储单词中转换后的字符

            # 转换单词中的每个字符
            for char in word:
                if char.isalpha():  # 检查字符是否为字母
                    if char.islower():
                        transformed_char = chr(219 - ord(char))  # 小写字母转换
                    else:
                        transformed_char = chr(155 - ord(char))  # 大写字母转换
                    print(f"  '{char}' -> '{transformed_char}'")
                else:
                    transformed_char = char  # 非字母字符保持不变
                    print(f"  '{char}' 保持不变")
                
                transformed_word.append(transformed_char)

            # 将单词中转换后的字符重新组合并添加到结果中
            result.append(''.join(transformed_word))

        # 将转换后的单词重新组合成句子并返回
        return ' '.join(result)


class AtbashCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
    
    @property
    def cipher_name(self) -> str:
        """返回密码的名称。
        """
        return "AtbashCipher"
    def encode(self, text: str, **kwargs) -> str:
        """使用 Atbash 密码对输入文本进行编码。

        Args:
            text (str): 要编码的输入文本。

        Returns:
            str: 应用 Atbash 密码后编码的文本。
        """
        print("编码过程:")
        cipher = AtbashCipher()
        encode_text = cipher.transform(text)
        print(f"编码后的文本: {encode_text}")
        return encode_text

    def decode(self, text: str, **kwargs) -> str:
        """使用 Atbash 密码对输入文本进行解码。

        Args:
            text (str): 要解码的输入文本。

        Returns:
            str: 应用 Atbash 密码后解码的文本。
        """
        print("解码过程:")
        cipher = AtbashCipher()
        decode_text = cipher.transform(text)
        print(f"解码后的文本: {decode_text}")
        return decode_text

    def get_encode_rule(self, ) -> str:
        """返回编码规则的自然语言描述。

        Returns:
            str: 编码规则的自然语言描述。
        """
        return "Atbash 密码的编码规则是将每个字母替换为其在字母表中的反向对应字母。例如，'A' 变成 'Z'，'B' 变成 'Y'，依此类推。非字母字符保持不变。"

    def get_decode_rule(self, ) -> str:
        """返回解码规则的自然语言描述。

        Returns:
            str: 解码规则的自然语言描述。
        """
        return "Atbash 密码的解码规则是将每个字母替换为其在字母表中的反向对应字母。由于 Atbash 密码是对称的，解码过程与编码过程相同。例如，'Z' 变成 'A'，'Y' 变成 'B'，依此类推。非字母字符保持不变。"

        
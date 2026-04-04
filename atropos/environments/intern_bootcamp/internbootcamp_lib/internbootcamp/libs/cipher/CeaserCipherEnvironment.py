from .BaseCipherEnvironment import BaseCipherEnvironment


class CeaserCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    def encode(self, text, shift):
        """
        编码函数：将输入文本中的字母向后移动指定的位数（shift）

        参数：
        text: 需要加密的文本（字符串）
        shift: 移动的位数（整数）

        返回：
        加密后的文本（字符串）
        """
        print(f"开始编码过程，文本为: {text}，位移数为: {shift}")
        result = []  # 用于存储加密后的字符
        for char in text:
            if char.isalpha():  # 判断是否是字母
                print(f"处理字符: {char}")
                start = ord('A') if char.isupper() else ord('a')  # 计算字母的起始ASCII码
                new_char = chr((ord(char) - start + shift) % 26 + start)  # 移动并保持在字母表范围内
                print(f"字符 {char} 向后移动 {shift} 位，变为 {new_char}")
                result.append(new_char)  # 添加加密后的字母
            else:
                print(f"字符 {char} 不是字母，保持不变")
                result.append(char)  # 非字母字符不做改变，直接添加到结果中
        encoded_text = ''.join(result)  # 将列表转化为字符串
        print(f"编码完成，结果为: {encoded_text}")
        return encoded_text
    
    def decode(self, text, shift):
        """
        解码函数：将输入文本中的字母向前移动指定的位数（shift），以还原原始文本

        参数：
        text: 需要解密的文本（字符串）
        shift: 移动的位数（整数）

        返回：
        解密后的文本（字符串）
        """
        print(f"开始解码过程，文本为: {text}，位移数为: {shift}")
        result = []  # 用于存储解密后的字符
        for char in text:
            if char.isalpha():  # 判断是否是字母
                print(f"处理字符: {char}")
                start = ord('A') if char.isupper() else ord('a')  # 计算字母的起始ASCII码
                new_char = chr((ord(char) - start - shift) % 26 + start)  # 向前移动并保持在字母表范围内
                print(f"字符 {char} 向前移动 {shift} 位，变为 {new_char}")
                result.append(new_char)  # 添加解密后的字母
            else:
                print(f"字符 {char} 不是字母，保持不变")
                result.append(char)  # 非字母字符不做改变，直接添加到结果中
        decoded_text = ''.join(result)  # 将列表转化为字符串
        print(f"解码完成，结果为: {decoded_text}")
        return decoded_text

    def get_encode_rule(self) -> str:
        return """加密规则：
1. 将输入文本中的每个字母向后移动指定的位数（shift）。
2. 输出加密后的文本。"""

    def get_decode_rule(self) -> str:
        return """解码规则：
1. 将输入文本中的每个字母向前移动指定的位数（shift）。
2. 输出解密后的文本。"""
    
    @property        
    def cipher_name(self) -> str:
        return "Caesar_Cipher"
    def get_question(self, is_gen) -> str:
        if is_gen:
            return self.get_encode_rule() + f"Generate a ciphertext using a Caesar cipher with a shift of {self.shift}."
        else:
            return self.get_decode_rule+ f"decode the following ciphertext using a Caesar cipher with a shift of {self.shift}: {self.ciphertext}"
    def get_hint(self) -> str:
        """
        提供关于如何解决问题的提示信息。
        """
        return "Try to think about how letters can be shifted in the alphabet."

    def get_additional_resources(self) -> list:
        """
        返回可用于解决问题的额外资源列表。
        """
        return ["https://en.wikipedia.org/wiki/Caesar_cipher"]
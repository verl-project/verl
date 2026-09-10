from .BaseCipherEnvironment import BaseCipherEnvironment

class Asc2Environment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
    
    @property
    def cipher_name(self):
        return "ASCII"
    def encode(self, text):
        # 初始化一个空列表，用于存储ASCII编码
        ascii_list = []
        # 遍历输入字符串中的每个字符
        for char in text:
            # 获取字符的ASCII编码
            ascii_code = ord(char)
            # 将ASCII编码添加到列表中
            ascii_list.append(ascii_code)
            # 打印字符及其对应的ASCII编码
            print(f"Character:   {char}, ASCII Code: {ascii_code}")
            # print(f"Current ASCII List: {ascii_list}\n")
        print("ASCII List:", ascii_list)
        # 返回ASCII编码列表
        return ascii_list

    def get_encode_rule(self, ):
        return """加密方案概述：将字符串转换为ASCII编码。"""


    def decode(self, text):
        """
        将ASCII编码列表转换为字符串，并打印每个ASCII码及其对应的字符。
        :param ascii_list: 包含ASCII编码的列表
        :return: 转换后的字符串
        """

        result = ""
        for ascii_code in text:
            # 检查ASCII码是否在可打印字符的范围内
            if 32 <= ascii_code <= 126:
                # 将ASCII码转换为字符，并添加到结果字符串中
                result += chr(ascii_code)
                # 打印ASCII码及其对应的字符
                print(f"ASCII   Code: {ascii_code}, Character: {chr(ascii_code)}")
            else:
                # 如果ASCII码不在可打印字符的范围内，打印警告信息
                print(f"Warning:   ASCII Code {ascii_code} is not a printable character.")
        print("Result:", result)
        return result

    def get_decode_rule(self, ):
        return """解密方案概述：遍历ASCII编码字符串,返回转换后的字符串"""
 
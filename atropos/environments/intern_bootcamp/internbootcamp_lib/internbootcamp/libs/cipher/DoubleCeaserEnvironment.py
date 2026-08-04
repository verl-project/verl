from .BaseCipherEnvironment import BaseCipherEnvironment
import random


class DoubleCeaserEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "对于每一个编码前的字母，计算其在字母表中的位置，然后找到两个可能的原始字母位置，这些位置的平均值等于编码后的字母位置。选择一个合理的字母对作为编码结果。"
        self.problem_description = problem_description
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "DoubleCeaser"

    def decode(self, text, **kwargs):
        # 将文本转换为大写以保持一致性
        text = text.upper()
        encoded_text = ""
        letters = text
        print
        print("开始解码过程：")
        print("密文： ",text)
        i = 0
        pair = ""
        print("去除非字母字符得到字母对: ", end='')
        letters = ''.join(filter(str.isalpha, letters))
        for i in range(0, len(letters), 2):
            pair = letters[i] + letters[i + 1]
            print(pair, end=' ')
            i += 2
        print('\n')
        i = 0
        pair = ""
        while i < len(text):
            # 若字母对里有非英文字符，则将该字符添加到解码结果中，然后将字母对里面的字母与下一个字母拼接得到字母对
            if not text[i].isalpha():
                encoded_text += text[i]
                print(f"添加非英文字符 {text[i]} 到解码结果\n")
                i += 1
                continue
            # 获取字母对
            pair += text[i]
            if len(pair) < 2:
                i += 1
                continue
            else:
                i += 1
            print(f"处理字母对: {pair}")
            

        
            # 计算每个字母的位置
            position1 = ord(pair[0]) - ord('A') + 1
            position2 = ord(pair[1]) - ord('A') + 1
            print(f"字母 {pair[0]} 的位置: {position1}, 字母 {pair[1]} 的位置: {position2}")
            
            # 计算位置平均值
            average_position = (position1 + position2) // 2
            print(f"计算平均位置 = {average_position}")
            
            # 转换回字母
            encoded_char = chr(average_position + ord('A') - 1)
            print(f"转换成字母: {encoded_char}\n")
            
            # 添加到结果中
            encoded_text += encoded_char
            # print(f"当前解码结果: {encoded_text}\n")

            pair = ""
        print(f"解码结果: {encoded_text}\n")
        
        return encoded_text

    def encode(self, text, **kwargs):
        # 将文本转换为大写以保持一致性
        text = text.upper()
        decoded_text = ""
        
        print("开始编码过程：")
        print(f"明文：{text}\n")
        for char in text:
            # 计算原始字母的位置
            position = ord(char) - ord('A') + 1
            print(f"处理字符: {char}, 位置: {position}")
            if position < 0 or position > 26:
                print(f"\n字符 {char} 不在字母表中，保持不变\n")
                decoded_text += char
            # 因为我们不知道原始字母的确切位置，我们假设它们是对称的
            # 这里我们使用两种可能的组合来恢复原始字母
            original_positions = [(position + randomint, position - randomint) for randomint in range(-6, 7)]
            # 乱序位置
            original_positions = random.sample(original_positions, len(original_positions))
            # 恢复原始字母
            for pos1, pos2 in original_positions:
                if 1 <= pos1 <= 26 and 1 <= pos2 <= 26:
                    original_pair = chr(pos1 + ord('A') - 1) + chr(pos2 + ord('A') - 1)
                    print(f"编码成基于该位置对称字母对: {original_pair} (位置: {pos1} 和 {pos2})\n")
                    decoded_text += original_pair
                    break
            
            # print(f"当前编码结果: {decoded_text}\n")
        print("解码完成。结果:", decoded_text)
        
        return decoded_text

    def get_decode_rule(self, ):
        return "对于每一对字母，计算它们在字母表中的位置，然后取这两个位置的平均值，再转换成对应的字母。"

    def get_encode_rule(self, ):
        return "对于每一个编码前的字母，计算其在字母表中的位置，然后找到两个可能的原始字母位置，这些位置的平均值等于编码后的字母位置。选择一个合理的字母对作为编码结果。"

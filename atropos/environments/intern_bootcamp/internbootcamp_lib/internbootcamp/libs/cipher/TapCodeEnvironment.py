from .BaseCipherEnvironment import BaseCipherEnvironment

class TapCodeEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
    
    @property
    def cipher_name(self) -> str:
        return "TapCode"
    
    def encode(self, text):
        # 定义Tap Code字母表，C和K合并
        tap_code_table = [['A', 'B', 'C', 'D', 'E'],
                        ['F', 'G', 'H', 'I', 'J'],
                        ['L', 'M', 'N', 'O', 'P'],
                        ['Q', 'R', 'S', 'T', 'U'],
                        ['V', 'W', 'X', 'Y', 'Z']]

        # 创建字母表索引的字典，方便快速查找
        letter_to_position = {}
        for i in range(len(tap_code_table)):
            for j in range(len(tap_code_table[i])):
                letter_to_position[tap_code_table[i][j]] = (i + 1, j + 1)

        # 将待加密的文字全部转换为大写，C和K视为同一个字母
        text = text.upper().replace('K', 'C')
        encrypted_message = []
        step_counter = 1

        # 对输入文本进行逐个字母的加密
        words = text.split()
        for word in words:
            encrypted_word = []
            for char in word:
                if char in letter_to_position:
                    row, col = letter_to_position[char]
                    encrypted_word.append('.' * row + ' ' + '.' * col)
                    print(
                        f"步骤 {step_counter}：正在加密字母 '{char}'，它位于第 {row} 行第 {col} 列，编码为 '{'.' * row} {'.' * col}'")
                    step_counter += 1
            encrypted_message.append('  '.join(encrypted_word))
        print(f"最终步骤：加密完成，加密后的消息是：{'  '.join(encrypted_message)}")

        # 使用 / 分隔不同单词的tap code
        return '  /  '.join(encrypted_message)


    def decode(self, text):
        # 定义Tap Code字母表，C和K合并
        tap_code_table = [['A', 'B', 'C', 'D', 'E'],
                        ['F', 'G', 'H', 'I', 'J'],
                        ['L', 'M', 'N', 'O', 'P'],
                        ['Q', 'R', 'S', 'T', 'U'],
                        ['V', 'W', 'X', 'Y', 'Z']]

        # 将tap code拆分为各个单词的编码
        words = text.split('  /  ')
        decrypted_message = []
        step_counter = 1

        # 对每个单词的编码进行解码
        for word in words:
            if word == '':
                continue
            letters = word.split('  ')
            for pair in letters:
                row_dots, col_dots = pair.split(' ')
                row = len(row_dots)
                col = len(col_dots)
                decrypted_char = tap_code_table[row - 1][col - 1]
                decrypted_message.append(decrypted_char)
                print(f"步骤 {step_counter}：正在解码 '{pair}'，它表示第 {row} 行第 {col} 列，对应的字母是 '{decrypted_char}'")
                step_counter += 1
            decrypted_message.append(' ')
        print(f"最终步骤：解码完成，解码后的消息是：{''.join(decrypted_message)}")

        # 将解码后的字母组合成完整字符串
        return ''.join(decrypted_message).strip()


    def get_encode_rule(self):
        return "Tap Code 是一种简单的密码技术，每个字母由一个点模式表示。模式通过计算行和列中的点数来编码。例如，字母 'A' 由第一行和第一列的一个点表示。"


    def get_decode_rule(self):
        return "要解码 Tap Code，需要计算每一行和每一列中的点数。通过查找表格中的位置，可以确定对应的字母。"
    
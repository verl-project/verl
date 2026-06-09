from .BaseCipherEnvironment import BaseCipherEnvironment

class KorRedefenceFigureCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "Redefence Figure Cipher from kor"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule16_RedefenceFigureCipher"
    
    def encode(self, text, **kwargs):
        print("加密步骤开始:")
        # 将输入转换为大写字母，去除空格和标点
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"1. 处理输入文本为大写字母: {text}")
        
        key = "ABCDE"
        textlen = len(text)
        rows = len(key)
        print(f"2. 准备{rows}行矩阵用于填充")
        
        matrix = [['' for _ in range(textlen)] for _ in range(rows)]
        dict = {}
        direction = True
        finish = False
        
        index = 0
        cindex = 0
        matrix[0][index] = text[cindex]
        print(f"3. 在第一行第一个位置填入第一个字母: {text[cindex]}")
        cindex += 1
        
        print("4. 开始交替向下和向上填充字母:")
        while(cindex < len(text)):
            if direction:
                print("  向下填充:")
                for i in range(1, rows):
                    if i == 1 and cindex != 1:
                        matrix[i-1][index] = '#'
                        print(f"    在第{i}行填入#")
                    if cindex == len(text) - 1:
                        matrix[i][index] = text[cindex]
                        print(f"    在第{i+1}行填入最后一个字母: {text[cindex]}")
                        finish = True
                        break
                    elif i == rows-1:
                        matrix[i][index] = text[cindex]
                        print(f"    在第{i+1}行填入: {text[cindex]}")
                        cindex += 1
                        index += 1
                        direction = False
                    else:
                        matrix[i][index] = text[cindex]
                        print(f"    在第{i+1}行填入: {text[cindex]}")
                        cindex += 1
            else:
                print("  向上填充:")
                for k in range(rows-2, -1, -1):
                    if k == rows-2:
                        matrix[k+1][index] = '#'
                        print(f"    在第{k+2}行填入#")
                    if cindex == len(text)-1:
                        matrix[k][index] = text[cindex]
                        print(f"    在第{k+1}行填入最后一个字母: {text[cindex]}")
                        finish = True
                        break
                    elif k == 0:
                        matrix[k][index] = text[cindex]
                        print(f"    在第{k+1}行填入: {text[cindex]}")
                        cindex += 1
                        index += 1
                        direction = True
                    else:
                        matrix[k][index] = text[cindex]
                        print(f"    在第{k+1}行填入: {text[cindex]}")
                        cindex += 1
            if finish:
                break
        
        print("\n5. 最终矩阵:")
        for row in matrix:
            print('  ' + ' '.join([c if c != '' else '_' for c in row]))
        
        print("\n6. 按行读取并添加*号分隔符")
        index = 0
        for char in key:
            dict[char] = matrix[index]
            index += 1
        myKeys = list(dict.keys())
        myKeys.sort()
        sorted_dict = {i: dict[i] for i in myKeys}
        result = ""
        for v in sorted_dict.values():
            for i in range(len(v)):
                if i != len(v)-1:
                    result += v[i]
                elif i == len(v)-1:
                    result += "*"
                    continue
        print(f"最终密文: {result}")
        return result

    def decode(self, text, **kwargs):
        print("解密步骤开始:")
        print(f"1. 接收到的密文: {text}")
        
        key = "ABCDE"
        textlen = len(text)
        rows = len(key)
        rail = [['' for _ in range(textlen)] for _ in range(rows)]
        
        print("2. 根据*号分割密文")
        segments = text.split('*')
        print(f"  分割得到{len(segments)}个部分")
        
        print("3. 将分割后的内容填入矩阵:")
        for i, segment in enumerate(segments):
            print(f"  第{i+1}行: {segment}")
            for j, char in enumerate(segment):
                rail[i][j] = char
        
        print("\n4. 交替读取矩阵内容:")
        decrypted_message = []
        for i in range(textlen):
            if i % 2 == 0:
                print(f"  向下读取第{i+1}列")
                for row in range(rows):
                    decrypted_message.append(rail[row][i])
            else:
                print(f"  向上读取第{i+1}列")
                for row in range(rows - 1, -1, -1):
                    decrypted_message.append(rail[row][i])
        
        print("5. 移除#号获得最终明文")
        decrypted_text = ''.join(char for char in decrypted_message if char != '#')
        print(f"解密结果: {decrypted_text}")
        return decrypted_text

    def get_encode_rule(self, ):
        encode_rule = """
加密规则:
- 输入:
    - 明文: 不含标点和空格的大写字母字符串
- 输出:
    - 密文: 不含标点和空格的字符串
- 准备:
    - 行数: 5
- 加密步骤:
    - 在第一行第一个位置填入第一个明文字母
    - 两种填充方式:
        - 向下填充: 在第一行填入"#"(除去第一列，因为第一个位置已经填入明文字母)，然后从第二行到最后一行(第五行)向下填充明文
        - 向上填充: 从最后一行(第五行)到第二行向上填充明文，然后在第一行填入"#"
    - 对于明文中的每个字母(除了已经填在第一个位置的第一个字母)，先进行向下填充，填满一列，然后转向上填充，再转向下填充，如此交替进行，直到所有字母都被填入
    - 填写完成后，逐行读取，读取每行内容后都添加一个*号，标记行的结束；然后读取第二行内容，依此类推，读取所有行，形成最终密文
"""
        return encode_rule

    def get_decode_rule(self, ):
        decode_rule = """
解密规则:
- 输入:
    - 密文: 不含标点和空格的字符串
- 输出:
    - 明文: 不含标点和空格的大写字母字符串
- 准备:
    - 行数: 5
- 解密步骤:
    - 根据密文中的*(不包括*号)，可以分成五组，依次填入五行。得到恢复的五行数据
    - 然后按照先向下读再向上读交替的方式读取所有列。得到未清理的消息
    - 从未清理的消息中删除#，得到最终明文
"""
        return decode_rule

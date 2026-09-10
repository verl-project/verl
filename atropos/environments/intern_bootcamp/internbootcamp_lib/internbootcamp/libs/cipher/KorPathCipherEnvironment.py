from .BaseCipherEnvironment import BaseCipherEnvironment

class KorPathCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) ->  str:
        return 'Kor_rule17_PathCipher'

    def encode(self, text, **kwargs):
        # 将输入转换为大写字母并去除非字母字符
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"1. 输入文本转换为大写并去除非字母字符: {text}")
        
        # 计算行数和列数
        cols = 5
        text_length = len(text)
        rows = (text_length + cols - 1) // cols
        print(f"2. 确定网格大小: {rows}行 x {cols}列")
        
        # 创建空网格
        grid = [[' ' for _ in range(cols)] for _ in range(rows)]
        
        # 按照特殊方式填充网格
        index = 0
        print("3. 开始填充网格:")
        for i in range(rows):
            if i % 2 == 0:
                print(f"   第{i+1}行从左到右填充:", end=" ")
                for j in range(cols):
                    if index < len(text):
                        grid[i][j] = text[index]
                        print(text[index], end=" ")
                        index += 1
                print()
            else:
                print(f"   第{i+1}行从右到左填充:", end=" ")
                for j in range(cols - 1, -1, -1):
                    if index < len(text):
                        grid[i][j] = text[index]
                        print(text[index], end=" ")
                        index += 1
                print()
        
        # 按列读取并添加#号
        cipher_text = []
        print("4. 按列读取并添加#号:")
        for j in range(cols):
            print(f"   第{j+1}列:", end=" ")
            for i in range(rows):
                char = grid[i][j]
                if (i == rows - 1) and char != ' ':
                    cipher_text.append(char)
                    cipher_text.append("#")
                    print(char, end="")
                    break
                elif char == ' ':
                    cipher_text.append("#")
                    print("#", end="")
                    break
                else:
                    cipher_text.append(char)
                    print(char, end="")
            print("#")
        
        result = ''.join(cipher_text)
        print(f"5. 最终密文: {result}")
        return result

    def decode(self, text, **kwargs):
        print(f"1. 收到密文: {text}")
        
        # 分割密文
        cols = 5
        parts = text.split('#')
        parts = parts[:-1]  # 移除最后一个空元素
        rows = max(len(part) for part in parts)
        print(f"2. 将密文分割成{cols}列: {parts}")
        
        # 创建网格并填充
        grid = [[' ' for _ in range(cols)] for _ in range(rows)]
        print("3. 按列填充网格:")
        for j in range(cols):
            print(f"   第{j+1}列:", end=" ")
            for i in range(len(parts[j])):
                grid[i][j] = parts[j][i]
                print(parts[j][i], end=" ")
            print()
        
        # 按特定顺序读取
        plain_text = []
        print("4. 按特定顺序读取:")
        for i in range(rows):
            if i % 2 == 0:
                print(f"   第{i+1}行从左到右读取:", end=" ")
                for j in range(cols):
                    if grid[i][j] != ' ':
                        plain_text.append(grid[i][j])
                        print(grid[i][j], end=" ")
                print()
            else:
                print(f"   第{i+1}行从右到左读取:", end=" ")
                for j in range(cols - 1, -1, -1):
                    if grid[i][j] != ' ':
                        plain_text.append(grid[i][j])
                        print(grid[i][j], end=" ")
                print()
        
        result = ''.join(plain_text)
        print(f"5. 最终明文: {result}")
        return result

    def get_encode_rule(self,):
        return """加密规则:
- 输入:
    - 明文: 大写字母字符串，不含标点和空格
- 输出:
    - 密文: 不含标点和空格的字符串
- 准备:
    - 每行最大字符数: 5
- 加密步骤:
    - 行数从1开始计数
    - 明文按特殊方式排列：奇数行从左到右写，偶数行从右到左写，每行最多五个字母
        - 例如，对于明文"LIDAHELLOWORLD"，先从左到右写第一行为LIDAH，然后从右到左写第二行为WOLLE，然后从左到右写第三行为ORLD，写完的全部内容表示如下
        LIDAH
        WOLLE
        ORLD
    - 然后按列读取，每列从上到下读取，每读完一列加一个"#"，读出的内容即为最终密文
        - 例如，上述写好的内容按列读取为LWO#IOR#DLL#ALD#HE#"""

    def get_decode_rule(self,):
        return """解密规则:
- 输入:
    - 密文: 不含标点和空格的字符串
- 输出:
    - 明文: 大写字母字符串，不含标点和空格
- 准备:
    - 行数: 5
- 解密步骤(与加密步骤完全相反):
    - 对密文中的每个字符，从上到下写入；如果遇到#，则切换到下一列继续写入，直到整个密文写完
        - 例如，对于密文LWO#IOR#DLL#ALD#HE#，写出如下
        LIDAH
        WOLLE
        ORLD
    - 然后按照奇数行从左到右读，偶数行从右到左读的顺序依次读取，最终结果即为解密后的明文
        - 例如，对于上述写好的内容，第一行从左到右读为LIDAH，第二行从右到左读为ELLOW，第三行从左到右读为ORLD，最终全部内容连接起来，解密明文为LIDAHELLOWORLD"""

from .BaseCipherEnvironment import BaseCipherEnvironment

class KorCollonCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule15_CollonCipher'
    
    def encode(self, text, **kwargs):
        grid = [['M', 'Z', 'S', 'D', 'P'],
                ['K', 'N', 'F', 'L', 'Q'],
                ['G', 'A', 'O', 'X', 'U'],
                ['W', 'R', 'Y', 'V', 'C'],
                ['B', 'T', 'E', 'H', 'I']]
        
        print("开始加密过程...")
        print(f"原始文本: {text}")
        
        # 预处理文本
        text = ''.join([char.upper() for char in text if char.isalpha()])
        text = text.replace('J', '')
        print(f"预处理后的文本(移除空格标点,转大写,移除J): {text}")
        
        encrypted_message = []
        print("\n逐字符加密:")
        for char in text:
            found = False
            for r in range(len(grid)):
                for c in range(len(grid[0])):
                    if grid[r][c] == char:
                        row_header = grid[r][0]
                        column_footer = grid[len(grid)-1][c]
                        encrypted_pair = row_header + column_footer
                        encrypted_message.append(encrypted_pair)
                        print(f"字符 {char} 位于第{r+1}行第{c+1}列")
                        print(f"-> 行首字符为{row_header}, 列尾字符为{column_footer}")
                        print(f"-> 加密为: {encrypted_pair}")
                        found = True
                        break
                if found:
                    break
        
        result = ''.join(encrypted_message)
        print(f"\n最终加密结果: {result}")
        return result

    def decode(self, text, **kwargs):
        grid = [['M', 'Z', 'S', 'D', 'P'],
                ['K', 'N', 'F', 'L', 'Q'],
                ['G', 'A', 'O', 'X', 'U'],
                ['W', 'R', 'Y', 'V', 'C'],
                ['B', 'T', 'E', 'H', 'I']]
        
        print("开始解密过程...")
        print(f"加密文本: {text}")
        
        decrypted_message = []
        print("\n每次解密两个字符:")
        for i in range(0, len(text), 2):
            bigram = text[i:i+2]
            if bigram == '':
                break
                
            row_header = bigram[0]
            column_footer = bigram[1]
            print(f"\n处理字符对: {bigram}")
            print(f"行首字符: {row_header}, 列尾字符: {column_footer}")
            
            found = False
            for r in range(len(grid)):
                if grid[r][0] == row_header:
                    for c in range(len(grid[0])):
                        if grid[len(grid)-1][c] == column_footer:
                            decrypted_char = grid[r][c]
                            decrypted_message.append(decrypted_char)
                            print(f"-> 在第{r+1}行第{c+1}列找到原文字符: {decrypted_char}")
                            found = True
                            break
                    if found:
                        break
        
        result = ''.join(decrypted_message)
        print(f"\n最终解密结果: {result}")
        return result

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 不含标点和空格的大写字母字符串
- 输出:
    - 密文: 不含标点和空格的大写字母字符串
- 准备:
    - 5x5网格(所有行和列号从0开始计数):
        - M Z S D P
        K N F L Q
        G A O X U
        W R Y V C
        B T E H I
        - 位于所有行第一个字母的MKGWB是行首字母
        - 位于所有列最后一个字母的PQUCL是列尾字母
- 加密步骤:
    - 移除明文中的空格、标点和字母J，并将所有字母转换为大写
    - 对明文中的每个字母p:
        - 在网格中找到字母p的位置，然后找到相应的行首和列尾字符
        - 将行首和列尾字符连接成二元组作为该字母p的加密消息
        - 例如，如果字母p是H，它在第4行，行首字符是B；它在第3列，列尾字符是H，所以加密消息是BH
    
    连接所有加密消息作为最终密文输出"""

    def get_decode_rule(self, ):
        return """解密规则:
- 输入:
    - 密文: 不含标点和空格的大写字母字符串
- 输出:
    - 明文: 不含标点和空格的大写字母字符串
- 准备:
    - 5x5网格(与加密相同)
- 解密步骤(与加密步骤完全相反):
    - 每次从密文中取两个字母c1,c2
        - 字母c1标识解密字母p在网格中的行位置，找到c1作为行首的行
        - 字母c2标识解密字母p在网格中的列位置，找到c2在列底的列
        - 在网格中找到这个行列位置的字母，即为c1,c2的解密消息p
        - 例如，如果c1,c2=BH，B是行首的行是第4行，H是行底的行是第2列，(4,2)处的字母是H，所以BH解密为H
    - 两个两个地解密密文中的字母，并将解密消息连接起来作为最终明文输出"""

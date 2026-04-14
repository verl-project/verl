from .BaseCipherEnvironment import BaseCipherEnvironment

class KorTranspositionCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "Transposition Cipher from kor-bench"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule20_TranspositionCipher"
    
    def encode(self, text, **kwargs):
        # 将输入转换为大写字母
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        # 定义转置序列
        transposed_sequence = [1, 4, 0, 6, 5, 2, 3]
        print(f"使用转置序列: {transposed_sequence}")
        
        # 计算需要的行数
        rows = (len(text) + 6) // 7
        print(f"需要 {rows} 行来存放文本")
        
        # 创建网格并填充
        grid = []
        index = 0
        for i in range(rows):
            row = []
            for j in range(7):
                if index < len(text):
                    row.append(text[index])
                    index += 1
                else:
                    row.append('$')
            grid.append(row)
        print("原始网格:")
        for row in grid:
            print(''.join(row))
        
        # 根据转置序列调整列顺序
        new_grid = []
        for i in range(rows):
            new_row = []
            for col in transposed_sequence:
                new_row.append(grid[i][col])
            new_grid.append(new_row)
        
        print("转置后的网格:")
        for row in new_grid:
            print(''.join(row))
        
        # 读取结果
        result = ''
        for row in new_grid:
            result += ''.join(row)
        
        print(f"最终加密结果: {result}")
        return result

    def decode(self, text, **kwargs):
        print(f"需要解密的文本: {text}")
        
        # 定义转置序列
        transposed_sequence = [1, 4, 0, 6, 5, 2, 3]
        print(f"使用转置序列: {transposed_sequence}")
        
        # 计算行数
        rows = (len(text) + 6) // 7
        print(f"需要 {rows} 行来存放文本")
        
        # 创建网格并填充
        grid = []
        index = 0
        for i in range(rows):
            row = []
            for j in range(7):
                row.append(text[index])
                index += 1
            grid.append(row)
        
        print("加密的网格:")
        for row in grid:
            print(''.join(row))
        
        # 还原原始顺序
        original_positions = [0] * 7
        for i, pos in enumerate(transposed_sequence):
            original_positions[pos] = i
        
        # 重建原始网格
        original_grid = []
        for i in range(rows):
            new_row = []
            for col in original_positions:
                new_row.append(grid[i][col])
            original_grid.append(new_row)
        
        print("还原后的网格:")
        for row in original_grid:
            print(''.join(row))
        
        # 读取结果并移除填充字符
        result = ''
        for row in original_grid:
            for char in row:
                if char != '$':
                    result += char
        
        print(f"解密结果: {result}")
        return result

    def get_encode_rule(self, ):
        return """加密规则:
    - 输入:
        - 明文: 不含标点和空格的大写字母字符串
    - 输出:
        - 密文: 不含标点和空格的字符串
    - 准备:
        - 转置序列表:
            - [1, 4, 0, 6, 5, 2, 3]
            - 转置序列表用于按顺序逐行写入明文，然后根据转置序列表调整列的顺序，使每行中的字符按给定顺序排列。
            - 列从0开始计数。
    - 加密步骤:
        - [1, 4, 0, 6, 5, 2, 3]转置序列表共7位，表示一行应写入7个字母。
        - 按顺序逐行写入明文，每行7个。当不足7个时，最后一行用$填充。可以得到一个写入网格。
        - 根据转置序列表调整列的顺序，即现在列的顺序为[原列1，原列4，原列0，原列6，原列5，原列2，原列3]，可以得到调整列顺序后的网格。
        - 逐行读取网格并连接起来得到最终密文。(注意需要保留$)"""

    def get_decode_rule(self, ):
        return """解密规则:
    - 输入:
        - 密文: 不含标点和空格的字符串
    - 输出:
        - 明文: 不含标点和空格的大写字母字符串
    - 准备:
        - 转置序列表(与加密相同)
    - 解密步骤:
        - 按顺序逐行写入密文，每行7个字母。
        - 逐行读取，但读取每行时，先读取对应0的第2列的字符，然后读取对应1的第0列的字符，然后读取对应2的第6列的字符，依此类推。
        - 最终逐行读取信息，去掉末尾的$，即可得到解密后的明文。"""

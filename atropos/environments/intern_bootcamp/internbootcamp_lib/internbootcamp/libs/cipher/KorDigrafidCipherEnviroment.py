from .BaseCipherEnvironment import BaseCipherEnvironment

import numpy as np

grid1 = np.array([
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O"],
    ["P", "A", "S", "D", "F", "G", "H", "J", "K"],
    ["L", "Z", "X", "C", "V", "B", "N", "M", "#"]
])

grid2 = np.array([
    ["Q", "W", "E"],
    ["R", "T", "Y"], 
    ["U", "I", "O"],
    ["P", "A", "S"], 
    ["D", "F", "G"], 
    ["H", "J", "K"],
    ["L", "Z", "X"], 
    ["C", "V", "B"], 
    ["N", "M", "#"]
])

grid3 = np.array([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]
])

def find_position(grid, char):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == char:
                return (i, j)
    return None

def encrypt_pair(l1, l2):
    print(f"处理字符对 {l1}{l2}:")
    l1_row, l1_col = find_position(grid1, l1)
    print(f"- 在grid1中找到{l1}的位置: 行={l1_row}, 列={l1_col}")
    l2_row, l2_col = find_position(grid2, l2)
    print(f"- 在grid2中找到{l2}的位置: 行={l2_row}, 列={l2_col}")
    num3 = grid3[l1_row, l2_col]
    print(f"- 在grid3中找到对应数字: {num3} (使用grid1的行{l1_row}和grid2的列{l2_col})")
    print(f"- 生成三元组: ({l1_col}, {num3}, {l2_row})")
    return l1_col, num3, l2_row

def decrypt_triple(x, y, z):
    print(f"\n解密三元组 ({x}, {y}, {z}):")
    l1_col = x
    l2_row = z
    l1_row, l2_col = find_position(grid3, y)
    print(f"- 在grid3中找到{y}的位置: 行={l1_row}, 列={l2_col}")
    l1 = grid1[l1_row, l1_col]
    print(f"- 在grid1中找到字符: {l1} (使用行={l1_row}, 列={l1_col})")
    l2 = grid2[l2_row, l2_col]
    print(f"- 在grid2中找到字符: {l2} (使用行={l2_row}, 列={l2_col})")
    print(f"- 解密结果: {l1}{l2}")
    return l1, l2


class KorDigrafidCipherEnviroment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule14_DigrafidCipher'
    
    def encode(self, text, **kwargs):
        print("\n开始加密过程:")
        print(f"原始文本: {text}")
        # 移除空格和标点，转换为大写
        message = ''.join(char.upper() for char in text if char.isalpha())
        print(f"预处理后的文本: {message}")
        
        # 补充#使长度为6的倍数
        while len(message) % 6 != 0:
            message += "#"
        print(f"补充#后的文本: {message}")
        
        # 分割成二元组
        bigrams = [message[i:i+2] for i in range(0, len(message), 2)]
        print(f"分割成二元组: {bigrams}")
        
        # 加密每个二元组
        triples = [encrypt_pair(l1, l2) for l1, l2 in bigrams]
        encrypted_pairs = ["".join(map(str, triple)) for triple in triples]
        encrypted_message = "".join(encrypted_pairs)
        print(f"\n最终加密结果: {encrypted_message}")
        return encrypted_message

    def decode(self, text, **kwargs):
        print("\n开始解密过程:")
        print(f"加密文本: {text}")
        
        # 分割成三元组
        original_triples = [text[i:i+3] for i in range(0, len(text), 3)]
        print(f"分割成三元组: {original_triples}")
        original_triples = [[int(item) for item in row] for row in original_triples]
        
        # 解密每个三元组
        decrypted_pairs = [decrypt_triple(*triple) for triple in original_triples]
        decrypted_message = "".join(sum(decrypted_pairs, ()))
        decrypted_message = decrypted_message.replace("#", "")
        print(f"\n最终解密结果: {decrypted_message}")
        return decrypted_message

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 大写字母字符串，不含标点和空格
- 输出:
    - 密文: 数字字符串，不含标点和空格
- 准备:
    - 3个网格(所有行列号从0开始计数):
        - 网格1 (3x9):
            Q W E R T Y U I O
            P A S D F G H J K
            L Z X C V B N M #
        - 网格2 (9x3):
            Q W E
            R T Y
            U I O
            P A S
            D F G
            H J K
            L Z X
            C V B
            N M #
        - 网格3 (3x3):
            1 2 3
            4 5 6
            7 8 9
- 加密步骤:
    - 移除所有空格和标点，将文本转换为大写字母
    - 将明文切分为6个字符一组，如果最后一组不足6个字符，用#填充
    - 将每组6个字符分成3个二元组
    - 对每个二元组(L1, L2)执行以下操作:
        - 确定L1在网格1中的行列号(l1_row, l1_col)
        - 确定L2在网格2中的行列号(l2_row, l2_col)
        - 在网格3中用l1_row和l2_col找到对应数字num3
        - 输出三元组(l1_col, num3, l2_row)
    - 将所有三元组连接成一个数字串作为加密信息"""

    def get_decode_rule(self, ):
        return """解密规则:
- 输入:
    - 密文: 数字字符串，不含标点和空格
- 输出:
    - 明文: 大写字母字符串，不含标点和空格
- 准备:
    - 3个网格(与加密相同)
- 解密步骤:
    - 将密文分成三个数字一组
    - 对每个三元组(x, y, z)执行以下操作:
        - 在网格3中找到y的行号作为L1_row
        - 在网格3中找到y的列号作为L2_col
        - L1_col等于x，L2_row等于z
        - 根据确定的(L1_row,L1_col)在网格1中找到对应字母p1
        - 根据确定的(L2_row,L2_col)在网格2中找到对应字母p2
        - p1p2为该三元组解密后的消息
    - 将所有解密后的消息连接起来，移除末尾的#(这些字符是为使消息长度为6的倍数而添加的填充字符)，形成解密后的明文"""

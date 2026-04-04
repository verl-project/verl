from .BaseCipherEnvironment import BaseCipherEnvironment

import numpy as np

def generate_initial_grid(keyword):
    cleaned_keyword = keyword.upper().replace('J', 'I')
    cleaned_keyword = ''.join(sorted(set(cleaned_keyword), key=cleaned_keyword.index))
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
    grid = np.array(list(cleaned_keyword + ''.join(filter(lambda c: c not in cleaned_keyword, alphabet))))
    grid = grid.reshape(5, 5)
    return grid

def generate_subsequent_grids(initial_grid):
    grids = [initial_grid]
    for i in range(1, 5):
        grid = np.roll(initial_grid, i, axis=0)
        grids.append(grid)
    for i in range(1, 4):
        grid = np.roll(grids[4], i, axis=0)
        grids.append(grid)
    return grids

def find_position(grid, letter):
    indices = np.where(grid == letter)
    return indices[0][0], indices[1][0]


class KorPhillipsFigureCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = """
        Kor Phillips Figure Cipher
        """
        super().__init__(problem_description, *args, **kwargs)
    
    @property
    def cipher_name(self) -> str:
        return "Kor_rule7_PhillipsFigureCipher"
    
    def encode(self, text, **kwargs):
        keyword = "PHILLIPS"
        # 处理输入文本，只保留字母并转为大写
        message = ''.join(char.upper() for char in text if char.isalpha())
        print(f"处理后的输入文本: {message}")
        
        initial_grid = generate_initial_grid(keyword)
        grids = generate_subsequent_grids(initial_grid)
        encrypted_message = []
        
        for i in range(0, len(message), 5):
            block = message[i:i+5]
            grid_index = (i // 5) % 8
            grid = grids[grid_index]
            print(f"\n处理第{i//5}个块: {block}")
            print(f"使用第{grid_index}号网格")
            
            encrypted_block = []
            for letter in block:
                if letter == 'J':
                    encrypted_block.append(letter)
                    print(f"字母{letter}是J，直接保留")
                else:
                    row, col = find_position(grid, letter)
                    encrypted_letter = grid[(row + 1) % 5, (col + 1) % 5]
                    print(f"字母{letter}在位置({row},{col})，向右下移动一格得到{encrypted_letter}")
                    encrypted_block.append(encrypted_letter)
            
            encrypted_message.append(''.join(encrypted_block))
            print(f"加密后的块: {''.join(encrypted_block)}")
        
        result = ''.join(encrypted_message)
        print(f"\n最终加密结果: {result}")
        return result

    def decode(self, text, **kwargs):
        keyword = "PHILLIPS"
        initial_grid = generate_initial_grid(keyword)
        grids = generate_subsequent_grids(initial_grid)
        decrypted_message = []
        
        print(f"需要解密的文本: {text}")
        
        for i in range(0, len(text), 5):
            block = text[i:i+5]
            grid_index = (i // 5) % 8
            grid = grids[grid_index]
            print(f"\n处理第{i//5}个块: {block}")
            print(f"使用第{grid_index}号网格")
            
            decrypted_block = []
            for letter in block:
                if letter == 'J':
                    decrypted_block.append(letter)
                    print(f"字母{letter}是J，直接保留")
                else:
                    row, col = find_position(grid, letter)
                    decrypted_letter = grid[(row - 1) % 5, (col - 1) % 5]
                    print(f"字母{letter}在位置({row},{col})，向左上移动一格得到{decrypted_letter}")
                    decrypted_block.append(decrypted_letter)
            
            decrypted_message.append(''.join(decrypted_block))
            print(f"解密后的块: {''.join(decrypted_block)}")
        
        result = ''.join(decrypted_message)
        print(f"\n最终解密结果: {result}")
        return result

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 不含标点和空格的大写字母字符串
- 输出:
    - 密文: 大写字母字符串
- 准备:
    - 字母表 = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'(不包含字母J)
    - 8个网格(Grid0-Grid7)
- 加密步骤:
    - 明文按5个字符分组，从0开始编号
    - 对于5个字符的块:
        - 使用的网格由grid_index = (i // 5) % 8确定，其中i是块号。整数除法运算符//将左边的数除以右边的数，向下取整结果
        - 对于当前块中的每个字符:
            - 如果字符是"J"，不加密直接添加到加密块
            - 否则，在当前网格中找到字符的位置。然后向右下方移动一个网格位置(row+1,col+1)(如果越界则在对应边界的另一侧继续)，移动后位置的字母作为加密字母
            - 将加密字母添加到加密块
    - 处理完所有块后，连接加密块形成最终加密消息"""

    def get_decode_rule(self, ):
        return """解密规则:
- 输入:
    - 密文: 大写字母字符串
- 输出:
    - 明文: 大写字母字符串
- 准备:
    - 字母表 = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'(不包含字母J)
    - 8个网格(与加密相同)
- 解密步骤:
    - 将密文分成5个字符的块:
        - 例如，如果密文是"KHOORTQHQTHHSAUQ"，第0块是"KHOOR"，第1块是"TQHQH"，以此类推(从0开始编号)
    - 确定用于当前块的网格:
        计算grid_index = (i // 5) % 8从网格列表中选择适当的网格。i是块号
    - 对于块中的每个字符:
        - 如果字符是"J": 直接将"J"添加到解密块，不进行解密
        - 否则在网格中找到字符的位置，通过向左上方移动一格获得(如果越界则在对应边界的另一侧继续)，移动后位置的字母作为解密字母
        - 将解密字母添加到解密块
    - 处理完块中所有字符后，将解密块添加到解密消息列表。形成最终解密消息"""

        
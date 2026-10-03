from .BaseCipherEnvironment import BaseCipherEnvironment

import numpy as np

def create_grid(template, message_segment):
    size = template.shape[0]
    grid = np.full((size, size), '', dtype=str)
    idx = 0
    for _ in range(4):
        for i in range(size):
            for j in range(size):
                if (template[i, j]==0) and idx < len(message_segment):
                    grid[i, j] = message_segment[idx]
                    idx += 1
        template = np.rot90(template)
    return grid


class KorRotatingGridCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "Rotating Grid Cipher from kor"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule18_RotatingGridCipher"
    
    def encode(self, text, **kwargs):
        return text
    
    def encode(self, text, **kwargs):
        # 处理输入文本，只保留字母并转为大写
        text = ''.join([char.upper() for char in text if char.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        template = np.array([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1]
        ], dtype=bool)
        
        print("使用的模板:")
        print("▮ ▮ ▮ ▮")
        print("▮ ▮ ▯ ▯")
        print("▮ ▮ ▮ ▯")
        print("▯ ▮ ▮ ▮")
        
        size = template.shape[0]
        segment_length = size * size
        ciphertext = ''
        
        print("\n开始加密过程:")
        for i in range(0, len(text), segment_length):
            print(f"\n处理第{i//segment_length + 1}个块:")
            segment = text[i:i + segment_length]
            print(f"当前块的文本: {segment}")
            
            if len(segment) < segment_length:
                print(f"文本长度不足{segment_length}，用#补充")
                segment += '#' * (segment_length - len(segment))
                print(f"补充后的文本: {segment}")
            
            filled_grid = create_grid(template, segment)
            print("\n填充后的网格:")
            print(filled_grid)
            
            block_cipher = ''.join(filled_grid.flatten())
            print(f"当前块的密文: {block_cipher}")
            ciphertext += block_cipher
        
        print(f"\n最终密文: {ciphertext}")
        return ciphertext

    def decode(self, text, **kwargs):
        template = np.array([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1]
        ], dtype=bool)
        
        print("使用的模板:")
        print("▮ ▮ ▮ ▮")
        print("▮ ▮ ▯ ▯")
        print("▮ ▮ ▮ ▯")
        print("▯ ▮ ▮ ▮")

        size = template.shape[0]
        segment_length = size * size
        message = ''
        
        print("\n开始解密过程:")
        for i in range(0, len(text), segment_length):
            print(f"\n处理第{i//segment_length + 1}个块:")
            current_segment = text[i:i + segment_length]
            print(f"当前块的密文: {current_segment}")
            
            filled_grid = np.array(list(current_segment)).reshape((size, size))
            print("\n重构的网格:")
            print(filled_grid)
            
            segment = []
            temp_template = template.copy()
            for rotation in range(4):
                print(f"\n第{rotation + 1}次旋转后读取:")
                for i in range(size):
                    for j in range(size):
                        if temp_template[i, j]==0:
                            segment.append(filled_grid[i, j])
                            print(f"从位置({i},{j})读取字符: {filled_grid[i, j]}")
                temp_template = np.rot90(temp_template)
            
            block_message = ''.join(segment)
            print(f"当前块解密结果: {block_message}")
            message += block_message
        
        message = message.rstrip('#')
        print(f"\n最终明文: {message}")
        return message

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 不含标点和空格的大写字母字符串
- 输出:
    - 密文: 不含标点和空格的大写字母字符串
- 准备:
    - 网格和模板:
        - 准备一个空白网格和一个带孔的模板(栅栏)
        - 使用的模板是:
            ▮ ▮ ▮ ▮
            ▮ ▮ ▯ ▯
            ▮ ▮ ▮ ▯
            ▯ ▮ ▮ ▮
            其中白色的是孔，将模板放在空白网格上，只通过白色孔洞，在网格的对应位置填入字符。
- 加密步骤:
    - 将明文逐个分成16个字母的块(如果明文少于16个长度则为一个块)
    - 对每个块执行以下加密操作:
        - 将带孔的模板放在空白网格上
        - 通过模板中的孔按顺序填入明文字母
        - 模板总共有四个孔，所以填完四个字母后，需要将模板逆时针旋转90度
        - 重复填充可见孔中的明文下一个字母并旋转模板，直到整个网格完全填满。这将执行4次填充+旋转，最终模板会转回原始模板。如果消息不足以填满整个网格，用填充字符(如'#')补充
        - 网格填满后，按行读取网格内容作为该块的加密消息
        - 进入下一个块时，清空网格内容并重做整个填充和旋转操作
    最后，将所有块的加密消息连接在一起作为最终密文。"""

    def get_decode_rule(self, ):
        return """解密规则:
- 输入:
    - 密文: 不含标点和空格的大写字母字符串
- 输出:
    - 明文: 不含标点和空格的大写字母字符串
- 准备:
    - 网格和模板(与加密相同)
- 解密步骤(与加密步骤完全相反):
    - 将密文逐个分成16个字母的块
    - 对每个块执行以下操作:
        - 将16个字母按行写入填满网格
        - 将带孔的模板放在填满的网格上
        - 读取通过孔可见的字母获得部分明文消息
        - 由于只有四个孔，此时需要将模板逆时针旋转90度读取下一组字母
        - 重复读取步骤四次以获得这块解密消息
    - 连接所有块的解密消息得到最终明文。"""

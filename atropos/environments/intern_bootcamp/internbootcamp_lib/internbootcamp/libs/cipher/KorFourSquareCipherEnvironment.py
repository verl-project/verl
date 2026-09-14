from .BaseCipherEnvironment import BaseCipherEnvironment

import re

def generate_table(key = ''):    
    alphabet = 'ABCDEFGHIJKLMNOPRSTUVWXYZ'    
    table = [[0] * 5 for row in range(5)] 

    key = re.sub(r'[\W]', '', key).upper()
    for row in range(5):
        for col in range(5):
            if len(key) > 0:
                table[row][col] = key[0]                
                alphabet = alphabet.replace(key[0], '') 
                key = key.replace(key[0], '')           
            else:
                table[row][col] = alphabet[0]
                alphabet = alphabet[1:]                 
    return table

def position(table, ch):    
    for row in range(5):    
        for col in range(5):    
            if table[row][col] == ch:    
                return (row, col)       
    return (None, None)


class KorFourSquareCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule11_FourSquareCipher'
    

    def encode(self, text, **kwargs):
        print("开始加密过程...")
        print(f"原始文本: {text}")
        
        # 清理文本
        plaintext = ''.join(char.upper() for char in text if char.isalpha())
        plaintext = plaintext.replace('Q', '')
        print(f"清理后的文本(移除非字母字符并转大写，移除Q): {plaintext}")
        
        # 如果长度为奇数，添加X
        if len(plaintext) % 2 != 0:
            plaintext += 'X'
            print(f"文本长度为奇数，添加X: {plaintext}")
        
        # 准备密钥表格
        key = ['ECHO', 'VORTEX']
        table = [['K', 'L', 'M', 'N', 'O'],
                ['P', 'R', 'S', 'T', 'U'],
                ['V', 'W', 'X', 'Y', 'Z'],
                ['A', 'B', 'C', 'D', 'E'],
                ['F', 'G', 'H', 'I', 'J']]
        topRight = generate_table(key[0])
        bottomLeft = generate_table(key[1])
        
        print("\n开始逐对字母加密:")
        ciphertext = ''
        for i in range(0, len(plaintext), 2):
            pair = plaintext[i:i+2]
            print(f"\n处理字母对: {pair}")
            
            # 在原始表格中找到位置
            pos1 = position(table, pair[0])
            pos2 = position(table, pair[1])
            print(f"第一个字母 {pair[0]} 在原始表格中的位置: ({pos1[0]}, {pos1[1]})")
            print(f"第二个字母 {pair[1]} 在原始表格中的位置: ({pos2[0]}, {pos2[1]})")
            
            # 在新表格中找到对应字母
            encrypted_pair = topRight[pos1[0]][pos1[1]] + bottomLeft[pos2[0]][pos2[1]]
            print(f"加密后的字母对: {encrypted_pair}")
            ciphertext += encrypted_pair
        
        print(f"\n最终加密结果: {ciphertext}")
        return ciphertext

    def decode(self, text, **kwargs):
        print("开始解密过程...")
        print(f"加密文本: {text}")
        
        # 准备密钥表格
        key = ['ECHO', 'VORTEX']
        table = [['K', 'L', 'M', 'N', 'O'],
                ['P', 'R', 'S', 'T', 'U'],
                ['V', 'W', 'X', 'Y', 'Z'],
                ['A', 'B', 'C', 'D', 'E'],
                ['F', 'G', 'H', 'I', 'J']]
        topRight = generate_table(key[0])
        bottomLeft = generate_table(key[1])
        
        print("\n开始逐对字母解密:")
        plaintext = ''
        for i in range(0, len(text), 2):
            pair = text[i:i+2]
            print(f"\n处理加密字母对: {pair}")
            
            # 在新表格中找到位置
            pos1 = position(topRight, pair[0])
            pos2 = position(bottomLeft, pair[1])
            print(f"第一个字母 {pair[0]} 在ECHO表格中的位置: ({pos1[0]}, {pos1[1]})")
            print(f"第二个字母 {pair[1]} 在VORTEX表格中的位置: ({pos2[0]}, {pos2[1]})")
            
            # 在原始表格中找到对应字母
            decrypted_pair = table[pos1[0]][pos1[1]] + table[pos2[0]][pos2[1]]
            print(f"解密后的字母对: {decrypted_pair}")
            plaintext += decrypted_pair
        
        print(f"\n最终解密结果: {plaintext}")
        return plaintext

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 大写字母字符串，不含标点和空格
- 输出:
    - 密文: 大写字母字符串
- 准备:
以下是经过整理后的文字：

---

### 5x5 网格布局

共有四个5x5的字符网格，每个网格代表了不同的排列方式。

1. **网格1 原始网格**

这是最初的字符排列，按照特定顺序组织如下：

```
K L M N O
P R S T U
V W X Y Z
A B C D E
F G H I J
```

2. **网格2 ECHO 网格**

该网格根据"ECHO"这个词进行了重新排列：

```
E C H O A
B D F G I
J K L M N
P R S T U
V W X Y Z
```

3. **网格3 VORTEX 网格**

此网格基于"VORTEX"一词进行了独特的字符重组：

```
V O R T E
X A B C D
F G H I J
K L M N P
S U W Y Z
```

4. **网格4 重复原始网格**

最后一个网格与第一个原始网格完全相同，没有进行任何改变：

```
K L M N O
P R S T U
V W X Y Z
A B C D E
F G H I J
```

每个网格展示了不同主题词下字符的独特排列。
- 加密步骤:
    - 清理明文，移除空格和非字母字符，移除字母Q，转换为大写
    - 如果明文长度为奇数，添加字母'X'使其成为偶数
    - 将处理后的明文分成两个字母一组
    - 对于每组两个字母p1,p2:
        - 在网格1和网格4中找到第一个字母和第二个字母的位置
        - 在网格2和网格3中找到这两个位置对应的字母，用这两个字母作为该组的加密结果
    - 连接所有加密后的字母组形成最终密文"""

    def get_decode_rule(self, ):
        return """解密规则:
- 输入:
    - 密文: 大写字母字符串
- 输出:
    - 明文: 大写字母字符串
- 准备:
    - 四个5x5网格(与加密相同)
- 解密步骤(与加密步骤相反):
    - 清理密文，移除空格和非字母字符，转换为大写
    - 将处理后的密文分成两个字母一组
    - 对于每组两个字母c1,c2:
        - 在网格2和网格3中找到第一个字母和第二个字母的位置
        - 在网格1和网格4中找到这两个位置对应的字母，用这两个字母作为该组的解密结果
    - 连接所有解密后的字母组形成最终明文"""

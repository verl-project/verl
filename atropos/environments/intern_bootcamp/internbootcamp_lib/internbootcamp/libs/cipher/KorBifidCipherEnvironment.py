from .BaseCipherEnvironment import BaseCipherEnvironment

def checkRepeat(secretKey):
    nonRepeated = ""
    for char in secretKey:
        if nonRepeated == "":
            nonRepeated += char
            continue
        if char in nonRepeated:
            continue
        else:
            nonRepeated += char
    return(nonRepeated)

def fillMatrix(nonRepeated):
    nonRepeated.replace("J", "I")
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    row = 0
    col = 0
    matrix = [[0 for i in range(5)] for j in range(5)]
    for letter in nonRepeated:
        if col == 5:
            row += 1
            col = 0
        matrix[row][col] = letter
        col += 1
    for letter in alphabet:
        if col == 5:
            row += 1
            col = 0
        if letter in nonRepeated:
            continue
        else:
            matrix[row][col] = letter
            col += 1
    return(matrix)

def getCoords(message, matrix):
    coordsRow = list()
    coordsCol = list()
    for letters in message:
        for i in range(5):
            for j in range(5):
                if matrix[i][j] == letters:
                    coordsRow.append(i)
                    coordsCol.append(j)
                    continue
    finalCoords = list()
    for letters in coordsRow:
        finalCoords.append(letters)
    for letters in coordsCol:
        finalCoords.append(letters)
    return(finalCoords)

def getNewMessage(finalCoords, matrix):
    newMessage = ""
    length = len(finalCoords)
    finalCoordsCopy = finalCoords.copy()
    for i in range(int(length / 2) + 1):
        x = finalCoordsCopy.pop(0)
        y = finalCoordsCopy.pop(0)
        newMessage += matrix[x][y]
        if not finalCoordsCopy: break
    return(newMessage)

def getOldCoords(coords):
    oldCoords = ''.join(str(coords))
    firstHalf = oldCoords[:len(oldCoords)//2]
    secondHalf = oldCoords[len(oldCoords)//2:]
    fH = "".join(c for c in firstHalf if c.isdecimal())
    sH = "".join(c for c in secondHalf if c.isdecimal())

    firstHalf = list(fH)
    secondHalf = list(sH)

    stepOneCoords = list()
    for coord in fH:
        x = firstHalf.pop(0)
        y = secondHalf.pop(0)
        stepOneCoords.append(x)
        stepOneCoords.append(y)

    stepTwoCoords = ''.join(str(stepOneCoords))
    firstHalf = stepTwoCoords[:len(stepTwoCoords)//2]
    secondHalf = stepTwoCoords[len(stepTwoCoords)//2:]
    
    fH = "".join(c for c in firstHalf if c.isdecimal())
    sH = "".join(c for c in secondHalf if c.isdecimal())

    firstHalf = list(fH)
    secondHalf = list(sH)

    fullOldCoords = list()
    for coord in fH:
        x = firstHalf.pop(0)
        y = secondHalf.pop(0)
        fullOldCoords.append(x)
        fullOldCoords.append(y)

    return fullOldCoords

def getOldMessage(fullOldCoords, matrix):
    oldMessage = ""
    length = len(fullOldCoords)
    fullOldCoordsCopy = fullOldCoords.copy()

    for i in range(int(length / 2) + 1):
        x = int(fullOldCoordsCopy.pop(0))
        y = int(fullOldCoordsCopy.pop(0))
        oldMessage += matrix[x][y]
        if not fullOldCoordsCopy: break
    return(oldMessage)


class KorBifidCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return 'Kor_rule13_BifidCipher'
    

    def encode(self, text, **kwargs):
        secretKey = "UBILANT"
        print("加密步骤:")
        # 1. 处理输入文本
        text = ''.join([c.upper() for c in text if c.isalpha()])
        text = text.replace('J','I')
        print(f"1. 去除标点和空格,将J替换为I后的文本: {text}")
        
        # 2. 生成5x5矩阵
        matrix = fillMatrix(checkRepeat(secretKey))
        print("2. 生成的5x5矩阵:")
        for row in matrix:
            print(row)
            
        # 3. 获取坐标
        coords = getCoords(text, matrix)
        print(f"3. 获取每个字母的行列坐标: {coords}")
        
        # 4. 重排坐标并加密
        ciphered = getNewMessage(coords, matrix)
        print(f"4. 最终加密结果: {ciphered}")
        
        return ciphered

    def decode(self, text, **kwargs):
        secretKey = "UBILANT"
        print("解密步骤:")
        # 1. 生成5x5矩阵
        matrix = fillMatrix(checkRepeat(secretKey))
        print("1. 生成的5x5矩阵:")
        for row in matrix:
            print(row)
            
        # 2. 获取密文坐标
        coords = getCoords(text, matrix)
        print(f"2. 获取密文字母的坐标: {coords}")
        
        # 3. 还原原始坐标
        oldCoords = getOldCoords(coords)
        print(f"3. 还原原始坐标: {oldCoords}")
        
        # 4. 解密
        plaintext = getOldMessage(oldCoords, matrix)
        print(f"4. 最终解密结果: {plaintext}")
        
        return plaintext

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 大写字母字符串,不含标点和空格
- 输出:
    - 密文: 大写字母字符串
- 准备:
    - 5x5网格(不含字母J,行列坐标在0到4之间):
        - U B I L A
        N T C D E
        F G H K M
        O P Q R S
        V W X Y Z
- 加密步骤:
    - 去除明文中的标点、空格和字母J
    - 对明文中的每个字母,找到其在网格中对应的行列坐标(0-4)。例如A是(0,4)
    - 将所有行列坐标重排,先读所有行坐标,再读所有列坐标形成新的坐标序列。例如原来是(0,4)(1,2),现在读作0142
    - 每次从新坐标序列中取出两个数作为新的行列坐标,在网格中找到对应字母形成密文。例如对于0142,密文对应(0,1)是B,(4,2)是X。最终密文是BX"""

    def get_decode_rule(self, ):
        return """解密规则:
- 输入:
    - 密文: 大写字母字符串,不含标点和空格
- 输出:
    - 明文: 大写字母字符串,不含标点和空格
- 准备:
    - 5x5网格(与加密相同)
- 解密步骤(与加密步骤完全相反):
    - 对密文中的每个字母,找到其在网格中对应的行列坐标(0-4)得到坐标序列
    - 将坐标序列分成两半,前半部分为所有原始行坐标,后半部分为所有原始列坐标,从原始行坐标读一个,从原始列坐标读一个,根据原始行列坐标在网格中找到对应字母形成明文
        - 例如[0,1,4,2],前半部分作为行坐标:[0,1],后半部分作为列坐标:[4,2]
        - 交替读取行坐标和列坐标。得到(0,4),(1,2),分别在网格中找到对应字母为AC"""

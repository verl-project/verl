from .BaseCipherEnvironment import BaseCipherEnvironment

def create_cipher_matrix(key):
    """
    根据给定的密钥构建5x5的加密矩阵。
    参数:
    - key: 用于构建矩阵的字符串

    返回:
    - 5x5加密矩阵
    """
    unique_key = "".join(dict.fromkeys(key.upper()))
    alphabet = "ABCDEFGHIJKLMNOPRSTUVWXYZ"  # delete Q
    remaining_letters = "".join([ch for ch in alphabet if ch not in unique_key])
    full_key = unique_key + remaining_letters
    matrix = [list(full_key[i:i + 5]) for i in range(0, 25, 5)]
    return matrix


def find_position(matrix, char):
    """
    查找字符在给定矩阵中的位置。
    参数:
    - matrix: 5x5的矩阵
    - char: 需要查找的字符

    返回:
    - 字符在矩阵中的行列位置（tuple）
    """
    if char == 'Q':
        return (4, 2)
    for i, row in enumerate(matrix):
        if char in row:
            return (i, row.index(char))
    return (-1, -1)

class FourSquareCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "FourSquareCipher"

    def encode(self, text, str1, str2, ):
        """
        使用Four-square Cipher算法加密文本。
        参数:
        - str1: 第一个密钥字符串，用于构建第一个加密矩阵
        - str2: 第二个密钥字符串，用于构建第二个加密矩阵
        - text: 待加密的明文字符串

        返回:
        - 加密后的字符串
    """
        # 打印解题方案
        print("解题方案：")
        print("1. 构建3个矩阵：默认字母表矩阵M_T，加密矩阵M1和M2（由str1和str2生成）。")
        print("2. 将明文文本转换为大写字母，仅保留字母字符并分成两个字母一组。")
        print("3. 对每对字母进行加密：")
        print("   - 查找每个字母在默认矩阵中的位置，交换其y坐标，")
        print("   - 根据新坐标从M1和M2矩阵中取出加密后的字母对。")
        print("4. 输出加密后的文本。\n")
        # 创建默认字母表矩阵（去除Q）
        default_matrix = create_cipher_matrix("ABCDEFGHIJKLMNOPRSTUVWXYZ")
        print("Step 1: 创建默认字母表矩阵 M_T (去除Q)：")
        for row in default_matrix:
            print(" ".join(row))
        print()

        # 使用str1和str2创建两个加密矩阵
        matrix1 = create_cipher_matrix(str1)
        matrix2 = create_cipher_matrix(str2)

        print("Step 2: 使用密钥构建加密矩阵 M1 和 M2")
        print("M1 矩阵 (根据密钥 str1):")
        for row in matrix1:
            print(" ".join(row))
        print("\nM2 矩阵 (根据密钥 str2):")
        for row in matrix2:
            print(" ".join(row))
        print()

        # 将明文转换为大写并移除非字母字符
        text = ''.join(filter(str.isalpha, text.upper()))

        # 如果明文字符个数为奇数，补一个字符 'X'
        if len(text) % 2 != 0:
            print('明文字符个数为奇数，补一个字符 X')
            text += "X"

        # 两个字母一组分组
        pairs = [text[i:i + 2] for i in range(0, len(text), 2)]
        print("Step 3: 将明文分成两个字母一组:", pairs)
        print()

        encrypted_text = []

        # 遍历每对字母进行加密
        for idx, pair in enumerate(pairs):
            print(f"Step 4.{idx + 1}: 加密字母对 {pair}")

            # 查找两个字母在默认字母表矩阵中的位置
            pos1 = find_position(default_matrix, pair[0])
            pos2 = find_position(default_matrix, pair[1])

            print(f"  - {pair[0]} 在 M_T 中的位置: {pos1}")
            print(f"  - {pair[1]} 在 M_T 中的位置: {pos2}")

            # 交换y坐标，得到新位置
            new_pos1 = (pos1[0], pos2[1])
            new_pos2 = (pos2[0], pos1[1])

            print(f"  - 交换列索引后新位置: {new_pos1} 和 {new_pos2}")

            # 从第一个和第二个加密矩阵中取出新位置上的字母
            encrypted_char1 = matrix1[new_pos1[0]][new_pos1[1]]
            encrypted_char2 = matrix2[new_pos2[0]][new_pos2[1]]

            print(f"  - 在 M1 中查找位置 {new_pos1} 的字母: {encrypted_char1}")
            print(f"  - 在 M2 中查找位置 {new_pos2} 的字母: {encrypted_char2}")
            print(f"  - 加密结果为: {encrypted_char1 + encrypted_char2}\n")

            # 将加密后的字母添加到结果列表中
            encrypted_text.append(encrypted_char1 + encrypted_char2)

        # 返回加密结果，每两个字母用空格分隔
        final_result = " ".join(encrypted_text)
        print("加密完成！最终加密结果为:", final_result)
        return final_result


    def decode(self, text, str1, str2):
        """
        使用Four-square Cipher算法解密文本。
        参数:
        - str1: 第一个密钥字符串，用于构建第一个加密矩阵
        - str2: 第二个密钥字符串，用于构建第二个加密矩阵
        - text: 待解密的密文字符串

        返回:
        - 解密后的字符串
        """
        # 打印解题方案
        print("解题方案：")
        print("1. 构建3个矩阵：默认字母表矩阵M_T，加密矩阵M1和M2（由str1和str2生成）。")
        print("2. 将密文按两个字母一组分组。")
        print("3. 对每对加密字母进行解密：")
        print("   - 在M1和M2中找到加密字母的位置，交换列坐标")
        print("   - 从默认矩阵中获取对应的明文字母。")
        print("4. 输出解密后的文本。\n")

        # 创建默认字母表矩阵（去除Q）
        default_matrix = create_cipher_matrix("ABCDEFGHIJKLMNOPRSTUVWXYZ")  # delete Q
        print("Step 1: 创建默认字母表矩阵 M_T (去除Q)：")
        for row in default_matrix:
            print(" ".join(row))
        print()

        # 使用str1和str2创建两个加密矩阵
        matrix1 = create_cipher_matrix(str1)
        matrix2 = create_cipher_matrix(str2)

        print("Step 2: 使用密钥构建加密矩阵 M1 和 M2")
        print("M1 矩阵 (根据密钥 str1):")
        for row in matrix1:
            print(" ".join(row))
        print("\nM2 矩阵 (根据密钥 str2):")
        for row in matrix2:
            print(" ".join(row))
        print()

        # 将密文转换为大写并按两个字母一组分组
        text = ''.join(filter(str.isalpha, text.upper()))
        pairs = [text[i:i + 2] for i in range(0, len(text), 2)]
        print("Step 3: 将密文分成两个字母一组:", pairs)
        print()

        decrypted_text = []

        # 遍历每对字母进行解密
        for idx, pair in enumerate(pairs):
            print(f"Step 4.{idx + 1}: 解密字母对 {pair}")

            # 在M1和M2中找到两个加密字母的位置
            pos1 = find_position(matrix1, pair[0])
            pos2 = find_position(matrix2, pair[1])

            print(f"  - {pair[0]} 在 M1 中的位置: {pos1}")
            print(f"  - {pair[1]} 在 M2 中的位置: {pos2}")

            try:
                # 交换列坐标得到新位置
                new_pos1 = (pos1[0], pos2[1])
                new_pos2 = (pos2[0], pos1[1])
            except Exception as e:
                print(e)
            print(f"  - 交换列索引后新位置: {new_pos1} 和 {new_pos2}")

            # 从默认矩阵中取出新位置上的字母
            decrypted_char1 = default_matrix[new_pos1[0]][new_pos1[1]]
            decrypted_char2 = default_matrix[new_pos2[0]][new_pos2[1]]

            print(f"  - 在 M_T 中查找位置 {new_pos1} 的字母: {decrypted_char1}")
            print(f"  - 在 M_T 中查找位置 {new_pos2} 的字母: {decrypted_char2}")
            print(f"  - 解密结果为: {decrypted_char1 + decrypted_char2}\n")

            # 将解密后的字母添加到结果列表中
            decrypted_text.append(decrypted_char1 + decrypted_char2)

        # 返回解密结果，去除多余空格
        final_result = "".join(decrypted_text).strip()
        print("解密完成！最终解密结果为:", final_result)
        return final_result

    def get_encode_rule(self):
        return "解题方案：\n1. 构建3个矩阵：默认字母表矩阵M_T，加密矩阵M1和M2（由str1和str2生成）。\n2. 将明文文本转换为大写字母，仅保留字母字符并分成两个字母一组。\n3. 对每对字母进行加密：\n   - 查找每个字母在默认矩阵中的位置，交换其y坐标，\n   - 根据新坐标从M1和M2矩阵中取出加密后的字母对。\n4. 输出加密后的文本。"

    def get_decode_rule(self):
        return "解题方案：\n1. 构建3个矩阵：默认字母表矩阵M_T，加密矩阵M1和M2（由str1和str2生成）。\n2. 将密文按两个字母一组分组。\n3. 对每对加密字母进行解密：\n   - 在M1和M2中找到加密字母的位置，交换列坐标\n   - 从默认矩阵中获取对应的明文字母。\n4. 输出解密后的文本。"

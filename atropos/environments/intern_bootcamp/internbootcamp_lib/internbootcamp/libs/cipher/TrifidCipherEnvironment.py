from .BaseCipherEnvironment import BaseCipherEnvironment

class TrifidCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description, *args, **kwargs)
        
    def encode(self, text):
        """
        Trifid Cipher 加密算法实现

        解题方案：
        1. 定义3x3x3立方体的布局，每个字母对应一个唯一的三维坐标 (层, 列, 行)。
        2. 过滤掉明文中的非字母字符，并将其转换为大写。
        3. 将每个字母转换为其对应的三维坐标，并分别存储层、列、行坐标。
        4. 将所有层坐标、列坐标、行坐标分别合并成一个字符串。
        5. 按照每三个字符一组的方式重新组合坐标。
        6. 根据新的坐标组在立方体中找到对应的字母，生成密文。
        """

        # 定义3x3x3立方体的布局
        cube = [
            ['A', 'B', 'C'],  # 第1层
            ['D', 'E', 'F'],
            ['G', 'H', 'I'],
            ['J', 'K', 'L'],  # 第2层
            ['M', 'N', 'O'],
            ['P', 'Q', 'R'],
            ['S', 'T', 'U'],  # 第3层
            ['V', 'W', 'X'],
            ['Y', 'Z', '.']
        ]

        # 打印立方体布局
        print("立方体布局:")
        for layer in range(3):
            print(f"第{layer + 1}层:")
            for row in range(3):
                print(' '.join(cube[layer * 3 + row]))
            print()

        # Step 1: 过滤掉非字母字符并转换为大写
        filtered_text = ''.join(filter(str.isalpha, text)).upper()
        print(f"Step 1: 过滤掉非字母字符并转换为大写: {filtered_text}")

        # 用于存储层、列、行坐标
        layer_coords = []
        column_coords = []
        row_coords = []

        # Step 2: 获取每个字符的坐标
        print("Step 2: 获取每个字符的坐标:")
        for char in filtered_text:
            found = False
            for layer in range(3):
                for row in range(3):
                    for col in range(3):
                        if cube[layer * 3 + row][col] == char:
                            layer_coords.append(str(layer + 1))
                            column_coords.append(str(col + 1))
                            row_coords.append(str(row + 1))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            print(f"  {char} -> 层: {layer_coords[-1]}, 列: {column_coords[-1]}, 行: {row_coords[-1]}")

        # Step 3: 合并坐标
        combined_layer_coords = ''.join(layer_coords)
        combined_column_coords = ''.join(column_coords)
        combined_row_coords = ''.join(row_coords)
        print(f"Step 3: 合并坐标:")
        print(f"  层坐标: {combined_layer_coords}")
        print(f"  列坐标: {combined_column_coords}")
        print(f"  行坐标: {combined_row_coords}")

        # Step 4: 重组坐标
        combined_coords = combined_layer_coords + combined_column_coords + combined_row_coords
        new_coords = [combined_coords[i:i + 3] for i in range(0, len(combined_coords), 3)]
        print(f"Step 4: 重组坐标: {new_coords}")

        # Step 5: 根据新坐标获取密文
        text = ''
        print("Step 5: 根据新坐标获取密文:")
        for coord in new_coords:
            layer = int(coord[0]) - 1
            col = int(coord[1]) - 1
            row = int(coord[2]) - 1
            index = layer * 3 + row
            letter = cube[index][col]
            text += letter
            print(f"  {coord} -> {letter}")

        return text

    def get_encode_rule(self):
        return """加密方案概述：
        1. 过滤掉非字母字符并转换为大写。
        2. 将每个字母转换为对应的3D坐标。
        3. 将3D坐标重新组合，每三个字符一组。
        4. 根据新的坐标组在3D立方体中找到对应的字母，生成密文。
        5. 按照加密逻辑，将密文输出。
        """


    def decode(self, text):
        # 解题方案说明
        # print("解题方案概述:")
        # print("1. 将密文转换为大写。")
        # print("2. 将密文中的每个字母映射到Trifid立方体中对应的3D坐标。")
        # print("3. 将所有坐标连接成一个字符串。")
        # print("4. 将连接后的坐标组织成一个3xN的矩阵。")
        # print("5. 转置3xN的矩阵。")
        # print("6. 从转置后的矩阵中重构原始坐标。")
        # print("7. 将重构的坐标映射回相应的字母。")
        # print("8. 将字母组合成明文信息。\n")

        # 步骤1: 创建字母到坐标的映射
        letter_to_coords = {
            'A': (1, 1, 1), 'B': (1, 2, 1), 'C': (1, 3, 1),
            'D': (1, 1, 2), 'E': (1, 2, 2), 'F': (1, 3, 2),
            'G': (1, 1, 3), 'H': (1, 2, 3), 'I': (1, 3, 3),
            'J': (2, 1, 1), 'K': (2, 2, 1), 'L': (2, 3, 1),
            'M': (2, 1, 2), 'N': (2, 2, 2), 'O': (2, 3, 2),
            'P': (2, 1, 3), 'Q': (2, 2, 3), 'R': (2, 3, 3),
            'S': (3, 1, 1), 'T': (3, 2, 1), 'U': (3, 3, 1),
            'V': (3, 1, 2), 'W': (3, 2, 2), 'X': (3, 3, 2),
            'Y': (3, 1, 3), 'Z': (3, 2, 3), '.': (3, 3, 3)
        }

        # 步骤2: 创建坐标到字母的逆映射
        coords_to_letter = {v: k for k, v in letter_to_coords.items()}

        # 步骤3: 转换密文为大写
        text = text.upper()
        print(f"步骤1: 将密文转换为大写: {text}")

        # 步骤4: 将密文转换为坐标序列
        coords_sequence = ''.join([
            ''.join(map(str, letter_to_coords[char]))
            for char in text
        ])
        print(f"步骤2: 将每个字母转换为坐标: {coords_sequence}")

        # 步骤5: 计算每行的长度
        n = len(coords_sequence) // 3
        print(f"步骤3: 计算列数: {n}")

        # 步骤6: 将坐标序列重新组织成3xN的矩阵
        coords_matrix = [coords_sequence[i:i + n] for i in range(0, len(coords_sequence), n)]
        print("步骤4: 将坐标组织成3xN的矩阵:")
        for row in coords_matrix:
            print(row)

        # 步骤7: 转置坐标矩阵
        transposed_matrix = [''.join(row) for row in zip(*coords_matrix)]
        print("步骤5: 转置矩阵:")
        for row in transposed_matrix:
            print(row)

        # 定义3x3x3立方体的布局
        cube = [
            ['A', 'B', 'C'],  # 第1层
            ['D', 'E', 'F'],
            ['G', 'H', 'I'],
            ['J', 'K', 'L'],  # 第2层
            ['M', 'N', 'O'],
            ['P', 'Q', 'R'],
            ['S', 'T', 'U'],  # 第3层
            ['V', 'W', 'X'],
            ['Y', 'Z', '.']
        ]

        # 打印立方体布局
        print("立方体布局:")
        for layer in range(3):
            print(f"第{layer + 1}层:")
            for row in range(3):
                print(' '.join(cube[layer * 3 + row]))
            print()

        # 步骤8: 将转置后的坐标矩阵重新组合成坐标列表
        combined_coords = [tuple(map(int, [transposed_matrix[i][0], transposed_matrix[i][1], transposed_matrix[i][2]])) for
                        i in range(n)]
        print(f"步骤6: 组合转置后的坐标: {combined_coords}")

        # 步骤9: 将坐标转换回原文
        text = ''.join([coords_to_letter[coord] for coord in combined_coords])
        print(f"步骤7: 将坐标转换回字母: {text}")

        return text

    def get_decode_rule(self):
        return """解密方案概述：
        1. 将密文转换为大写字母。
        2. 将密文转换为对应的3D坐标。
        3. 将3D坐标重新组合，每三个字符一组。
        4. 根据新的坐标组在3D立方体中找到对应的字母，生成明文。
        5. 按照解密逻辑，将明文输出。
        """

    @property
    def cipher_name(self) -> str:
        return "TrifidCipher"
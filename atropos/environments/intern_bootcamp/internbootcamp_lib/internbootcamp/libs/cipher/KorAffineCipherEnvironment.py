from .BaseCipherEnvironment import BaseCipherEnvironment

class KorAffineCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = ''
        super().__init__(problem_description=problem_description,*args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule5_AffineCipher"
    
    def encode(self, text, **kwargs):
        # 将输入转换为大写字母并移除非字母字符
        text = ''.join([char.upper() for char in text if char.isalpha()])
        
        print(f"原始文本: {text}")
        print("开始加密过程...")
        print("使用仿射字母表: XMJQUDONPRGTVBWFAKSHZCYEIL")
        print("参数 A=3, B=5")
        
        alphabet = "XMJQUDONPRGTVBWFAKSHZCYEIL"
        A = 3
        B = 5
        n = len(alphabet)
        letter_to_index = {char: idx for idx, char in enumerate(alphabet)}
        encrypted_message = []
        
        for char in text:
            if char in letter_to_index:
                x = letter_to_index[char]
                y = (A * x + B) % n
                print(f"字符 {char} -> 位置 {x} -> 计算(3*{x}+5)%26={y} -> 加密为 {alphabet[y]}")
                encrypted_message.append(alphabet[y])
        
        result = ''.join(encrypted_message)
        print(f"加密完成，结果: {result}")
        return result

    def decode(self, text, **kwargs):
        print(f"加密文本: {text}")
        print("开始解密过程...")
        print("使用仿射字母表: XMJQUDONPRGTVBWFAKSHZCYEIL")
        print("参数 A=3, B=5, A的逆元=9")
        
        alphabet = "XMJQUDONPRGTVBWFAKSHZCYEIL"
        A = 3
        B = 5
        n = len(alphabet)
        letter_to_index = {char: idx for idx, char in enumerate(alphabet)}
        A_inv = 9
        decrypted_message = []
        
        for char in text:
            if char in letter_to_index:
                y = letter_to_index[char]
                x = A_inv * (y - B) % n
                print(f"字符 {char} -> 位置 {y} -> 计算9*({y}-5)%26={x} -> 解密为 {alphabet[x]}")
                decrypted_message.append(alphabet[x])
        
        result = ''.join(decrypted_message)
        print(f"解密完成，结果: {result}")
        return result

    def get_encode_rule(self, ):
        return """加密规则:
- 输入:
    - 明文: 仅包含大写字母的字符串，不含标点和空格
- 输出:
    - 密文: 大写字母字符串
- 准备:
    - 仿射字母表 = "XMJQUDONPRGTVBWFAKSHZCYEIL"
    - 将每个字母与其在仿射字母表中的位置关联（从0开始）:
        X->0, M->1, J->2, Q->3, U->4, D->5, O->6, N->7,
        P->8, R->9, G->10, T->11, V->12, B->13, W->14, F->15,
        A->16, K->17, S->18, H->19, Z->20, C->21, Y->22, E->23, I->24, L->25
    - A: 3
    - B: 5
    - A_inv: 9
- 加密步骤:
    - 对于每个明文字符p:
        - 设x为其在仿射字母表中的位置
        - 应用仿射编码函数计算y:
            - y = (Ax + B) mod 26
        - 在仿射字母表中找到位置y对应的字母，形成加密消息"""

    def get_decode_rule(self, ):
        return """解密规则:
    - 输入:
        - 密文: 大写字母字符串
    - 输出:
        - 明文: 大写字母字符串
    - 准备:
        - 仿射字母表 = "XMJQUDONPRGTVBWFAKSHZCYEIL"
        - 将每个字母与其在仿射字母表中的位置关联（从0开始）:
            X->0, M->1, J->2, Q->3, U->4, D->5, O->6, N->7,
            P->8, R->9, G->10, T->11, V->12, B->13, W->14, F->15,
            A->16, K->17, S->18, H->19, Z->20, C->21, Y->22, E->23, I->24, L->25
        - A: 3
        - B: 5
        - A_inv: 9
    - 解密步骤:
        - 对于每个密文字符c:
            - 设y为其在仿射字母表中的位置
            - 计算x:
                - x = A_inv * (y - B) % n
            - 用仿射字母表中位置x处的字母替换c，形成解密消息"""

        
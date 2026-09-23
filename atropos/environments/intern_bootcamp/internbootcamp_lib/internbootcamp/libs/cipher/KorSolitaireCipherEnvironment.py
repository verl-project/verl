from .BaseCipherEnvironment import BaseCipherEnvironment


import random
from typing import Callable, List
import functools

LETTERS = ['J', 'D', 'W', 'O', 'T', 'R', 'A', 'C', 'X', 'Q', 'M', 'F', 'Y', 
            'E', 'Z', 'G', 'U', 'K', 'P', 'V', 'B', 'S', 'H', 'N', 'L', 'I']

LETTER_TO_NUM_MAP = {}
for i, letter in enumerate(LETTERS):
    LETTER_TO_NUM_MAP[letter] = i

input_key = [9, 25, 44, 38, 40, 22, 11, 36, 13, 39, 18, 42, 10, 53, 26, 12, 1, 16, 3, 43, 37, 17, 30, 4, 28, 48, 27, 41, 32, 15, 47, 29, 20, 51, 6, 7, 52, 34, 35, 5, 50, 9, 54, 46, 23, 31, 24, 14, 8, 33, 2, 49, 45, 21]


def move_joker_a(cards: List[int]) -> list:
    return move_joker(len(cards) - 1, cards)

def move_joker_b(cards: List[int]) -> list:
    return move_joker(len(cards), cards)

def move_joker(joker: int, cards: List[int]) -> List[int]:
    def wraparound(n: int) -> int:
        if n >= len(cards):
            return n % len(cards) + 1
        return n

    cards = list(cards)
    jump = 2 if joker is len(cards) else 1
    index = cards.index(joker)
    cards.insert(wraparound(index + jump), cards.pop(index))
    return cards

def triple_cut_by_jokers(cards: List[int]) -> list:
    joker_a = len(cards) - 1
    joker_b = len(cards)
    return triple_cut((cards.index(joker_a), cards.index(joker_b)), cards)

def triple_cut(cut_indices: tuple, cards: list) -> List[int]:
    lower, higher = sorted(cut_indices)
    return cards[higher + 1:] + cards[lower:higher + 1] + cards[:lower]

def count_cut(cards: List[int]) -> List[int]:
    last = len(cards) - 1
    value = cards[last]
    if is_joker(value, cards):
        return list(cards)
    return cards[value:last] + cards[:value] + [cards[last]]

def get_keystream_value(cards: List[int]) -> int:
    index = cards[0] if not is_joker(cards[0], cards) else len(cards) - 1
    return cards[index]

def is_joker(value: int, cards: List[int]) -> bool:
    return value > len(cards) - 2


class KorSolitaireCipherEnvironment(BaseCipherEnvironment):
    def __init__(self, *args, **kwargs):
        problem_description = "Solitaire Cipher from kor-bench"
        super().__init__(problem_description, *args, **kwargs)
        
    @property
    def cipher_name(self) -> str:
        return "Kor_rule6_SolitaireCipher"
    
    def encode(self, text, **kwargs):
        # 将输入转换为大写字母,去除标点和空格
        text = ''.join([c.upper() for c in text if c.isalpha()])
        print(f"处理后的输入文本: {text}")
        
        print("开始加密过程:")
        cards = list(input_key)
        output = []
        
        for i, char in enumerate(text):
            print(f"\n处理第{i+1}个字符 '{char}':")
            # 获取字母位置值
            x = LETTER_TO_NUM_MAP[char]
            print(f"字符'{char}'在字母表中的位置是: {x}")
            
            # 生成密钥流值
            print("生成密钥流值:")
            print("1. 移动A王牌下移一位")
            cards = move_joker_a(cards)
            print("2. 移动B王牌下移两位") 
            cards = move_joker_b(cards)
            print("3. 以王牌为界进行三切")
            cards = triple_cut_by_jokers(cards)
            print("4. 根据底牌进行计数切牌")
            cards = count_cut(cards)
            
            y = get_keystream_value(cards)
            while is_joker(y, cards):
                print("获得的是王牌值,重新生成密钥流值")
                cards = move_joker_a(cards)
                cards = move_joker_b(cards)
                cards = triple_cut_by_jokers(cards)
                cards = count_cut(cards)
                y = get_keystream_value(cards)
            
            print(f"生成的密钥流值是: {y}")
            
            # 计算加密位置
            z = (x + y) % 26
            print(f"加密计算: ({x} + {y}) % 26 = {z}")
            
            # 获取加密字母
            cipher_char = LETTERS[z]
            print(f"加密后的字符是: {cipher_char}")
            output.append(cipher_char)
            
        result = ''.join(output)
        print(f"\n最终加密结果: {result}")
        return result

    def decode(self, text, **kwargs):
        print(f"开始解密密文: {text}")
        
        cards = list(input_key)
        output = []
        
        for i, char in enumerate(text):
            print(f"\n处理第{i+1}个字符 '{char}':")
            # 获取密文字母位置
            z = LETTER_TO_NUM_MAP[char]
            print(f"密文字符'{char}'在字母表中的位置是: {z}")
            
            # 生成密钥流值
            print("生成密钥流值:")
            print("1. 移动A王牌下移一位")
            cards = move_joker_a(cards)
            print("2. 移动B王牌下移两位")
            cards = move_joker_b(cards)
            print("3. 以王牌为界进行三切")
            cards = triple_cut_by_jokers(cards)
            print("4. 根据底牌进行计数切牌")
            cards = count_cut(cards)
            
            y = get_keystream_value(cards)
            while is_joker(y, cards):
                print("获得的是王牌值,重新生成密钥流值")
                cards = move_joker_a(cards)
                cards = move_joker_b(cards)
                cards = triple_cut_by_jokers(cards)
                cards = count_cut(cards)
                y = get_keystream_value(cards)
                
            print(f"生成的密钥流值是: {y}")
            
            # 计算原文位置
            x = (z - y) % 26
            print(f"解密计算: ({z} - {y}) % 26 = {x}")
            
            # 获取原文字母
            plain_char = LETTERS[x]
            print(f"解密后的字符是: {plain_char}")
            output.append(plain_char)
            
        result = ''.join(output)
        print(f"\n最终解密结果: {result}")
        return result

    def get_encode_rule(self, ):
        return """加密规则:
    - 输入:
        - 明文: 大写字母字符串,不含标点和空格
    - 输出:
        - 密文: 大写字母字符串
    - 准备:
        - LETTERS = ['J', 'D', 'W', 'O', 'T', 'R', 'A', 'C', 'X', 'Q', 'M', 'F', 'Y', 'E', 'Z', 'G', 'U', 'K', 'P', 'V', 'B', 'S', 'H', 'N', 'L', 'I']
        - 将每个字母与其在LETTERS中的位置关联(从0开始):
            J->0, D->1, W->2, O->3, T->4, R->5, A->6, C->7, X->8, Q->9, M->10, F->11, Y->12, E->13, Z->14, G->15, U->16, K->17, P->18, V->19, B->20, S->21, H->22, N->23, L->24, I->25
        - 初始卡牌序列:
            - 54张卡牌的列表,包括52张花色牌和2张可区分的王牌(A王牌和B王牌)。花色牌按四种花色顺序编号1-52,王牌值为53和54。
            - [9, 25, 44, 38, 40, 22, 11, 36, 13, 39, 18, 42, 10, 53, 26, 12, 1, 16, 3, 43, 37, 17, 30, 4, 28, 48, 27, 41, 32, 15, 47, 29, 20, 51, 6, 7, 52, 34, 35, 5, 50, 9, 54, 46, 23, 31, 24, 14, 8, 33, 2, 49, 45, 21]
        - 密钥流算法:
            该算法通过移动卡牌生成密钥流值。算法是确定性的,意味着密钥流值仅取决于卡牌的初始顺序。卡牌组被视为循环数组,允许需要移到底部的卡牌绕到顶部。
            
            执行以下步骤生成密钥流的一个字符:
            1. 找到A王牌并下移一位。如果是最后一张牌,则成为第二张牌。不能成为第一张牌。
            2. 找到B王牌并下移两位。如果是倒数第二张牌,则绕到第二位。如果是最后一张牌,则成为第三张牌。不能成为第一张牌。
            3. 进行"三切":用王牌作为边界将牌组分成三部分,然后交换顶部和底部。王牌本身及其之间的牌保持不变。
            4. 进行"计数切":检查牌组底牌。如果是王牌(53/54),其值固定为53。从牌组顶部取出该数量的牌,插入到最后一张牌的上方。
            5. 查看顶牌的值。同样,任何王牌计为53。计算该牌下方的位置数,使用该位置的牌值作为下一个密钥流值。如果计算出的牌是王牌,忽略它并重复密钥流算法。
            6. 返回生成的密钥流值
    - 加密步骤:
        - cards=初始卡牌序列
        - 对每个明文字符p:
            - 使用字母表将p转换为对应的位置值x(从0开始)
            - 使用初始卡牌序列为p生成密钥流值y:
                - y, cards = 密钥流算法(cards)
                - 该算法修改卡牌顺序,下次执行使用新顺序
            - 当密钥流值y加到位置值x时,应用模26运算得到z:
                - z=(y+x) % 26
            - 使用LETTERS列表返回对应位置z的字母
            - 将其附加到密文"""

    def get_decode_rule(self, ):
        return """解密规则:
    - 输入:
        - 密文: 大写字母字符串
    - 输出:
        - 明文: 大写字母字符串
    - 准备:
        - LETTERS = ['J', 'D', 'W', 'O', 'T', 'R', 'A', 'C', 'X', 'Q', 'M', 'F', 'Y', 'E', 'Z', 'G', 'U', 'K', 'P', 'V', 'B', 'S', 'H', 'N', 'L', 'I']
        - 将每个字母与其在LETTERS中的位置关联(从0开始):
            J->0, D->1, W->2, O->3, T->4, R->5, A->6, C->7, X->8, Q->9, M->10, F->11, Y->12, E->13, Z->14, G->15, U->16, K->17, P->18, V->19, B->20, S->21, H->22, N->23, L->24, I->25
        - 初始卡牌序列(与加密相同)
        - 密钥流算法(与加密相同)
    - 解密步骤(与加密步骤完全相反):
        - cards=初始卡牌序列
        - 对每个密文字符c:
            - 使用LETTERS将c转换为对应的位置值z(从0开始)
            - 为c生成密钥流值y:
                - y, cards = 密钥流算法(cards)
                - 该算法修改卡牌顺序,下次执行使用新顺序
            - 从密文字符c计算原始位置值x:
                - x=(z-y) mod 26
            - 使用LETTERS列表返回对应位置x的字母
            - 将其附加到解密明文"""

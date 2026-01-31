from abc import abstractmethod
from .BaseEnvironment import BaseEnvironment
from bootcamp_utils import catch_print


class BaseCipherEnvironment(BaseEnvironment):
    """
    Cipher环境基类
    """

    def __init__(self,problem_description, *args, **kwargs):
        super().__init__(problem_description, *args, **kwargs)
    def generator(self, plaintext: str = "", **kwargs) -> tuple:
        """
        根据问题描述生成或加载题目数据，这里使用给定的明文，并加密它。
        """
        self.plaintext = plaintext
        self.gen_steps,self.ciphertext = catch_print(self.encode,text=self.plaintext,**kwargs)
        for k,v in kwargs.items():
            setattr(self,k,v)
        return self.gen_steps,self.ciphertext

    def solver(self, **kwargs) -> tuple:
        """
        解题函数，执行解题逻辑并返回结果。
        """
        self.solve_steps,self.decodetext = catch_print(self.decode,text=self.ciphertext,**kwargs)
        return self.solve_steps,self.decodetext
    
    @abstractmethod
    def encode(self,text, **kwargs) -> str:
        """
        编码函数

        参数：
        text: 需要加密的文本（字符串）

        返回：
        加密后的文本（字符串）
        """
        pass

    
    @abstractmethod
    def decode(self, text, **kwargs) -> str:
        """
        解码函数
        
        参数：
        text: 需要解密的文本（字符串）

        返回：
        解密后的文本（字符串）
        """
        pass

    
    @abstractmethod
    def get_encode_rule(self) ->  str:
        """
        获取编码规则。
        """
        pass
    
    @abstractmethod
    def get_decode_rule(self) ->  str:
        """
        获取解码规则。
        """
        pass
    
    @property
    @abstractmethod
    def cipher_name(self) ->  str:
        """
        获取密码名称。
        """
        pass
    
    
    def check_solution(self, solution: object) -> bool:
        """
        检查提供的解答是否正确。
        """
        return solution == self.plaintext

    @property
    def is_solved(self) -> bool:
        """
        获取题目是否已被解决的状态。
        """
        return self.plaintext and self.decodetext
    
    def reset(self) -> None:
        """
        重置环境到初始状态。
        """
        self.plaintext = ""
        self.ciphertext = ""
        
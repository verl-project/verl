from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    def __init__(self, problem_description: str = None, *args, **kwargs):
        """
        构造函数接收题目描述和其他参数，初始化环境。

        :param problem_description: 题目描述文本
        :param args: 其他位置参数
        :param kwargs: 其他关键字参数
        """
        self.problem_description = problem_description
        self.kwargs = kwargs
        for k,v in self.kwargs.items():
            setattr(self, k, v)
    @abstractmethod
    def generator(self) -> object:
        """
        构建具体的题目实例，生成或加载题目数据。
        """
        pass

    @abstractmethod
    def solver(self) -> object:
        """
        解题函数，执行解题逻辑并返回结果。
        """
        pass

    @abstractmethod
    def check_solution(self, solution: object) -> bool:
        """
        检查提供的解答是否正确。

        :param solution: 提供的答案对象
        :return: 如果解答正确返回True，否则False
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        重置环境到初始状态。
        """
        pass

    @property
    @abstractmethod
    def is_solved(self) -> bool:
        """
        获取题目是否已被解决的状态。
        """
        pass
    
    def get_question(self) -> str:
        """
        提供生成题目的问题或解决题目的问题。
        """
        return "No question available."

    def get_question_following(self) -> str:
        """
        提供题目回答的可替换后缀，如回复格式规范等。
        """
        followings = []
        followings.append("""\n请以如下格式输出结果：
```json
{
    'final_answer': <your answer>
}
```""")
    
    def get_hint(self) -> str:
        """
        提供关于如何生成或解决问题的提示信息（可选）。
        """
        return "No hint available."

    def get_additional_resources(self) -> list:
        """
        返回可用于解决问题的额外资源列表（可选）。
        """
        return []

    def __str__(self) -> str:
        """
        返回环境的字符串表示，通常包含当前环境的状态信息。
        """
        return f"Environment for problem: {self.problem_description}"
"""# 谜题训练场开发任务

## 任务概述
你是一位资深程序员，我需要你帮我实现一个特定谜题的训练场环境类。这个类继承自`Basebootcamp`，用于生成谜题实例并验证解答。

## 背景说明
我正在开发一系列谜题训练场，每个训练场对应一个特定类型的谜题。训练场类命名为`{PuzzleName}bootcamp`，其中`PuzzleName`是谜题的名称。

每个训练场类主要提供两个核心功能：
1. 生成该谜题类型的问题实例
2. 验证用户对问题的回答是否正确

## 技术接口规范

### 类方法实现要求

```python
class {PuzzleName}bootcamp(Basebootcamp):
    def __init__(self, **params):
        \"\"\"
        请你自定义params，以保存该puzzle相关的参数，例如网格大小等，参数配有默认值
        \"\"\"
        pass
    
    def case_generator(self):
        \"\"\"
        生成谜题实例，提示：为保证谜题有解，可以先生成结果再对结果处理得到谜题
        返回：一个可JSON序列化的字典（避免包含set等无法通过json.dumps处理的数据结构）
        \"\"\"
        pass
    
    @staticmethod
    def prompt_func(question_case) -> str:
        \"\"\"
        将case_generator生成的谜题实例转换为文本形式的问题，问题中包含问题背景、对谜题规则的介绍、具体要解决的谜题实例、期望最终答案的格式，
        例如：你是xxxx，请你解答yyyy，规则如下：yyyy，最终答案放置在：zzzzz
        注意：规则应当描述详细，包括任务背景、具体任务操作规则、对题目格式和答案格式的含义介绍等

        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 格式化的问题字符串
            
        注意:
            1. 需考虑问题的格式，以便后续能正确提取
            2. 问题描述中应包含期望的答案格式说明，以便后续能正确提取，为了避免抽取时匹配出干扰项，请要求模型将答案放在特定标签，如[answer] [/answer]内
        \"\"\"
        pass
    
    @staticmethod
    def extract_output(output):
        \"\"\"
        从LLM的回复中提取符合格式要求的答案，如有多个，请抽取最后一个，避免使用re.search等只抽取第一个结果的方式。
        
        参数:
            output: LLM的完整输出（包含原始问题和回答）
            
        返回:
            提取的答案，若未找到符合格式的答案则返回None
        \"\"\"
        pass
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        \"\"\"
        验证提取的答案是否正确，注意一个问题可以能有多个解，按照谜题规则进行检验，不要直接匹配可能的答案。
        
        参数:
            solution: extract_output提取的答案
            identity: case_generator生成的谜题实例
            
        返回:
            bool: 答案是否正确
        \"\"\"
        pass
```

### 验证评分方法（基类已实现）

```python
@classmethod
def verify_score(cls, model_output, identity:dict, format_score=0.1) -> float:
    \"\"\"
    验证输出结果并评分。
    
    参数:
        model_output: 模型的完整输出
        identity: 谜题实例（由case_generator生成）
        format_score: 答案格式正确时的基础分数
    
    返回:
        float: 评分结果（0-1之间）
    \"\"\"
    score = 0. 
    try:
        extract_solution = cls.extract_output(model_output)
        if extract_solution is None:
            return score
        else:
            score = format_score # 格式正确时的基础分数
        if cls._verify_correction(extract_solution, identity):
            score = 1.  # 答案完全正确时的满分
    except Exception as e:
        # 处理异常情况
        pass
    return score
```

### 使用示例

```python
# 初始化谜题训练场
bootcamp = Puzzlebootcamp()

# 生成谜题实例
case = bootcamp.case_generator()

# 将谜题转换为文本问题
prompt = Puzzlebootcamp.prompt_func(case)

# 获取LLM对问题的解答
response = get_response(prompt, \"LLM\")

# 从完整对话中提取答案
extracted_output = Puzzlebootcamp.extract_output(prompt + response)

# 验证答案并评分
score = Puzzlebootcamp.verify_score(extracted_output, case)
```

## 你的任务
请根据以下谜题描述（谜题描述可能不完整，请先结合你的知识澄清规则），实现一个完整的谜题训练场类：

### 谜题描述
[Description]
# BBEH Temporal Sequence

In this task, the calendar schedules of a few people is provided for an entire
week. The blocked times for the calendar of each person is sampled randomly, and
is provided as text either by giving the times of the day when it is blocked or
giving the times of the day when it is free. The goal is to find: 1- the longest
meeting that can be scheduled for them, and 2- the number of possibilities for
such a meeting. These people may also have some constraints or we might have
some information about them that has to be taken into account for meeting
scheduling. Examples include: being in a different timezone than the other
participants, needing some free time before/after the meeting, being flexible to
miss a portion of the meeting, requiring some free time for lunch, only being
able to attend meetings of up to a certain length, being willing to free up some
specific parts of the day if needed, etc.

[/Description]
<example_1>
Sarah, Zoey, Amelia, Alice and Emma work from 9 to 5 their local time on weekdays. They want to have a meeting some time this week.

The schedule for Sarah's week in their local time is as follows:
Monday: Free only at the following times: from 9:00 to 10:25, from 10:30 to 11:45, from 12:25 to 13:00, from 13:30 to 17:00
Tuesday: Booked at the following times: from 10:00 to 10:50, from 12:00 to 13:25
Wednesday: Free only at the following times: from 9:00 to 10:00, from 11:25 to 12:40, from 14:05 to 14:55, from 15:20 to 16:50
Thursday: Booked at the following times: from 9:55 to 10:30, from 12:00 to 13:15
Friday: Booked at the following times: from 9:35 to 9:40, from 10:50 to 12:15, from 12:30 to 13:05, from 13:40 to 14:35


The schedule for Zoey's week in their local time is as follows:
Monday: Booked at the following times: from 10:25 to 10:45, from 12:10 to 13:00, from 13:40 to 14:40
Tuesday: Booked at the following times: from 9:45 to 10:35, from 10:50 to 11:10, from 12:10 to 12:55
Wednesday: Booked at the following times: from 10:25 to 11:15, from 11:55 to 13:15, from 14:20 to 15:20
Thursday: Free only at the following times: anytime before 10:10, from 10:50 to 12:00, from 12:25 to 13:40, from 14:30 to 17:00
Friday: Free only at the following times: anytime before 9:40, from 10:20 to 10:40, from 12:10 to 12:45, from 13:20 to 13:40, anytime after 13:50


The schedule for Amelia's week in their local time is as follows:
Monday: Booked at the following times: from 9:55 to 10:50, from 11:45 to 12:40, from 12:45 to 13:30, from 13:45 to 14:20
Tuesday: Free only at the following times: from 9:00 to 9:50, from 10:20 to 11:15, anytime after 11:55
Wednesday: Free only at the following times: anytime before 9:05, from 9:20 to 10:25, from 11:25 to 17:00
Thursday: Booked at the following times: from 9:40 to 10:40, from 11:35 to 12:20
Friday: Free only at the following times: from 9:00 to 9:10, from 9:20 to 10:10, from 11:10 to 11:15, from 11:30 to 17:00


The schedule for Alice's week in their local time is as follows:
Monday: Booked at the following times: from 9:40 to 11:10, from 12:15 to 12:40, from 12:45 to 13:25
Tuesday: Free only at the following times: from 10:05 to 10:20, anytime after 11:10
Wednesday: Booked at the following times: from 9:40 to 10:55, from 11:50 to 12:00
Thursday: Booked at the following times: from 10:30 to 10:55, from 10:55 to 11:45, from 13:00 to 13:50, from 14:20 to 15:15
Friday: Booked at the following times: from 10:05 to 10:20, from 10:40 to 11:15


The schedule for Emma's week in their local time is as follows:
Monday: Booked at the following times: anytime before 9:20, from 10:15 to 10:35, from 11:30 to 12:35, from 12:35 to 12:50
Tuesday: Booked at the following times: from 10:00 to 11:10, from 12:00 to 13:05
Wednesday: Booked at the following times: from 10:15 to 11:40, from 12:15 to 13:25
Thursday: Booked at the following times: from 9:05 to 9:15, from 9:50 to 11:05, from 11:45 to 12:25, from 12:50 to 12:55
Friday: Booked at the following times: from 9:00 to 10:20, from 11:10 to 11:20


Sarah is flexible to miss the last (at most) 10 minutes of the meeting
Zoey is fine to stay until 6pm to accommodate the meeting
Amelia needs a break for lunch on Wednesday from 12:30 to 1:00
Alice can clear any meeting of 30 minutes or less from their schedule to accommodate the meeting
Emma is in a timezone that is one hour ahead of other participants
Let X be the length of the longest meeting (in minutes) that can be scheduled if the meeting has to start on the hour or half hour. Also, let Y be number of possible options of such meetings. Compute X and Y and report them separated by a comma (e.g., 45, 3). If X is 0, report 0, 0.
</example_1>
<example_2>
Alice, Nancy and Victoria work from 9 to 5 their local time on weekdays. They want to have a meeting some time this week.

The schedule for Alice's week in their local time is as follows:
Monday: Booked at the following times: from 10:05 to 11:05, from 11:05 to 12:10, from 12:50 to 14:15, from 15:40 to 16:50
Tuesday: Booked at the following times: from 10:30 to 10:45, from 11:50 to 12:50
Wednesday: Booked at the following times: anytime before 10:20, from 10:40 to 12:00
Thursday: Booked at the following times: from 9:20 to 9:30, from 10:20 to 10:55, from 11:20 to 11:55, from 11:55 to 12:20
Friday: Booked at the following times: from 10:25 to 10:50, from 11:00 to 12:05, from 12:20 to 13:30, from 13:50 to 13:55


The schedule for Nancy's week in their local time is as follows:
Monday: Booked at the following times: from 9:50 to 10:50, from 10:50 to 10:55, from 12:10 to 12:20
Tuesday: Booked at the following times: from 9:45 to 10:05, from 11:15 to 11:50, from 13:10 to 14:05
Wednesday: Free only at the following times: from 9:00 to 10:00, from 10:50 to 11:30, anytime after 12:45
Thursday: Booked at the following times: anytime before 9:45, from 11:05 to 11:50, from 13:15 to 14:30
Friday: Free only at the following times: from 9:00 to 10:25, from 11:55 to 13:20, from 14:00 to 17:00


The schedule for Victoria's week in their local time is as follows:
Monday: Free only at the following times: anytime before 9:05, from 10:15 to 11:15, from 11:30 to 12:30, from 13:25 to 14:40, from 14:45 to 17:00
Tuesday: Booked at the following times: from 9:40 to 9:45, from 10:25 to 11:55
Wednesday: Free only at the following times: anytime before 9:45, from 10:15 to 11:35, from 12:55 to 13:45, from 14:45 to 15:25, from 15:55 to 17:00
Thursday: Free only at the following times: anytime before 10:30, from 11:20 to 11:55, from 13:25 to 17:00
Friday: Free only at the following times: from 9:00 to 10:25, from 11:20 to 11:45, from 12:05 to 12:55, from 14:25 to 14:30, from 16:00 to 17:00


Alice can clear any meeting of 30 minutes or less from their schedule to accommodate the meeting
Nancy prefers long meetings on Tuesdays and Fridays and accepts new meetings on these two days only if they are at least 30 minutes long
Victoria needs a break for lunch on Wednesday from 12:30 to 1:00
Let X be the length of the longest meeting (in minutes) that can be scheduled if the meeting has to start on the hour or half hour. Also, let Y be number of possible options of such meetings. Compute X and Y and report them separated by a comma (e.g., 45, 3). If X is 0, report 0, 0.
</example_2>
<example_3>
Charlotte, Yolanda, Isabella, Olivia and Alice work from 9 to 5 their local time on weekdays. They want to have a meeting some time this week.

The schedule for Charlotte's week in their local time is as follows:
Monday: Free only at the following times: anytime before 10:10, from 11:40 to 12:35, from 13:05 to 13:05, from 13:20 to 13:25, from 14:15 to 17:00
Tuesday: Free only at the following times: anytime before 10:15, from 11:35 to 12:30, from 13:10 to 14:25, from 14:40 to 17:00
Wednesday: Free only at the following times: anytime before 10:20, from 11:25 to 12:50, from 13:05 to 13:10, from 14:10 to 15:15, from 15:25 to 17:00
Thursday: Booked at the following times: from 10:10 to 11:40, from 11:45 to 12:50, from 14:20 to 14:50, from 15:45 to 16:40
Friday: Booked at the following times: from 9:55 to 10:20, from 11:20 to 12:25, from 12:55 to 13:55, from 14:55 to 15:30


The schedule for Yolanda's week in their local time is as follows:
Monday: Booked at the following times: from 9:05 to 9:25, from 10:00 to 11:05, from 12:30 to 12:40
Tuesday: Booked at the following times: from 10:05 to 11:25, from 11:45 to 11:55, from 12:20 to 13:05, from 13:35 to 15:05
Wednesday: Free only at the following times: from 9:00 to 9:20, from 9:50 to 11:20, from 12:20 to 12:55, anytime after 13:10
Thursday: Free only at the following times: anytime before 10:20, from 11:00 to 11:20, from 11:50 to 12:50, from 13:05 to 14:20, anytime after 14:45
Friday: Free only at the following times: anytime before 10:15, from 11:15 to 12:20, from 12:25 to 12:55, anytime after 13:55


The schedule for Isabella's week in their local time is as follows:
Monday: Free only at the following times: from 9:00 to 9:35, from 11:00 to 12:05, from 12:40 to 17:00
Tuesday: Free only at the following times: anytime before 10:00, from 10:45 to 10:55, from 11:40 to 11:50, from 12:30 to 17:00
Wednesday: Booked at the following times: from 9:20 to 10:10, from 11:40 to 12:35, from 14:05 to 14:30
Thursday: Booked at the following times: from 9:30 to 10:40, from 10:40 to 11:50, from 13:15 to 14:00
Friday: Free only at the following times: from 9:00 to 9:10, from 9:25 to 10:55, from 11:35 to 17:00


The schedule for Olivia's week in their local time is as follows:
Monday: Free only at the following times: from 9:00 to 9:45, from 10:15 to 10:15, from 11:05 to 11:45, from 12:40 to 13:15, from 14:30 to 17:00
Tuesday: Free only at the following times: anytime before 9:20, from 10:35 to 11:35, anytime after 13:00
Wednesday: Booked at the following times: from 9:05 to 9:20, from 9:30 to 10:10, from 10:45 to 11:05
Thursday: Booked at the following times: from 10:10 to 11:05, from 12:20 to 12:55, from 14:00 to 14:25, from 15:35 to 16:00
Friday: Free only at the following times: anytime before 10:30, from 11:00 to 11:05, from 12:05 to 13:15, from 13:30 to 17:00


The schedule for Alice's week in their local time is as follows:
Monday: Booked at the following times: from 9:20 to 9:30, from 10:10 to 10:25
Tuesday: Booked at the following times: from 9:20 to 9:35, from 10:40 to 11:40, from 13:00 to 13:40
Wednesday: Booked at the following times: from 10:00 to 10:10, from 11:40 to 12:30, from 12:40 to 13:55
Thursday: Booked at the following times: from 9:25 to 10:20, from 11:20 to 12:00, from 13:05 to 14:00
Friday: Booked at the following times: from 9:10 to 9:55, from 11:20 to 11:50


Charlotte is in a timezone that is one hour ahead of other participants
Yolanda is flexible to miss the last (at most) 10 minutes of the meeting
Isabella is fine to stay until 6pm to accommodate the meeting
Olivia prefers long meetings on Tuesdays and Fridays and accepts new meetings on these two days only if they are at least 30 minutes long
Alice can clear their morning schedule from 9:00 to 9:45 (if blocked) everyday if needed
Let X be the length of the longest meeting (in minutes) that can be scheduled if the meeting has to start on the hour or half hour. Also, let Y be number of possible options of such meetings. Compute X and Y and report them separated by a comma (e.g., 45, 3). If X is 0, report 0, 0.
</example_3>
<example_4>
Rachel, Nancy, Karen and Emma work from 9 to 5 their local time on weekdays. They want to have a meeting some time this week.

The schedule for Rachel's week in their local time is as follows:
Monday: Free only at the following times: from 9:00 to 10:00, from 11:30 to 12:40, anytime after 14:00
Tuesday: Free only at the following times: from 9:00 to 9:35, from 10:25 to 11:15, anytime after 12:00
Wednesday: Booked at the following times: from 9:15 to 9:35, from 10:05 to 10:45, from 11:05 to 12:35, from 13:50 to 14:45
Thursday: Free only at the following times: anytime before 9:35, from 10:30 to 10:55, from 11:55 to 12:10, from 12:50 to 17:00
Friday: Booked at the following times: from 9:15 to 10:05, from 10:35 to 11:55


The schedule for Nancy's week in their local time is as follows:
Monday: Free only at the following times: anytime before 9:50, from 11:15 to 12:45, from 13:20 to 14:15, anytime after 15:00
Tuesday: Booked at the following times: from 10:05 to 10:25, from 10:40 to 12:05, from 12:10 to 13:30, from 14:30 to 14:45
Wednesday: Booked at the following times: from 9:20 to 10:20, from 10:20 to 11:20, from 11:45 to 12:05, from 12:45 to 13:00
Thursday: Free only at the following times: from 9:00 to 9:10, from 10:05 to 11:15, from 12:25 to 13:35, anytime after 14:45
Friday: Free only at the following times: from 9:00 to 9:25, from 9:55 to 10:25, from 11:00 to 11:25, from 12:20 to 12:50, from 13:15 to 17:00


The schedule for Karen's week in their local time is as follows:
Monday: Booked at the following times: from 9:45 to 10:15, from 11:00 to 12:30
Tuesday: Booked at the following times: from 9:30 to 10:50, from 11:30 to 12:05, from 12:45 to 13:15
Wednesday: Booked at the following times: from 9:50 to 11:20, from 12:20 to 13:15, from 14:20 to 15:25
Thursday: Free only at the following times: anytime before 9:15, from 9:55 to 11:15, from 11:25 to 12:45, from 12:50 to 17:00
Friday: Booked at the following times: from 9:45 to 10:55, from 11:05 to 12:35, from 12:35 to 14:00


The schedule for Emma's week in their local time is as follows:
Monday: Booked at the following times: from 10:25 to 11:55, from 12:55 to 13:20
Tuesday: Free only at the following times: from 9:00 to 9:20, from 10:35 to 12:05, from 12:45 to 13:45, anytime after 15:05
Wednesday: Free only at the following times: anytime before 9:10, from 10:10 to 10:55, anytime after 11:35
Thursday: Booked at the following times: from 9:40 to 11:00, from 11:55 to 12:30
Friday: Booked at the following times: from 10:05 to 10:30, from 11:55 to 12:45, from 14:15 to 15:05, from 15:05 to 15:20


Rachel feels tired late in the day and requires the meeting to end before 4pm
Nancy is flexible to miss (at most) 10 minutes of the meeting
Karen is fine to stay until 6pm to accommodate the meeting
Emma prefers long meetings on Tuesdays and Fridays and accepts new meetings on these two days only if they are at least 30 minutes long
Let X be the length of the longest meeting (in minutes) that can be scheduled if the meeting has to start on the hour or half hour. Also, let Y be number of possible options of such meetings. Compute X and Y and report them separated by a comma (e.g., 45, 3). If X is 0, report 0, 0.
</example_4>
<example_5>
Isabella, Charlotte, Amelia, Nancy and Alice work from 9 to 5 their local time on weekdays. They want to have a meeting some time this week.

The schedule for Isabella's week in their local time is as follows:
Monday: Booked at the following times: from 9:10 to 9:40, from 9:45 to 10:10
Tuesday: Free only at the following times: anytime before 9:35, from 10:10 to 11:10, anytime after 11:40
Wednesday: Free only at the following times: from 9:00 to 9:45, from 10:00 to 11:05, from 11:40 to 12:50, from 13:15 to 13:20, anytime after 14:00
Thursday: Booked at the following times: from 10:10 to 10:55, from 11:30 to 12:10, from 12:55 to 13:15, from 13:30 to 13:40
Friday: Free only at the following times: from 9:00 to 9:10, from 9:55 to 10:30, from 10:45 to 10:50, from 11:20 to 12:15, anytime after 13:45


The schedule for Charlotte's week in their local time is as follows:
Monday: Free only at the following times: anytime before 10:25, from 10:50 to 11:30, from 12:10 to 12:55, from 13:10 to 13:35, from 14:10 to 17:00
Tuesday: Booked at the following times: from 9:25 to 10:30, from 11:50 to 12:30, from 13:15 to 14:15, from 14:25 to 14:30
Wednesday: Booked at the following times: from 9:05 to 9:50, from 10:30 to 11:45
Thursday: Free only at the following times: anytime before 9:40, from 9:45 to 10:05, from 11:10 to 11:35, from 12:50 to 14:05, from 14:55 to 17:00
Friday: Booked at the following times: from 10:00 to 10:35, from 10:55 to 11:10, from 12:15 to 13:00, from 13:40 to 14:15


The schedule for Amelia's week in their local time is as follows:
Monday: Free only at the following times: anytime before 10:15, from 10:20 to 11:30, from 12:25 to 12:35, anytime after 13:10
Tuesday: Booked at the following times: from 10:05 to 10:20, from 10:45 to 11:20, from 12:05 to 12:40, from 13:50 to 14:20
Wednesday: Free only at the following times: anytime before 9:50, from 11:10 to 12:30, from 13:40 to 14:40, from 15:45 to 17:00
Thursday: Free only at the following times: from 9:00 to 9:10, from 9:45 to 11:05, from 11:45 to 17:00
Friday: Free only at the following times: anytime before 10:05, from 10:20 to 11:10, from 11:55 to 12:55, anytime after 13:55


The schedule for Nancy's week in their local time is as follows:
Monday: Booked at the following times: from 10:05 to 10:40, from 12:00 to 12:45, from 13:50 to 15:00, from 16:10 to 16:25
Tuesday: Booked at the following times: from 9:35 to 10:50, from 11:40 to 12:50
Wednesday: Booked at the following times: from 10:15 to 10:45, from 11:15 to 12:45, from 14:05 to 14:15, from 14:15 to 15:00
Thursday: Booked at the following times: from 10:20 to 10:35, from 10:45 to 10:55, from 12:15 to 13:00, from 14:20 to 15:15
Friday: Free only at the following times: anytime before 9:20, from 10:15 to 10:45, from 12:15 to 13:00, from 13:05 to 13:25, from 13:55 to 17:00


The schedule for Alice's week in their local time is as follows:
Monday: Booked at the following times: from 10:30 to 11:05, from 11:40 to 12:55
Tuesday: Free only at the following times: from 9:00 to 9:35, from 10:40 to 11:30, from 11:55 to 13:20, from 13:55 to 17:00
Wednesday: Booked at the following times: from 10:25 to 11:30, from 12:30 to 12:40
Thursday: Free only at the following times: from 9:00 to 9:35, from 9:40 to 10:50, anytime after 11:50
Friday: Free only at the following times: from 9:25 to 9:35, from 11:00 to 12:15, anytime after 13:25


Isabella can clear any meeting of 30 minutes or less from their schedule to accommodate the meeting
Charlotte is flexible to miss the last (at most) 10 minutes of the meeting
Amelia needs a break for lunch on Wednesday from 12:30 to 1:00
Nancy feels tired late in the day and requires the meeting to end before 4pm
Alice is fine to stay until 6pm to accommodate the meeting
Let X be the length of the longest meeting (in minutes) that can be scheduled if the meeting has to start on the hour or half hour. Also, let Y be number of possible options of such meetings. Compute X and Y and report them separated by a comma (e.g., 45, 3). If X is 0, report 0, 0.
</example_5>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from internbootcamp.bootcamp.base import Basebootcamp
import random
import re
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import copy
from faker import Faker
import re

def generate(n):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    participants = []
    timezone_participant = None  # 跟踪时区不同的参与者

    # 生成每个参与者的数据
    for i in range(n):
        # 基础工作时间（本地时间）
        work_hours = {day: (540, 1080) for day in days}  # 9:00-18:00转换为分钟数
        
        # 随机生成占用时间（至少占50%工作时间）
        busy = generate_busy_times(work_hours)
        
        #先复制
        work_hours_org = copy.deepcopy(work_hours)
        busy_org = copy.deepcopy(busy)
        free_org = {day: [] for day in days}
        
        for day in work_hours_org:
            work_start, work_end = work_hours_org[day]
            # 基础可用时间
            free = calculate_free_intervals(work_start, work_end, busy_org[day])
            free_org[day] = free
        
        
        # 处理特殊约束
        constraint_type, constraint_value = apply_random_constraint(work_hours, busy)
        
        # 转换时区后的工作时间
        timezone_offset = constraint_value if constraint_type == "timezone" else 0
        converted_work = {day: (s - timezone_offset*60, e - timezone_offset*60) for day, (s, e) in work_hours.items()}
        
        # 转换占用时间到参考时区
        converted_busy = {
            day: [(s - timezone_offset*60, e - timezone_offset*60) for (s, e) in busy[day]]
            for day in days
        }
        
        # 计算可用时间（考虑约束）
        available = calculate_available(converted_work, converted_busy, constraint_type, constraint_value)
        
        participants.append({
            "work_hours_org":work_hours_org,
            "busy_org":busy_org,
            "free_org":free_org,
            "work_hours": converted_work,
            "busy": converted_busy,
            "constraint": (constraint_type, constraint_value),
            "available": available,
        })

    # 计算所有参与者的共同可用时间
    common_available = find_common_available(participants, days)
    
    # 找出最长会议时间和出现次数
    max_duration, count = 0, 0
    for day in days:
        for start, end in common_available.get(day, []):
            duration = end - start
            if duration > max_duration:
                max_duration = duration
                count = 1
            elif duration == max_duration:
                count += 1

    # 转换为可读的时间格式
    #puzzle = format_puzzle(participants)
    answer = {"max_duration": max_duration, "count": count}
    
    return participants, answer

def generate_busy_times(work_hours):
    busy = defaultdict(list)
    for day, (work_start, work_end) in work_hours.items():
        total_time = work_end - work_start
        target = random.randint(int(0.3*total_time), int(0.6*total_time))
        current = 0
        
        while current < target:
            duration = random.choice([20 , 30, 40, 50, 60,70, 80, 90, 100, 110, 130, 120, 140, 150])
            max_start = work_end - duration
            if max_start <= work_start:
                break
            start = random.randint(work_start, max_start)
            end = start + duration
            
            # 检查是否与现有占用冲突
            conflict = False
            for s, e in busy[day]:
                if not (end <= s or start >= e):
                    conflict = True
                    break
            if not conflict:
                busy[day].append((start, end))
                busy[day].sort()
                current += duration
    return busy

def apply_random_constraint(work_hours, busy):
    constraint_type = random.choice(["lunch", "timezone", "conflict", "buffer", "none"])
    if constraint_type == "lunch":
        # 随机找一天中午插入30分钟占用
        for day in random.sample(work_hours.keys(), 5):
            work_start, work_end = work_hours[day]
            lunch_start = 660  # 11:00
            lunch_end = 810    # 13:30
            free = calculate_free_intervals(work_start, work_end, busy[day])
            for s, e in free:
                overlap_s = max(s, lunch_start)
                overlap_e = min(e, lunch_end)
                if overlap_e - overlap_s >= 30:
                    # 插入占用
                    new_s = random.randint(overlap_s, overlap_e - 30)
                    new_e = new_s + 30
                    busy[day].append((new_s, new_e))
                    busy[day].sort()
                    return "lunch", [day,(new_s, new_e)]
        return "none", None
    elif constraint_type == "timezone":
        offset = random.choices([1, 2, 3, -1, -2, -3], weights=[5,3,1,5,3,1])[0]
        return "timezone", offset
    elif constraint_type == "conflict":
        return "conflict", random.choice([20, 40, 60])
    elif constraint_type == "buffer":
        return "buffer", random.choice([5, 10, 15])
    else:
        return "none", None

def calculate_available(work_hours, busy, constraint_type, constraint_value):
    available = defaultdict(list)
    for day in work_hours:
        work_start, work_end = work_hours[day]
        # 基础可用时间
        free = calculate_free_intervals(work_start, work_end, busy[day])
        # 应用约束
        if constraint_type == "conflict":
            max_conflict = constraint_value
            # 添加所有短于max_conflict的占用时间为可用
            for s, e in busy[day]:
                if e - s <= max_conflict:
                    free.append((s, e))
            free = merge_intervals(sorted(free))
        elif constraint_type == "buffer":
            buffer = constraint_value
            new_free = []
            for s, e in free:
                new_s = s + buffer
                new_e = e - buffer
                if new_s < new_e:
                    new_free.append((new_s, new_e))
            free = new_free
        available[day] = free
    return available

def calculate_free_intervals(start, end, busy):
    free = []
    prev_end = start
    for s, e in sorted(busy, key=lambda x: x[0]):
        if s > prev_end:
            free.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < end:
        free.append((prev_end, end))
    return free

def merge_intervals(intervals):
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(interval) for interval in merged]

def find_common_available(participants, days):
    common = {day: [] for day in days}
    for day in days:
        # 收集所有参与者的可用时间
        all_available = []
        for p in participants:
            all_available.append(p['available'][day])
        # 计算交集
        current_common = all_available[0]
        for avail in all_available[1:]:
            current_common = interval_intersection(current_common, avail)
        common[day] = current_common
    return common

def interval_intersection(a, b):
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        a_s, a_e = a[i]
        b_s, b_e = b[j]
        start = max(a_s, b_s)
        end = min(a_e, b_e)
        if start < end:
            result.append((start, end))
        if a_e < b_e:
            i += 1
        else:
            j += 1
    return result



def format_puzzle(participants):
    puzzle = []
    for i, p in enumerate(participants):
        constraint_type, value = p['constraint']
        timezone = 0
        if constraint_type == "timezone":
            timezone = value
        puzzle.append({
            "timezone_offset": timezone,
            "work_hours": p['work_hours'],
            "busy": p['busy'],
            "constraint": p['constraint']
        })
    return puzzle


def minutes_to_time(m):
    hours = m // 60
    minutes = m % 60
    return f"{hours}:{minutes:02d}"

def format_problem_string(participants, names):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    problem_lines = []
    
    # Header with participant names
    problem_lines.append(f"{', '.join(names)} work from 9:00 to 18:00 their local time on weekdays. They want to have a meeting some time this week.\n")
    
    # Process each participant's schedule
    for name, p in zip(names, participants):
        schedule_lines = []
        for day in days:
            work_start, work_end = p['work_hours_org'][day]
            busy = p['busy_org'][day]
            free = p['free_org'][day]
            
            # Determine which intervals to display
            if not busy:
                day_desc = f"{day}: Free all day"
            elif not free:
                day_desc = f"{day}: Booked all day"
            else:
                if len(free) <= len(busy):
                    intervals = free
                    prefix = "Free only at the following times:"
                else:
                    intervals = busy
                    prefix = "Booked at the following times:"
                
                interval_strs = []
                for s, e in intervals:
                    if s == work_start:
                        interval_strs.append(f"anytime before {minutes_to_time(e)}")
                    elif e == work_end:
                        interval_strs.append(f"anytime after {minutes_to_time(s)}")
                    else:
                        interval_strs.append(f"from {minutes_to_time(s)} to {minutes_to_time(e)}")
                day_desc = f"{day}: {prefix} {', '.join(interval_strs)}"
            
            schedule_lines.append(day_desc)
        
        # Combine schedule for this participant
        problem_lines.append(f"The schedule for {name}'s week in their local time is as follows:")
        problem_lines.extend(schedule_lines)
        problem_lines.append("")  # Empty line
    
    # Process constraints
    constraints = []
    for name, p in zip(names, participants):
        c_type, c_value = p['constraint']
        if c_type == "lunch":
            day, (s, e) = c_value
            constraints.append(f"{name} needs a break for lunch on {day} from {minutes_to_time(s)} to {minutes_to_time(e)}")
        elif c_type == "timezone":
            offset = c_value
            direction = "ahead" if offset > 0 else "behind"
            hours = abs(offset)
            plural = "hour" if hours == 1 else "hours"
            constraints.append(f"{name} is in a timezone that is {hours} {plural} {direction} of other participants")
        elif c_type == "conflict":
            constraints.append(f"{name} can clear any meeting of {c_value} minutes or less from their schedule to accommodate the meeting")
        elif c_type == "buffer":
            constraints.append(f"{name} requires {c_value} minutes free before and after the meeting")
    
    problem_lines.extend(constraints)
    problem_lines.append("\nLet X be the length of the longest meeting (in minutes) that can be scheduled. Also, let Y be number of possible options of such meetings. Compute X and Y and report them separated by a comma (e.g., 45, 3). If X is 0, report 0, 0.\n")
    
    return "\n".join(problem_lines)



def extract_answer(puzzle_response: str) -> tuple[int, int]:
    """
    从LLM响应中提取最后一个\boxed{X,Y}格式的答案
    
    参数:
        puzzle_response (str): LLM返回的完整文本（可能含多个答案）
        
    返回: 
        tuple[int, int]: 最后一个有效(X, Y)数值对
        
    示例:
        >>> extract_answer("中间答案 \boxed{30,2} 最终答案 \boxed{90,5}")
        (90, 5)
    """
    # 使用finditer获取所有匹配项（内存高效）
    pattern = r"\\boxed\{\s*(\d+)\s*,\s*(\d+)\s*\}"
    matches = [m for m in re.finditer(pattern, puzzle_response)]
    
    if not matches:
        return None
    
    # 取最后一个匹配项
    last_match = matches[-1]
    
    try:
        return (
            int(last_match.group(1)), 
            int(last_match.group(2))
        )
    except (ValueError, IndexError):
        return None






class BbehTemporalSequencesbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_num_people = params.get("max_num_people", 6)
        self.min_num_people = params.get("min_num_people", 3)
        self.fake = Faker()

    def case_generator(self) -> Dict:
        num_people = random.randint(self.min_num_people, self.max_num_people)
        names = [self.fake.name() for _ in range(num_people)]
        participants, answer = generate(num_people)
        
        case = {"people":{},"solution":[]}
        
        for idx, n in enumerate(names):
            chosen_participant = participants[idx]
            chosen_participant.pop('work_hours') 
            chosen_participant.pop('busy') 
            chosen_participant.pop('available') 
            case["people"][n] = chosen_participant
        
        case["solution"] = answer
        return case

    

    @staticmethod
    def prompt_func(question_case: Dict) -> str:
        names = []
        participants =[]
        for n,v in question_case["people"].items():
            names.append(n)
            participants.append(v)
        puzzle_content = format_problem_string(participants, names)
        
        problem_des = "You are an intelligent assistant specializing in solving custom puzzle problems. Now the puzzle is Temporal Sequence.\nIn this task, the calendar schedules of a few people is provided for an entire week. The blocked times for the calendar of each person is sampled randomly, and is provided as text either by giving the times of the day when it is blocked or giving the times of the day when it is free.\nThe goal is to find: 1- the longest meeting that can be scheduled for them, and 2- the number of possibilities for such a meeting.\nThese people may also have some constraints or we might have some information about them that has to be taken into account for meeting scheduling. Examples include: being in a different timezone than the other participants, needing some free time before and after the meeting, requiring some free time for lunch, etc.\n\n"
        
        instruction = """
**Instruction for Answer Submission**  
After analyzing the schedules and constraints, provide your answer in the following format:  
`Final Answer: \\boxed{X,Y}`  
Where:  
- `X` = Length of longest possible meeting (minutes)  
- `Y` = Number of possible options of such meetings    
""" 

        return  problem_des + puzzle_content + instruction

    @staticmethod
    def extract_output(output: str) -> Optional[tuple[int]]:
        ans  = extract_answer(output)
        if ans is None:
            return (-1,-1)
        else: 
            return ans
       

    @classmethod
    def _verify_correction(cls, solution: tuple[int], identity: Dict) -> bool:
        if solution[0] == -1:
            return False
        else:
            X = solution[0]
            Y = solution[1]
            return X == identity["solution"]["max_duration"] and Y == identity["solution"]["count"]


if __name__ == "__main__":
    bootcamp = BbehTemporalSequencesbootcamp()
    case = bootcamp.case_generator()
    print(bootcamp.prompt_func(case))
    example_answer = "\box{60,30}"
    print(f"answer: {example_answer}")
    score = bootcamp.verify_score(example_answer, case)
    print(f"score: {score}")
    print(f"ground_truth: {case['solution']}")

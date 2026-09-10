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
# BBEH Object Counting

Given a long list of objects that a person has, the model has to count the
number of items of a certain type. For examples, the items might belong to
classes (fruits, cell phones, cars) and the goal may be to count the total
number of cell phones that the person has. We consider two types of questions:
1- counting the sum of the number of items belonging to two different classes,
and 2- finding the absolute difference of the number of items belonging to two
different classes. To add to the difficulty of the task, some irrelevant
information, including the number of the same items that other people have,
are added to the input context so the problem becomes one of finding multiple
needles in a haystack.

[/Description]
<example_1>
My grandmother has 98 lemons. I have 49 crabs (here is why I have these many: initially I had 71 of them, then I lost 26 of them, but then I got another 63 of them, but then I lost 59 of them, so at the end I am left with 49 of them). I also have 31 bananas (here is why I have these many: initially I had 23 of them, then I lost 22 of them, but then I got another 55 of them, but then I lost 25 of them, so at the end I am left with 31 of them). I also have 61 hyenas (here is why I have these many: initially I had 66 of them, then I lost 30 of them, but then I got another 67 of them, but then I lost 42 of them, so at the end I am left with 61 of them). I also have 74 shrimps (feeling great about this). I also have 89 nebbiolo grapes (it is a funny story how I have 89 of them: initially I had 35, then I also got another 82 making it a total of 117, then I got another 10 making it a total of 127, but then I lost 38 of them so eventually I am left with 89 of them now). My cousin has 88 iphone 5 mobiles. My mother has 33 dates. I also have 38 cantaloupes (they were given to me as a gift). I also have 20 sony mobiles (it is a funny story how I have 20 of them: initially I had 43, then I also got another 52 making it a total of 95, then I got another 36 making it a total of 131, but then I lost 111 of them so eventually I am left with 20 of them now). I also have 49 samsung galaxy a35 mobiles (I really like them). I also have 57 sloths (it is a funny story how I have 57 of them: initially I had 28, then I also got another 19 making it a total of 47, then I got another 53 making it a total of 100, but then I lost 43 of them so eventually I am left with 57 of them now). I also have 57 kangaroos (received them last month). I also have 97 horses (here is why I have these many: initially I had 27 of them, then I lost 15 of them, but then I got another 87 of them, but then I lost 2 of them, so at the end I am left with 97 of them). My grandfather has 81 huawei pura 70 ultra mobiles. I also have 50 bats (it is a funny story how I have 50 of them: initially I had 33, then I also got another 44 making it a total of 77, then I got another 46 making it a total of 123, but then I lost 73 of them so eventually I am left with 50 of them now). I also have 56 motorola mobiles (feeling great about this). My mother also has 15 oneplus mobiles. My uncle has 88 huawei pura 70 pro mobiles. My brother has 72 iphone 6 mobiles. My brother also has 85 merlot grapes. My brother also has 58 pigs. My uncle also has 38 pinot noir grapes. My cousin also has 57 google (pixel) mobiles. My aunt has 14 mosquitoes. I also have 70 plums (it is a funny story how I have 70 of them: initially I had 21, then I also got another 84 making it a total of 105, then I got another 33 making it a total of 138, but then I lost 68 of them so eventually I am left with 70 of them now). My cousin also has 81 quinces. My uncle also has 77 cotton candy grapes. My grandfather also has 76 canarys. My mother also has 10 aardvarks. My sister has 28 prairie dogs. My grandfather also has 66 samsung galaxy z fold6 mobiles. My cousin also has 54 iphone 16 mobiles. My grandmother also has 1 jonagold apple. My uncle also has 30 raspberrys. My cousin also has 99 samsung galaxy c55 mobiles. My cousin also has 62 grenache grapes. My father has 75 groundhogs. My mother also has 59 penguins. My brother also has 42 huawei nova 12 pro mobiles. My mother also has 68 bears. I also have 16 dragonfruits (they were given to me as a gift). I also have 96 salmon (here is why I have these many: initially I had 36 of them, then I lost 13 of them, but then I got another 79 of them, but then I lost 6 of them, so at the end I am left with 96 of them). I also have 84 ravens (one day, I will have more of them). I also have 11 buffaloes (it is a funny story how I have 11 of them: initially I had 51, then I also got another 48 making it a total of 99, then I got another 30 making it a total of 129, but then I lost 118 of them so eventually I am left with 11 of them now). I also have 83 cobras (I am in love with them). I also have 57 grapefruits (one day, I will have more of them). My sister also has 13 huawei mate 70 mobiles. My uncle also has 36 honor mobiles. My sister also has 89 cattles. My grandmother also has 10 impalas. I also have 10 manatees (I like these a lot). I also have 30 badgers (happt to have them). My cousin also has 58 iphone 15 mobiles. My father also has 69 sharks. My uncle also has 2 huawei nova 12s mobiles. I also have 51 watermelons (maybe I just need some of them). My friend has 8 squirrels. My grandmother also has 10 huawei nova 13 mobiles. My sister also has 59 plantains. My uncle also has 40 ginger gold apples. My mother also has 65 cranberrys. My grandmother also has 78 leopards. My sister also has 28 samsung galaxy a55 mobiles. My aunt also has 12 mice. My grandmother also has 75 fuji apples. I also have 77 nectarines (happt to have them). I also have 98 autumn royal grapes (I really like them). I also have 29 vivo mobiles (one day, I will have more of them). My brother also has 71 pomegranates. My mother also has 12 deer. My brother also has 20 jellyfish. I also have 7 iguanas (here is why I have these many: initially I had 65 of them, then I lost 60 of them, but then I got another 81 of them, but then I lost 79 of them, so at the end I am left with 7 of them). My brother also has 96 samsung galaxy s24 mobiles. I also have 2 iphone 7 mobiles (happt to have them). I also have 12 iphone 13 mobiles (received them last week). My brother also has 77 oppo mobiles. My mother also has 9 mulberrys. My father also has 23 clementinefigs. My aunt also has 72 htc mobiles. I also have 2 huawei nova 11 se mobiles (it is a funny story how I have 2 of them: initially I had 42, then I also got another 93 making it a total of 135, then I got another 16 making it a total of 151, but then I lost 149 of them so eventually I am left with 2 of them now). I also have 27 ants (they are great). I also have 18 huawei pura 70 pro+ mobiles (here is why I have these many: initially I had 57 of them, then I lost 16 of them, but then I got another 79 of them, but then I lost 102 of them, so at the end I am left with 18 of them). I also have 46 iphone x mobiles (it is a funny story how I have 46 of them: initially I had 59, then I also got another 54 making it a total of 113, then I got another 88 making it a total of 201, but then I lost 155 of them so eventually I am left with 46 of them now). I also have 75 hummingbirds (here is why I have these many: initially I had 44 of them, then I lost 13 of them, but then I got another 84 of them, but then I lost 40 of them, so at the end I am left with 75 of them). My cousin also has 89 elderberryoranges. My grandmother also has 85 lemurs. My sister also has 30 hens.  What is the absolute difference of the total number of fruits and animals/insects that I have?
</example_1>
<example_2>
I have 92 double bass (here is why I have these many: initially I had 18 of them, then I lost 10 of them, but then I got another 90 of them, but then I lost 6 of them, so at the end I am left with 92 of them). I also have 63 nadaswarams (here is why I have these many: initially I had 14 of them, then I lost 12 of them, but then I got another 62 of them, but then I lost 1 of them, so at the end I am left with 63 of them). My aunt has 27 congas. I also have 92 honda civic cars (I am in love with them). I also have 25 mazda3 cars (feeling good about this). My aunt also has 39 tars. I also have 95 mazda cx-80 cars (here is why I have these many: initially I had 43 of them, then I lost 24 of them, but then I got another 84 of them, but then I lost 8 of them, so at the end I am left with 95 of them). My cousin has 74 toyota 4runnertoyota sequoia cars. My sister has 9 muscat grapes. My brother has 65 sruti boxes. My brother also has 11 pomegranates. My aunt also has 61 blueberrys. I also have 9 tablas (I really like them). My aunt also has 8 honda prologuq cars. I also have 22 lutes (it is a funny story how I have 22 of them: initially I had 63, then I also got another 64 making it a total of 127, then I got another 39 making it a total of 166, but then I lost 144 of them so eventually I am left with 22 of them now). My cousin also has 7 nectarines. My mother has 22 toyota sienna cars. I also have 19 mazda mx-30 cars (feeling great about this). I also have 14 buick enclave cars (it is a funny story how I have 14 of them: initially I had 23, then I also got another 23 making it a total of 46, then I got another 18 making it a total of 64, but then I lost 50 of them so eventually I am left with 14 of them now). I also have 11 flutes (received them last month). I also have 3 trombones (I like these a lot). My mother also has 58 balafons. My sister also has 18 lychees. I also have 8 rebecs (one day, I will have more of them). I also have 79 castanets (I do not think I need this much). I also have 47 raspberrys (here is why I have these many: initially I had 51 of them, then I lost 38 of them, but then I got another 39 of them, but then I lost 5 of them, so at the end I am left with 47 of them). My sister also has 7 mazda cx-30 cars. I also have 94 toyota highlander cars (it is a funny story how I have 94 of them: initially I had 27, then I also got another 19 making it a total of 46, then I got another 50 making it a total of 96, but then I lost 2 of them so eventually I am left with 94 of them now). My mother also has 52 honda passport cars. My uncle has 10 autumn royal grapes. My father has 33 chardonnay grapes. My friend has 84 mazda cx-90 cars. I also have 50 bouzoukis (they were given to me as a gift). I also have 69 plantains (received them last month). My aunt also has 13 pineapples. My sister also has 10 fuji apples. My uncle also has 1 pink lady apple. My aunt also has 23 dates. My aunt also has 39 honda odyssey cars. My sister also has 99 hyundai santa fe cars. My father also has 87 toyota rav4 cars. My cousin also has 2 clavichords. My aunt also has 45 subaru ascent cars. My mother also has 19 subaru impreza cars. My uncle also has 54 mazda cx-5 cars. I also have 78 electric bass (feeling great about this). I also have 81 nissan leaf cars (received them yesterday). I also have 25 bananas (received them yesterday). My cousin also has 26 recorders. I also have 8 drums (it is a funny story how I have 8 of them: initially I had 87, then I also got another 36 making it a total of 123, then I got another 65 making it a total of 188, but then I lost 180 of them so eventually I am left with 8 of them now). My sister also has 31 watermelons. I also have 4 peaches (happt to have them). My cousin also has 65 kalimbas. My grandfather has 1 toyota avalon cars. I also have 99 organs (feeling good about this). My father also has 89 honda cr-z cars. I also have 17 strawberrys (it is a funny story how I have 17 of them: initially I had 74, then I also got another 33 making it a total of 107, then I got another 18 making it a total of 125, but then I lost 108 of them so eventually I am left with 17 of them now). My grandmother has 70 oboes. I also have 8 ocarinas (they are great). I also have 67 n'gonis (very happy to have them). My cousin also has 96 quinces. My uncle also has 95 hammered dulcimers. My aunt also has 19 nissan titan cars. My mother also has 81 clementinefigs. My aunt also has 58 toyota tundra cars. My grandfather also has 33 ginger gold apples. I also have 58 honeydews (it is a funny story how I have 58 of them: initially I had 71, then I also got another 86 making it a total of 157, then I got another 16 making it a total of 173, but then I lost 115 of them so eventually I am left with 58 of them now). I also have 87 electric drums (here is why I have these many: initially I had 22 of them, then I lost 20 of them, but then I got another 90 of them, but then I lost 5 of them, so at the end I am left with 87 of them). My grandfather also has 93 mangoes. I also have 74 koras (it is a funny story how I have 74 of them: initially I had 48, then I also got another 10 making it a total of 58, then I got another 83 making it a total of 141, but then I lost 67 of them so eventually I am left with 74 of them now). My uncle also has 37 cherrys. My father also has 87 persimmons. My grandmother also has 11 hyundai elantra cars. My uncle also has 35 nissan armada cars. My mother also has 36 audi a6 cars. My brother also has 56 honeycrisp apples. My grandfather also has 76 nebbiolo grapes. My aunt also has 15 jeep gladiator cars. My grandfather also has 95 ouds. My father also has 52 mazda cx-3 cars. My uncle also has 81 harps. I also have 72 ukuleles (received them last month). My sister also has 97 nissan pathfinder cars. My grandmother also has 23 kiwis. My friend also has 32 pinot noir grapes. My grandfather also has 30 jazz apples. I also have 55 synthesizer keyboards (here is why I have these many: initially I had 56 of them, then I lost 12 of them, but then I got another 18 of them, but then I lost 7 of them, so at the end I am left with 55 of them). My brother also has 97 merlot grapes. My friend also has 28 nissan kicks cars. I also have 35 tanpuras (here is why I have these many: initially I had 77 of them, then I lost 46 of them, but then I got another 48 of them, but then I lost 44 of them, so at the end I am left with 35 of them).  What is the absolute difference of the total number of fruits and musical instruments that I have?
</example_2>
<example_3>
I have 43 baboons (it is a funny story how I have 43 of them: initially I had 75, then I also got another 22 making it a total of 97, then I got another 15 making it a total of 112, but then I lost 69 of them so eventually I am left with 43 of them now). My brother has 18 toyota tundra cars. My father has 83 electric pianos. I also have 38 tars (here is why I have these many: initially I had 90 of them, then I lost 18 of them, but then I got another 40 of them, but then I lost 74 of them, so at the end I am left with 38 of them). I also have 39 n'gonis (it is a funny story how I have 39 of them: initially I had 61, then I also got another 65 making it a total of 126, then I got another 50 making it a total of 176, but then I lost 137 of them so eventually I am left with 39 of them now). I also have 82 nissan murano cars (it is a funny story how I have 82 of them: initially I had 46, then I also got another 87 making it a total of 133, then I got another 38 making it a total of 171, but then I lost 89 of them so eventually I am left with 82 of them now). I also have 5 stingrays (received them last month). My friend has 54 bass guitars. I also have 93 toyota corolla cars (I really like them). My cousin has 16 mazda3 cars. My uncle has 6 nissan ariya cars. My father also has 68 bongos. I also have 13 ants (received them last week). My cousin also has 6 electric cellos. My friend also has 30 opossums. My mother has 53 gorillas. My sister has 21 honda accord cars. I also have 63 corgis (received them last month). I also have 25 mazda2 cars (they were given to me as a gift). My sister also has 7 clarinets. My cousin also has 4 honda pilot cars. I also have 47 french horns (feeling good about this). I also have 32 deer (very happy to have them). My friend also has 58 xylophones. My grandfather has 77 newts. My cousin also has 28 subaru outback cars. My mother also has 83 balafons. My father also has 98 banjoes. My sister also has 46 toyota avalon cars. I also have 92 groundhogs (they are great). My grandfather also has 32 crows. My grandmother has 25 cows. My friend also has 22 rabbits. I also have 26 nissan maxima cars (here is why I have these many: initially I had 83 of them, then I lost 67 of them, but then I got another 65 of them, but then I lost 55 of them, so at the end I am left with 26 of them). My brother also has 71 panpipes. I also have 95 mules (feeling great about this). My grandfather also has 49 kazoos. My father also has 29 dolphins. My cousin also has 96 nissan rogue cars. My uncle also has 74 rababs. My grandfather also has 57 rhinoceros. My aunt has 81 rebecs. My brother also has 83 nissan titan cars. My uncle also has 29 subaru ascent cars. I also have 66 ocarinas (they are great). My brother also has 30 mazda mx-30 cars. I also have 94 ouds (I do not think I need this much). My brother also has 50 parrots. My grandmother also has 99 honda passport cars. My grandmother also has 30 electric violins. My sister also has 45 honda cr-v cars. I also have 21 toyota prius cars (happt to have them). I also have 61 honda element cars (it is a funny story how I have 61 of them: initially I had 59, then I also got another 42 making it a total of 101, then I got another 70 making it a total of 171, but then I lost 110 of them so eventually I am left with 61 of them now). My mother also has 62 otters. My sister also has 28 dodge challenger cars. My grandfather also has 97 subaru impreza cars. I also have 41 trombones (here is why I have these many: initially I had 33 of them, then I lost 22 of them, but then I got another 86 of them, but then I lost 56 of them, so at the end I am left with 41 of them). My sister also has 64 prairie dogs. My friend also has 17 marmots. My uncle also has 94 seals. I also have 33 ford edge cars (here is why I have these many: initially I had 93 of them, then I lost 55 of them, but then I got another 18 of them, but then I lost 23 of them, so at the end I am left with 33 of them). I also have 16 turkeys (it is a funny story how I have 16 of them: initially I had 73, then I also got another 50 making it a total of 123, then I got another 58 making it a total of 181, but then I lost 165 of them so eventually I am left with 16 of them now). My sister also has 56 nissan pathfinder cars. I also have 48 nissan sentra cars (it is a funny story how I have 48 of them: initially I had 48, then I also got another 18 making it a total of 66, then I got another 93 making it a total of 159, but then I lost 111 of them so eventually I am left with 48 of them now). My cousin also has 51 toyota rav4 cars. I also have 62 toyota 4runnertoyota sequoia cars (I like these a lot). My grandfather also has 41 hummingbirds. I also have 48 clavichords (here is why I have these many: initially I had 42 of them, then I lost 25 of them, but then I got another 78 of them, but then I lost 47 of them, so at the end I am left with 48 of them). I also have 58 echidnas (one day, I will have more of them). My cousin also has 21 mandolins. My sister also has 78 kotoes. My father also has 19 armadillos. I also have 75 mazda cx-80 cars (it is a funny story how I have 75 of them: initially I had 25, then I also got another 66 making it a total of 91, then I got another 87 making it a total of 178, but then I lost 103 of them so eventually I am left with 75 of them now). My friend also has 30 snails. I also have 85 violins (maybe I just need some of them). My cousin also has 83 nissan kicks cars. My friend also has 26 theremins. My sister also has 70 sruti boxes. My aunt also has 19 honda civic cars. My brother also has 37 trout. I also have 76 kia niro cars (one day, I will have more of them). I also have 58 alligators (very happy to have them). My aunt also has 70 antelopes. My uncle also has 17 honda cr-z cars. My grandmother also has 97 mazda6 cars. I also have 85 hedgehogs (I do not think I need this much). My friend also has 95 toyota sienna cars. My friend also has 39 hyundai santa fe cars. My brother also has 93 timpanis. My aunt also has 45 foxes. My sister also has 82 hyenas. My brother also has 47 giraffes. My mother also has 98 mazda cx-5 cars. My mother also has 18 nissan armada cars. I also have 37 quails (hope to have more in the future). My uncle also has 82 swans. I also have 74 vibraphone, steelpans (it is a funny story how I have 74 of them: initially I had 60, then I also got another 80 making it a total of 140, then I got another 78 making it a total of 218, but then I lost 144 of them so eventually I am left with 74 of them now). My aunt also has 3 honda odyssey cars. My cousin also has 25 sheep. My mother also has 73 lizards. My brother also has 10 honda insight cars. I also have 8 trumpets (here is why I have these many: initially I had 87 of them, then I lost 27 of them, but then I got another 68 of them, but then I lost 120 of them, so at the end I am left with 8 of them). My mother also has 40 mridangams. I also have 1 lemur (received them yesterday). My mother also has 78 frogs. My uncle also has 75 nissan leaf cars. I also have 48 ukuleles (here is why I have these many: initially I had 99 of them, then I lost 77 of them, but then I got another 75 of them, but then I lost 49 of them, so at the end I am left with 48 of them). My brother also has 91 mazda cx-90 cars. I also have 70 goats (it is a funny story how I have 70 of them: initially I had 37, then I also got another 97 making it a total of 134, then I got another 44 making it a total of 178, but then I lost 108 of them so eventually I am left with 70 of them now). My grandmother also has 88 toyota tacoma cars. I also have 61 honda prologuq cars (it is a funny story how I have 61 of them: initially I had 100, then I also got another 30 making it a total of 130, then I got another 100 making it a total of 230, but then I lost 169 of them so eventually I am left with 61 of them now). My uncle also has 56 king cobra snakes. My grandfather also has 24 mice. My cousin also has 54 drums. I also have 21 tigers (received them last month). I also have 41 volkswagen golf cars (it is a funny story how I have 41 of them: initially I had 31, then I also got another 21 making it a total of 52, then I got another 18 making it a total of 70, but then I lost 29 of them so eventually I am left with 41 of them now). I also have 19 possums (it is a funny story how I have 19 of them: initially I had 37, then I also got another 58 making it a total of 95, then I got another 64 making it a total of 159, but then I lost 140 of them so eventually I am left with 19 of them now). My mother also has 30 macaws. My father also has 69 tablas. I also have 62 mazda cx-50 cars (here is why I have these many: initially I had 49 of them, then I lost 20 of them, but then I got another 50 of them, but then I lost 17 of them, so at the end I am left with 62 of them). My mother also has 84 nissan altima cars. I also have 45 jeep gladiator cars (feeling good about this). My cousin also has 20 pigeons. My uncle also has 51 tesla model s cars. My grandmother also has 55 vultures. I also have 83 hawks (it is a funny story how I have 83 of them: initially I had 52, then I also got another 13 making it a total of 65, then I got another 55 making it a total of 120, but then I lost 37 of them so eventually I am left with 83 of them now). My grandmother also has 26 beavers. My sister also has 22 audi a6 cars. I also have 71 electric guitars (here is why I have these many: initially I had 51 of them, then I lost 46 of them, but then I got another 98 of them, but then I lost 32 of them, so at the end I am left with 71 of them). I also have 84 tamburas (I like these a lot). My sister also has 49 ravens. I also have 27 wasps (they are great).  What is the sum of the total number of musical instruments and animals/insects that I have?
</example_3>
<example_4>
My friend has 51 subaru impreza cars. My grandmother has 85 honda cr-z cars. My friend also has 80 honda prologuq cars. My aunt has 13 honda odyssey cars. I have 97 alligators (here is why I have these many: initially I had 35 of them, then I lost 21 of them, but then I got another 97 of them, but then I lost 14 of them, so at the end I am left with 97 of them). My grandmother also has 9 mandolins. I also have 17 jeep gladiator cars (here is why I have these many: initially I had 43 of them, then I lost 10 of them, but then I got another 42 of them, but then I lost 58 of them, so at the end I am left with 17 of them). My grandfather has 42 kingsnakes. I also have 72 electric guitars (planning to have more in the future). I also have 87 dulcimers (it is a funny story how I have 87 of them: initially I had 36, then I also got another 55 making it a total of 91, then I got another 74 making it a total of 165, but then I lost 78 of them so eventually I am left with 87 of them now). My mother has 6 oboes. My cousin has 96 electric violins. I also have 5 organs (here is why I have these many: initially I had 53 of them, then I lost 52 of them, but then I got another 91 of them, but then I lost 87 of them, so at the end I am left with 5 of them). My grandmother also has 19 toyota c-hr cars. I also have 43 mazda cx-90 cars (I like these a lot). My sister has 17 volkswagen golf cars. My aunt also has 49 seals. My aunt also has 23 electric pianos. My brother has 39 nissan armada cars. My friend also has 21 maracas. My father has 10 mazda mx-30 cars. My cousin also has 93 audi a6 cars. I also have 95 chickens (it is a funny story how I have 95 of them: initially I had 94, then I also got another 68 making it a total of 162, then I got another 28 making it a total of 190, but then I lost 95 of them so eventually I am left with 95 of them now). My grandmother also has 11 nissan frintier cars. I also have 66 cheetahs (here is why I have these many: initially I had 17 of them, then I lost 15 of them, but then I got another 88 of them, but then I lost 24 of them, so at the end I am left with 66 of them). I also have 96 mules (received them yesterday). My uncle has 1 python snake. I also have 93 foxes (planning to have more in the future). My mother also has 82 wasps. My brother also has 64 tars. I also have 10 eagles (happt to have them). My friend also has 50 koras. My grandfather also has 91 shehnais. My brother also has 90 hammered dulcimers. My brother also has 34 glockenspiels. I also have 13 marimbas (they are great). I also have 69 rattlesnakes (it is a funny story how I have 69 of them: initially I had 94, then I also got another 61 making it a total of 155, then I got another 88 making it a total of 243, but then I lost 174 of them so eventually I am left with 69 of them now). I also have 60 newts (here is why I have these many: initially I had 80 of them, then I lost 56 of them, but then I got another 56 of them, but then I lost 20 of them, so at the end I am left with 60 of them). My grandmother also has 24 toyota tacoma cars. I also have 11 viper snakes (it is a funny story how I have 11 of them: initially I had 98, then I also got another 45 making it a total of 143, then I got another 74 making it a total of 217, but then I lost 206 of them so eventually I am left with 11 of them now). My grandmother also has 64 guinea pigs. My mother also has 90 double bass. I also have 52 nissan maxima cars (feeling great about this). I also have 48 toyota highlander cars (here is why I have these many: initially I had 15 of them, then I lost 14 of them, but then I got another 88 of them, but then I lost 41 of them, so at the end I am left with 48 of them). I also have 67 vultures (I really like them). My uncle also has 69 horses. My cousin also has 25 stingrays. I also have 74 toyota rav4 cars (it is a funny story how I have 74 of them: initially I had 53, then I also got another 62 making it a total of 115, then I got another 82 making it a total of 197, but then I lost 123 of them so eventually I am left with 74 of them now). My grandmother also has 91 armadillos. I also have 22 mazda cx-5 cars (feeling great about this). My mother also has 78 ouds. My uncle also has 83 timpanis. I also have 47 tambourines (here is why I have these many: initially I had 62 of them, then I lost 17 of them, but then I got another 73 of them, but then I lost 71 of them, so at the end I am left with 47 of them). I also have 58 mazda6 cars (it is a funny story how I have 58 of them: initially I had 11, then I also got another 14 making it a total of 25, then I got another 43 making it a total of 68, but then I lost 10 of them so eventually I am left with 58 of them now). My friend also has 7 vibraphone, steelpans. My uncle also has 14 mbiras. My mother also has 30 toyota 4runnertoyota sequoia cars. My mother also has 13 ford edge cars. I also have 51 toads (it is a funny story how I have 51 of them: initially I had 87, then I also got another 61 making it a total of 148, then I got another 73 making it a total of 221, but then I lost 170 of them so eventually I am left with 51 of them now). My grandfather also has 82 owls. My grandmother also has 27 trout. I also have 94 shamisens (here is why I have these many: initially I had 74 of them, then I lost 16 of them, but then I got another 53 of them, but then I lost 17 of them, so at the end I am left with 94 of them). I also have 55 zithers (they are great). My mother also has 32 bagpipes. My grandmother also has 55 pianos. I also have 61 hawks (received them last month). I also have 42 nissan rogue cars (here is why I have these many: initially I had 64 of them, then I lost 26 of them, but then I got another 84 of them, but then I lost 80 of them, so at the end I am left with 42 of them). I also have 31 raccoons (received them last week). I also have 85 pigs (they were given to me as a gift). I also have 12 macaws (planning to have more in the future). My uncle also has 88 boa snakes. I also have 49 clavichords (it is a funny story how I have 49 of them: initially I had 60, then I also got another 92 making it a total of 152, then I got another 76 making it a total of 228, but then I lost 179 of them so eventually I am left with 49 of them now). I also have 35 bansuris (here is why I have these many: initially I had 23 of them, then I lost 10 of them, but then I got another 27 of them, but then I lost 5 of them, so at the end I am left with 35 of them). I also have 96 bongos (I like these a lot). I also have 79 kia niro cars (it is a funny story how I have 79 of them: initially I had 56, then I also got another 57 making it a total of 113, then I got another 69 making it a total of 182, but then I lost 103 of them so eventually I am left with 79 of them now). My brother also has 64 rebecs. I also have 43 octopus (here is why I have these many: initially I had 42 of them, then I lost 34 of them, but then I got another 99 of them, but then I lost 64 of them, so at the end I am left with 43 of them). My brother also has 35 boars. My grandmother also has 63 hyundai santa fe cars. I also have 12 nissan leaf cars (it is a funny story how I have 12 of them: initially I had 88, then I also got another 72 making it a total of 160, then I got another 54 making it a total of 214, but then I lost 202 of them so eventually I am left with 12 of them now). I also have 16 trumpets (received them last month). My father also has 90 electric drums. My aunt also has 36 narwhals. I also have 31 tablas (I really like them). I also have 18 llamas (it is a funny story how I have 18 of them: initially I had 74, then I also got another 37 making it a total of 111, then I got another 89 making it a total of 200, but then I lost 182 of them so eventually I am left with 18 of them now). My grandfather also has 32 honda civic cars. I also have 73 guzhengs (I really like them). My cousin also has 19 banjoes. My grandmother also has 19 ravens. I also have 85 garter snakes (planning to have more in the future). My sister also has 29 monkeys. My grandmother also has 79 honda insight cars. My grandmother also has 89 kalimbas. My grandmother also has 30 kazoos. I also have 40 toyota tundra cars (it is a funny story how I have 40 of them: initially I had 26, then I also got another 62 making it a total of 88, then I got another 72 making it a total of 160, but then I lost 120 of them so eventually I am left with 40 of them now). I also have 64 mazda cx-3 cars (it is a funny story how I have 64 of them: initially I had 12, then I also got another 91 making it a total of 103, then I got another 32 making it a total of 135, but then I lost 71 of them so eventually I am left with 64 of them now). My grandmother also has 81 skunks. My brother also has 43 mallards. My father also has 50 autoharps. I also have 29 electric bass (maybe I just need some of them). I also have 32 nissan sentra cars (here is why I have these many: initially I had 91 of them, then I lost 33 of them, but then I got another 25 of them, but then I lost 51 of them, so at the end I am left with 32 of them). My uncle also has 13 seahorses. My mother also has 65 crows. My grandfather also has 85 saxophones. My uncle also has 50 nissan titan cars. I also have 48 didgeridoos (happt to have them). My friend also has 52 violins. I also have 7 shrews (here is why I have these many: initially I had 52 of them, then I lost 26 of them, but then I got another 28 of them, but then I lost 47 of them, so at the end I am left with 7 of them). My mother also has 58 clarinets. My father also has 72 yaks. I also have 26 geese (it is a funny story how I have 26 of them: initially I had 14, then I also got another 81 making it a total of 95, then I got another 10 making it a total of 105, but then I lost 79 of them so eventually I am left with 26 of them now). I also have 55 magpies (I am in love with them). My friend also has 20 buick enclave cars. My friend also has 80 honda passport cars. I also have 87 accordions (happt to have them). I also have 15 badgers (feeling great about this). My father also has 40 honda element cars. My father also has 27 peacocks. My sister also has 94 ducks. My father also has 18 hummingbirds. My brother also has 8 subaru ascent cars. I also have 70 mazda cx-30 cars (planning to have more in the future). I also have 29 wolves (it is a funny story how I have 29 of them: initially I had 42, then I also got another 96 making it a total of 138, then I got another 75 making it a total of 213, but then I lost 184 of them so eventually I am left with 29 of them now). My friend also has 55 king cobra snakes. My grandmother also has 73 pennywhistles. My friend also has 52 sitars. I also have 63 ants (here is why I have these many: initially I had 35 of them, then I lost 32 of them, but then I got another 74 of them, but then I lost 14 of them, so at the end I am left with 63 of them). My grandmother also has 72 cobras.  What is the sum of the total number of cars and musical instruments that I have?
</example_4>
<example_5>
I have 2 jeep gladiator cars (here is why I have these many: initially I had 23 of them, then I lost 16 of them, but then I got another 11 of them, but then I lost 16 of them, so at the end I am left with 2 of them). I also have 72 lg mobiles (it is a funny story how I have 72 of them: initially I had 71, then I also got another 27 making it a total of 98, then I got another 63 making it a total of 161, but then I lost 89 of them so eventually I am left with 72 of them now). I also have 52 toyota tacoma cars (I like these a lot). My grandmother has 91 mazda cx-50 cars. I also have 83 htc mobiles (feeling great about this). My cousin has 31 honda pilot cars. My cousin also has 83 oboes. My uncle has 99 kotoes. I also have 81 mazda mx-30 cars (it is a funny story how I have 81 of them: initially I had 62, then I also got another 59 making it a total of 121, then I got another 55 making it a total of 176, but then I lost 95 of them so eventually I am left with 81 of them now). My brother has 3 pipas. I also have 97 electric violins (I like these a lot). My grandmother also has 18 huawei pura 70 mobiles. I also have 28 motorola mobiles (it is a funny story how I have 28 of them: initially I had 21, then I also got another 29 making it a total of 50, then I got another 55 making it a total of 105, but then I lost 77 of them so eventually I am left with 28 of them now). I also have 60 mazda cx-70 cars (I am in love with them). My uncle also has 85 dodge challenger cars. My father has 3 tamburas. My father also has 62 mazda cx-80 cars. My uncle also has 78 nissan ariya cars. My grandmother also has 93 iphone 14 mobiles. My uncle also has 43 iphone 16 mobiles. My friend has 23 sony mobiles. My sister has 35 cellos. I also have 17 bagpipes (thrilled to have these). My mother has 77 mbiras. My aunt has 93 xylophones. I also have 49 subaru ascent cars (they were given to me as a gift). My grandmother also has 9 electric bass. My friend also has 42 djembes. My mother also has 23 huawei pura 70 ultra mobiles. My brother also has 25 accordions. My father also has 53 oneplus mobiles. My friend also has 37 audi a6 cars. My grandmother also has 4 nissan titan cars. My grandfather has 50 marimbas. I also have 76 honda element cars (one day, I will have more of them). I also have 94 subaru impreza cars (feeling good about this). My aunt also has 45 banjoes. My aunt also has 6 sazzes. My cousin also has 86 iphone 4 mobiles. My father also has 87 toyota prius cars. I also have 86 bansuris (here is why I have these many: initially I had 73 of them, then I lost 66 of them, but then I got another 97 of them, but then I lost 18 of them, so at the end I am left with 86 of them). My grandmother also has 20 iphone 8 mobiles. I also have 65 nissan rogue cars (received them last week). I also have 74 mazda cx-5 cars (it is a funny story how I have 74 of them: initially I had 36, then I also got another 83 making it a total of 119, then I got another 100 making it a total of 219, but then I lost 145 of them so eventually I am left with 74 of them now). My cousin also has 73 huawei nova 11 se mobiles. My aunt also has 18 realme mobiles. My brother also has 19 hammered dulcimers. I also have 50 glockenspiels (it is a funny story how I have 50 of them: initially I had 82, then I also got another 99 making it a total of 181, then I got another 14 making it a total of 195, but then I lost 145 of them so eventually I am left with 50 of them now). My friend also has 12 honor mobiles. I also have 77 samsung galaxy a15 mobiles (feeling good about this). My father also has 5 nissan pathfinder cars. My father also has 55 tars. My grandmother also has 51 nissan altima cars. My grandfather also has 76 mazda6 cars. My uncle also has 63 tecno mobiles. I also have 29 ukuleles (here is why I have these many: initially I had 54 of them, then I lost 16 of them, but then I got another 58 of them, but then I lost 67 of them, so at the end I am left with 29 of them). My father also has 41 honda accord cars. My mother also has 46 iphone 13 mobiles. I also have 28 recorders (maybe I just need some of them). My friend also has 4 didgeridoos. My friend also has 76 autoharps. I also have 50 saxophones (I like these a lot). I also have 5 buick enclave cars (it is a funny story how I have 5 of them: initially I had 27, then I also got another 43 making it a total of 70, then I got another 91 making it a total of 161, but then I lost 156 of them so eventually I am left with 5 of them now). I also have 1 mazda3 cars (they were given to me as a gift). I also have 91 iphone 6 mobiles (planning to have more in the future). I also have 33 ouds (I am in love with them). My sister also has 92 trumpets. My cousin also has 84 toyota corolla cars. My grandfather also has 66 xiaomi mobiles. I also have 79 iphone 15 mobiles (they are great). My grandmother also has 58 iphone 5 mobiles. I also have 35 theremins (here is why I have these many: initially I had 31 of them, then I lost 11 of them, but then I got another 62 of them, but then I lost 47 of them, so at the end I am left with 35 of them). My grandfather also has 79 toyota sienna cars. My grandfather also has 52 hyundai santa fe cars. My friend also has 85 synthesizer keyboards. I also have 76 nissan frintier cars (here is why I have these many: initially I had 16 of them, then I lost 10 of them, but then I got another 82 of them, but then I lost 12 of them, so at the end I am left with 76 of them). I also have 42 violins (it is a funny story how I have 42 of them: initially I had 53, then I also got another 28 making it a total of 81, then I got another 34 making it a total of 115, but then I lost 73 of them so eventually I am left with 42 of them now). My grandmother also has 43 drums. My sister also has 5 pianos.  What is the absolute difference of the total number of mobiles and cars that I have?
</example_5>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from internbootcamp.bootcamp.base import Basebootcamp
import random
import re
from typing import Dict, Any, List

class Bbehobjectcountingbootcamp(Basebootcamp):
    
    category_vocab = {
            'fruits': [
                'apples', 'bananas', 'grapes', 'lemons', 'oranges', 'pears',
                'plums', 'kiwis', 'mangoes', 'pineapples', 'watermelons',
                'cantaloupes', 'dates', 'peaches', 'strawberries', 'raspberries',
                'blueberries', 'blackberries', 'cherries', 'pomegranates',
                'lychees', 'passion fruits', 'dragonfruits', 'persimmons',
                'star fruits', 'jackfruits', 'custard apples', 'soursops'
            ],
            'animals/insects': [
                'dogs', 'cats', 'elephants', 'lions', 'tigers', 'bears', 'wolves',
                'zebras', 'giraffes', 'monkeys', 'kangaroos', 'koalas', 'pandas',
                'rhinoceroses', 'hippopotamuses', 'hyenas', 'cheetahs', 'leopards',
                'antelopes', 'buffaloes', 'crocodiles', 'snakes', 'spiders', 'bees',
                'mosquitoes', 'butterflies', 'dragonflies', 'scorpions', 'ladybugs',
                'grasshoppers', 'crickets', 'fireflies', 'cockroaches'
            ],
            'cars': [
                'Toyota Corolla', 'Honda Civic', 'Ford Mustang', 'Chevrolet Silverado',
                'Nissan Altima', 'Volkswagen Golf', 'BMW 3 Series', 'Mercedes-Benz C-Class',
                'Jeep Gladiator', 'Honda CR-V', 'Mazda CX-5', 'Toyota RAV4', 'Tesla Model 3',
                'Hyundai Tucson', 'Kia Sportage', 'Subaru Outback', 'Ford F-150',
                'Toyota Highlander', 'Honda Accord', 'Nissan Rogue', 'Chevrolet Equinox',
                'Toyota Camry', 'Hyundai Elantra', 'Jeep Wrangler', 'Ford Explorer',
                'Toyota Tacoma', 'Honda Pilot', 'Nissan Sentra', 'Mazda3', 'Subaru Forester'
            ],
            'mobiles': [
                'iPhone 5', 'iPhone 7', 'iPhone 13', 'iPhone 15', 'Samsung Galaxy S21',
                'Samsung Galaxy Z Fold6', 'Huawei P40', 'Huawei Pura 70 Ultra',
                'Google Pixel 6', 'OnePlus 9', 'Xiaomi Mi 11', 'Sony Xperia',
                'LG Velvet', 'Motorola Edge', 'Huawei Nova 11 SE', 'Samsung Galaxy A35',
                'Google Pixel 7a', 'Xiaomi Redmi Note 13', 'Oppo Reno 10',
                'Vivo V30 Pro', 'Realme GT 6', 'Nokia X30', 'Asus Zenfone 11',
                'iPhone X', 'Samsung Galaxy C55', 'Huawei Mate 70', 'Google Pixel 8 Pro'
            ],
            'musical instruments': [
                'grand piano', 'acoustic guitar', 'electric guitar', 'violin',
                'cello', 'double bass', 'flute', 'clarinet', 'saxophone', 'trumpet',
                'trombone', 'french horn', 'tuba', 'drum set', 'conga drums',
                'bongo drums', 'marimba', 'xylophone', 'harp', 'ukulele',
                'mandolin', 'banjo', 'accordion', 'harmonica', 'sitar', 'oud',
                'didgeridoo', 'kalimba', 'theremin', 'synthesizer'
            ]
        }
    def __init__(self, categories=None, min_items=3, max_items=7, 
                 min_count=5, max_count=200, max_other_items=8, 
                 max_other_people=15):
        super().__init__()
        self.categories = categories or [
            'fruits', 'animals/insects', 'cars', 
            'mobiles', 'musical instruments'
        ]
        self.min_items = min_items
        self.max_items = max_items
        self.min_count = min_count
        self.max_count = max_count
        self.max_other_items = max_other_items
        self.max_other_people = max_other_people

    def _generate_story(self, num: int) -> str:
        """生成数学正确的背景故事，包含至少三个步骤"""
        steps = random.choices(['gain', 'loss'], k=3)  # 确保三种操作
        current = num
        history = []
        
        # 逆向工程生成步骤
        for step in reversed(steps):
            if step == 'gain':
                delta = random.randint(1, max(10, current//2))
                history.insert(0, ('loss', delta))
                current += delta
            else:
                delta = random.randint(1, current-1) if current > 1 else 1
                history.insert(0, ('gain', delta))
                current -= delta
        
        # 构建故事
        parts = [f"initially I had {current}"]
        temp = current
        for action, value in history:
            if action == 'gain':
                temp += value
                parts.append(f"got {value} more")
            else:
                temp -= value
                parts.append(f"lost {value}")
        
        # 添加随机转折
        endings = [
            "but later found discrepancies",
            "after quality control checks",
            "due to unexpected circumstances",
            "following inventory adjustments"
        ]
        story = ', '.join(parts) + f", {random.choice(endings)} finally ending with {num}"
        return f"({story})"

    def case_generator(self) -> Dict[str, Any]:
        # 确保选择不同的类别
        cat1, cat2 = random.sample(self.categories, 2)
        operation = random.choice(['sum', 'difference'])
        
        def generate_category(category: str) -> (List[str], int):
            item_count = random.randint(self.min_items, self.max_items)
            items = []
            total = 0
            for _ in range(item_count):
                num = random.randint(self.min_count, self.max_count)
                item = random.choice(self.category_vocab[category])
                story = self._generate_story(num)
                items.append(f"I have {num} {item} {story}.")
                total += num
            return items, total
        
        # 生成目标类别物品
        items1, sum1 = generate_category(cat1)
        items2, sum2 = generate_category(cat2)
        
        # 生成干扰物品（其他类别）
        other_items = []
        for cat in set(self.categories) - {cat1, cat2}:
            for _ in range(random.randint(0, self.max_other_items)):
                num = random.randint(1, self.max_count)
                item = random.choice(self.category_vocab[cat])
                other_items.append(f"I have {num} {item} {self._generate_story(num)}.")
                
        # 生成其他人物物品（至少5个）
        people_roles = ['grandmother', 'grandfather', 'mother', 'father',
                       'sister', 'brother', 'uncle', 'aunt', 'cousin', 'friend']
        other_people_items = []
        for _ in range(random.randint(self.max_other_people//2, self.max_other_people)):
            person = random.choice(people_roles)
            category = random.choice(self.categories)
            num = random.randint(1, 300)
            item = random.choice(self.category_vocab[category])
            other_people_items.append(f"My {person} has {num} {item}.")
        
        # 混合并打乱所有条目
        all_items = items1 + items2 + other_items + other_people_items
        random.shuffle(all_items)
        
        # 计算正确答案
        if operation == 'sum':
            answer = sum1 + sum2
        else:
            answer = abs(sum1 - sum2)
        
        return {
            'items': all_items,
            'categories': (cat1, cat2),
            'operation': operation,
            'correct_answer': answer,
            '_sums': (sum1, sum2)  # 用于调试
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        cat1, cat2 = question_case['categories']
        operation = question_case['operation']
        example1 = ', '.join(Bbehobjectcountingbootcamp.category_vocab[cat1][:3])
        example2 = ', '.join(Bbehobjectcountingbootcamp.category_vocab[cat2][:3])
        inventory_list = "\n".join(question_case['items'])
        operation_description = 'sum' if operation == 'sum' else 'absolute difference'
        
        return f"""You are an inventory analyst. Calculate the {operation} between my total {cat1} and {cat2} based on:

**Rules**
1. {cat1} include: {example1}, etc.
2. {cat2} include: {example2}, etc.

**Inventory List**
{inventory_list}

**Required Format**
Calculate the {operation_description} and put your final answer within [answer][/answer] tags."""

    @staticmethod
    def extract_output(output: str) -> int:
        # 处理多个可能的答案格式
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        # 提取最后一个答案并清洗格式
        raw_answer = matches[-1].strip().replace(',', '').replace(' ', '')
        # 处理可能的数学表达式
        if '=' in raw_answer:
            raw_answer = raw_answer.split('=')[-1]
        # 提取所有数字候选
        numbers = re.findall(r'-?\d+', raw_answer)
        return int(numbers[-1]) if numbers else None

    @classmethod
    def _verify_correction(cls, solution: int, identity: Dict) -> bool:
        return solution == identity['correct_answer']

from typing import Dict, Any

def run_tests():
    # 初始化训练场对象
    bootcamp = Bbehobjectcountingbootcamp()

    print("Running tests for BbehObjectCountingbootcamp...\n")

    # 测试 _generate_story 方法
    print("Testing _generate_story...")
    num = 50
    story = bootcamp._generate_story(num)
    assert isinstance(story, str), "_generate_story should return a string"
    assert str(num) in story, "The final number should appear in the story"
    print("_generate_story passed.\n")

    # 测试 case_generator 方法
    print("Testing case_generator...")
    case = bootcamp.case_generator()
    assert 'items' in case, "case should contain 'items'"
    assert 'categories' in case, "case should contain 'categories'"
    assert 'operation' in case, "case should contain 'operation'"
    assert 'correct_answer' in case, "case should contain 'correct_answer'"

    assert isinstance(case['items'], list), "'items' should be a list"
    assert len(case['items']) > 0, "'items' should not be empty"

    cat1, cat2 = case['categories']
    assert cat1 in bootcamp.categories, f"{cat1} should be a valid category"
    assert cat2 in bootcamp.categories, f"{cat2} should be a valid category"
    assert cat1 != cat2, "Categories should be different"

    assert case['operation'] in ['sum', 'difference'], "Operation should be 'sum' or 'difference'"
    assert isinstance(case['correct_answer'], int), "Correct answer should be an integer"
    print("case_generator passed.\n")

    # 测试 prompt_func 方法
    print("Testing prompt_func...")
    prompt = bootcamp.prompt_func(case)
    assert isinstance(prompt, str), "prompt_func should return a string"
    assert '[answer]' in prompt, "Prompt should contain [answer] tag"
    assert case['categories'][0] in prompt, "Prompt should include the first category"
    assert case['categories'][1] in prompt, "Prompt should include the second category"
    print("prompt_func passed.\n")

    # 测试 extract_output 方法
    print("Testing extract_output...")
    output = "Some text [answer]1234[/answer] more text"
    extracted = bootcamp.extract_output(output)
    assert extracted == 1234, "extract_output should correctly extract the answer"

    output_with_comma = "Some text [answer]1,234[/answer] more text"
    extracted = bootcamp.extract_output(output_with_comma)
    assert extracted == 1234, "extract_output should handle commas"

    output_with_equals = "Some text [answer]result=1234[/answer] more text"
    extracted = bootcamp.extract_output(output_with_equals)
    assert extracted == 1234, "extract_output should handle equal signs"

    no_answer_output = "Some text without answer tags"
    extracted = bootcamp.extract_output(no_answer_output)
    assert extracted is None, "extract_output should return None when no answer is found"
    print("extract_output passed.\n")

    # 测试 _verify_correction 方法
    print("Testing _verify_correction...")
    correct_answer = case['correct_answer']
    assert bootcamp._verify_correction(correct_answer, case), "Correct answer should pass verification"

    wrong_answer = correct_answer + 1
    assert not bootcamp._verify_correction(wrong_answer, case), "Wrong answer should fail verification"
    print("_verify_correction passed.\n")

    print("All tests passed!")
    
    print(prompt)
    print("So, the correct answer is:", correct_answer, ".")

if __name__ == "__main__":
    run_tests()
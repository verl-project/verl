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
# BBEH Object Properties

In this task, an initial collection of objects with different properties (color,
size, origin, smell, and material) are provided (e.g., a extra-small blue
Canadian jar made of glass and with a smell of rose). Then, the collection goes
through several updates corresponding to adding, removing or editing some of the
objects. The updates are explained in the prompt and the models require a full
grasp of the object properties to identify what changes to the collection must
be made for each update. A simple example of an update is as follows:

    My dad threw away all objects of a certain color from my collection.
    After this, my collection only had 5 blue objects and 3 white objects.

For the above update, one has to find which color has been removed by comparing
the new colors with the object colors in the previous collection, and then
update the collection accordingly. The set of updates that the collection goes
through in each of the examples are randomly selected from a large set of
possible changes. At the end, a question is asked about the final collection.
The question is either an **either** question in which we ask how many items in
the final collection have property 1 or property 2, ... (e.g., how many items
are either blue or small), or a **neither** question in which we ask how many
items neither have property 1 nor property 2, ... (e.g., how many items are not
blue and are not small).

[/Description]
<example_1>
I had a collection of 35 weird items that went through a few changes. Initially, I had an extra-large Spanish calculator made of plastic with a smell of vinegar, a medium Spanish hat made of steel with a smell of pine needles, a medium Turkish jar made of steel with a smell of freshly cut grass, an extra-extra-large French plier made of plastic with a smell of pine needles, an extra-extra-large Russian sunglasses made of steel with a smell of pine needles, a medium Turkish drill made of concrete with a smell of coffee, a small Russian canvas made of ceramic with a smell of pine needles, a large Iranian ring made of glass with a smell of rose, an extra-large Afghan screwdriver made of concrete with a smell of rose, an extra-extra-large French camera made of concrete with a smell of citrus fruits, a large Portuguese bird made of plastic with a smell of pine needles, a medium Russian ruler made of plastic with a smell of lavender, an extra-extra-small Russian hammer made of glass with a smell of burning wood, an extra-large Afghan umbrella made of glass with a smell of lavender, an extra-extra-large British shoe made of steel with a smell of burning wood, an extra-large Japanese ruler made of ceramic with a smell of rose, an extra-extra-large French cup made of glass with a smell of vinegar, an extra-extra-large Spanish fork made of plastic with a smell of lavender, a small Spanish key made of steel with a smell of garlic, a medium British hat made of ceramic with a smell of coconut, an extra-large Japanese screwdriver made of plastic with a smell of popcorn, a small Mexican house made of glass with a smell of wet dog, a medium Afghan umbrella made of plastic with a smell of gasoline, a small British brush made of plastic with a smell of coconut, an extra-extra-large Mexican book made of ceramic with a smell of leather, an extra-small Canadian candle made of glass with a smell of freshly cut grass, an extra-large Afghan brush made of concrete with a smell of lavender, an extra-large Turkish umbrella made of glass with a smell of coconut, an extra-extra-small Mexican fork made of steel with a smell of coffee, an extra-extra-small Polish pencil made of ceramic with a smell of coconut, a medium Brazilian plier made of ceramic with a smell of lavender, an extra-extra-large Iranian umbrella made of plastic with a smell of citrus fruits, an extra-extra-small Canadian candle made of plastic with a smell of wet dog, an extra-small Portuguese bird made of concrete with a smell of baking bread, a large Turkish bicycle made of ceramic with a smell of coconut. The color of the items was respectively as follows: the first 1 was beige, the next 1 was blue, the next 4 were crimson, the next 3 were cyan, the next 2 were gold, the next 2 were gray, the next 1 was green, the next 2 were indigo, the next 1 was ivory, the next 3 were khaki, the next 3 were magenta, the next 1 was maroon, the next 1 was orange, the next 1 was pink, the next 5 were purple, the next 2 were teal, the next 1 was violet, the next 1 was yellow. Then, for any item of color pink in my collection, my mom gave me another one of the same item, but with a teal color, Turkish origin, extra-small size, wet dog smell, and made of plastic. Then, for any item with a smell of leather, freshly cut grass, or wet dog in my new collection, my cousin gifted me another one of that item with the same properties but with her favorite smell (different from the three smells above). After this, my new collection had: 2 item(s) with vinegar smell, 5 item(s) with pine needles smell, 2 item(s) with freshly cut grass smell, 2 item(s) with coffee smell, 7 item(s) with gasoline smell, 3 item(s) with rose smell, 2 item(s) with citrus fruits smell, 5 item(s) with lavender smell, 2 item(s) with burning wood smell, 1 item(s) with garlic smell, 5 item(s) with coconut smell, 1 item(s) with popcorn smell, 3 item(s) with wet dog smell, 1 item(s) with leather smell, 1 item(s) with baking bread smell. Then, my brother replaced any item of size extra-extra-large in my new collection with an exact copy but with one specific size (different from extra-extra-large) and another exact copy but with color green. After this, my new collection had: 16 extra-large item(s), 8 medium item(s), 5 small item(s), 3 large item(s), 5 extra-extra-small item(s), 9 extra-extra-large item(s), 5 extra-small item(s). Then, for any item made of concrete in my new collection, my sister changed their color to her favorite color and changed their smell to gasoline. After this, my new collection had: 1 beige item(s), 1 blue item(s), 4 crimson item(s), 2 cyan item(s), 1 gold item(s), 2 gray item(s), 9 green item(s), 2 indigo item(s), 1 ivory item(s), 3 khaki item(s), 4 magenta item(s), 1 maroon item(s), 2 orange item(s), 2 pink item(s), 4 purple item(s), 5 teal item(s), 6 violet item(s), 1 yellow item(s). Then, my fiance compared my new collection with my initial collection and if there was an item in my new collection with a (smell, color) pair that was different from the (smell, color) pair of all items in my initial collection, she added another copy of that to my new collection.  Then, I lost one of the extra-large items in my collection.  In my current collection, how many items have the following attributes: color is not gray, size is not extra-large, material is not concrete, smell is not lavender, and origin is not British? If the exact number cannot be computed, the answer must be \"unknown\".
</example_1>
<example_2>
I had a collection of 40 weird items that went through a few changes. Initially, I had an extra-small French banana made of glass with a smell of pine needles, a large Afghan table made of concrete with a smell of baking bread, a medium Mexican piano made of glass with a smell of wet dog, an extra-small Russian sofa made of glass with a smell of pine needles, an extra-extra-small British calculator made of glass with a smell of baking bread, an extra-extra-small Russian camera made of concrete with a smell of pine needles, a large Spanish vase made of plastic with a smell of freshly cut grass, a large Polish spoon made of steel with a smell of burning wood, a medium German calculator made of plastic with a smell of coffee, a medium British book made of plastic with a smell of rose, an extra-extra-large Afghan fork made of glass with a smell of chocolate, a small Turkish screwdriver made of steel with a smell of lavender, an extra-small Brazilian plier made of glass with a smell of wet dog, an extra-small Portuguese key made of glass with a smell of burning wood, a large Brazilian earring made of plastic with a smell of chocolate, an extra-extra-small Polish scissors made of ceramic with a smell of vinegar, a medium Mexican trash can made of concrete with a smell of leather, an extra-extra-small Polish wallet made of glass with a smell of vinegar, a medium British marker made of concrete with a smell of freshly cut grass, a large French shirt made of ceramic with a smell of gasoline, a medium Chinese fork made of concrete with a smell of burning wood, an extra-small Italian key made of steel with a smell of coconut, an extra-small Italian plate made of plastic with a smell of citrus fruits, an extra-large Brazilian key made of plastic with a smell of gasoline, a small Afghan flower pot made of concrete with a smell of citrus fruits, a large Japanese bottle made of plastic with a smell of leather, an extra-extra-large Polish wallet made of glass with a smell of popcorn, a large French sofa made of ceramic with a smell of citrus fruits, an extra-large Chinese calculator made of glass with a smell of gasoline, an extra-extra-small Japanese wallet made of ceramic with a smell of gasoline, an extra-small Mexican house made of concrete with a smell of freshly cut grass, a large German umbrella made of ceramic with a smell of coconut, a large Russian flower pot made of concrete with a smell of popcorn, an extra-extra-small Brazilian marker made of concrete with a smell of burning wood, a small British jar made of steel with a smell of popcorn, an extra-extra-small British box made of plastic with a smell of rose, an extra-extra-small Iranian watch made of steel with a smell of burning wood, a large Chinese chair made of ceramic with a smell of freshly cut grass, an extra-extra-large Russian pencil made of ceramic with a smell of lavender, a large French screwdriver made of ceramic with a smell of leather. The color of the items was respectively as follows: the first 1 was beige, the next 1 was blue, the next 3 were crimson, the next 1 was cyan, the next 3 were gold, the next 3 were gray, the next 2 were indigo, the next 1 was ivory, the next 1 was khaki, the next 2 were magenta, the next 4 were orange, the next 1 was pink, the next 4 were purple, the next 2 were red, the next 2 were silver, the next 2 were teal, the next 3 were turquoise, the next 1 was violet, the next 2 were white, the next 1 was yellow. Then, my brother replaced any item of size extra-small in my collection with an exact copy but with one specific size (different from extra-small) and another exact copy but with color gold. After this, my collection had: 13 medium item(s), 11 large item(s), 8 extra-extra-small item(s), 7 extra-small item(s), 3 extra-extra-large item(s), 3 small item(s), 2 extra-large item(s). Then, my uncle threw away any French item in my new collection, but gave me two of the same item, one Japanese and one Chinese (all other properties the same). Then, if I had an even number of an item type in my new collection (e.g. two knives, regardless of their size, color, origin, material, or smell), I threw all of them away. Then, my friend took any item with a smell of vinegar, coffee or popcorn in my new collection and threw it away. Then, my fiance compared my new collection with my initial collection and if there was an item in my new collection with a (smell, color) pair that was different from the (smell, color) pair of all items in my initial collection, she added another copy of that to my new collection.  Then, I lost one of the extra-extra-small items in my collection.  In my current collection, how many items have the following attributes: color is not yellow, size is not extra-extra-small, material is not ceramic, smell is not gasoline, and origin is not Afghan? If the exact number cannot be computed, the answer must be \"unknown\".
</example_2>
<example_3>
I had a collection of 33 weird items that went through a few changes. Initially, I had an extra-extra-small Italian table made of steel with a smell of citrus fruits, a small Portuguese earring made of ceramic with a smell of rose, a medium Portuguese bottle made of glass with a smell of garlic, an extra-extra-large Italian drill made of steel with a smell of freshly cut grass, a large Mexican chair made of steel with a smell of pine needles, an extra-large Russian plate made of ceramic with a smell of popcorn, a medium British guitar made of plastic with a smell of chocolate, a large Mexican cup made of ceramic with a smell of baking bread, an extra-large French hat made of concrete with a smell of vanilla, an extra-extra-small German book made of steel with a smell of garlic, an extra-extra-large Chinese hammer made of concrete with a smell of burning wood, an extra-extra-small French speaker made of glass with a smell of gasoline, an extra-extra-large Polish sofa made of ceramic with a smell of vanilla, a medium Brazilian bottle made of glass with a smell of garlic, a small French chair made of steel with a smell of garlic, a large British wallet made of ceramic with a smell of wet dog, an extra-small Brazilian candle made of steel with a smell of chocolate, an extra-extra-large German trash can made of ceramic with a smell of baking bread, an extra-large Afghan flower pot made of concrete with a smell of lavender, an extra-large Polish ring made of glass with a smell of coconut, an extra-large British pen made of ceramic with a smell of garlic, a large Afghan plate made of glass with a smell of citrus fruits, an extra-extra-large Iranian banana made of steel with a smell of burning wood, a small Turkish candle made of concrete with a smell of pine needles, an extra-extra-small Brazilian ruler made of plastic with a smell of rose, a medium British ruler made of ceramic with a smell of vinegar, an extra-small Turkish stapler made of concrete with a smell of popcorn, a small French hat made of concrete with a smell of baking bread, an extra-small Spanish flower pot made of glass with a smell of popcorn, an extra-small British pen made of steel with a smell of leather, an extra-extra-large Turkish chair made of steel with a smell of citrus fruits, an extra-extra-small Spanish banana made of ceramic with a smell of pine needles, a small Spanish wrench made of plastic with a smell of gasoline. The color of the items was respectively as follows: the first 1 was beige, the next 2 were black, the next 1 was brown, the next 3 were crimson, the next 1 was gold, the next 1 was gray, the next 2 were green, the next 2 were indigo, the next 4 were khaki, the next 1 was magenta, the next 2 were maroon, the next 1 was orange, the next 3 were pink, the next 3 were purple, the next 1 was red, the next 1 was silver, the next 2 were teal, the next 1 was turquoise, the next 1 was yellow. Then, for each item of size medium in my collection, my aunt took it but then gave me another one of the same item made of her favorite material and with a smell of coconut (all other properties the same). After this, my collection had: 9 item(s) made of steel, 12 item(s) made of ceramic, 6 item(s) made of concrete, 4 item(s) made of glass, 2 item(s) made of plastic. Then, my teacher took any item from my new collection whose color started with \"b\" or whose name had 6 characters and threw it away. Then, for any item made of plastic in my new collection, my sister changed their color to her favorite color and changed their smell to coffee. After this, my new collection had: 1 blue item(s), 2 crimson item(s), 1 gold item(s), 1 gray item(s), 1 green item(s), 2 indigo item(s), 1 khaki item(s), 1 magenta item(s), 2 maroon item(s), 1 orange item(s), 1 pink item(s), 2 purple item(s), 1 red item(s), 1 silver item(s), 2 teal item(s). Then, my uncle threw away any Mexican item in my new collection, but gave me two of the same item, one British and one Spanish (all other properties the same). Then, my fiance compared my new collection with my initial collection and if there was an item in my new collection with a (smell, color) pair that was different from the (smell, color) pair of all items in my initial collection, she added another copy of that to my new collection.  Then, I lost one of the large items in my collection.  In my current collection, how many items have the following attributes: color is not orange, size is not large, material is not steel, smell is not popcorn, and origin is not Polish? If the exact number cannot be computed, the answer must be \"unknown\".
</example_3>
<example_4>
I had a collection of 47 weird items that went through a few changes. Initially, I had a medium Turkish ring made of plastic with a smell of coffee, a small Japanese bed made of plastic with a smell of wet dog, an extra-extra-large Chinese umbrella made of plastic with a smell of leather, an extra-large Spanish hat made of ceramic with a smell of gasoline, a small French bird made of ceramic with a smell of burning wood, a medium Canadian guitar made of steel with a smell of leather, an extra-small Italian table made of plastic with a smell of baking bread, a medium Iranian pen made of concrete with a smell of chocolate, a medium Canadian necklace made of steel with a smell of rose, an extra-large Spanish plier made of plastic with a smell of gasoline, an extra-large Portuguese candle made of concrete with a smell of rose, an extra-extra-small Japanese ruler made of concrete with a smell of pine needles, a medium Chinese chair made of glass with a smell of burning wood, a small Spanish box made of glass with a smell of coconut, a small Mexican fork made of glass with a smell of garlic, a medium Chinese plier made of ceramic with a smell of chocolate, a large Chinese clock made of concrete with a smell of leather, a small Russian wrench made of plastic with a smell of rose, an extra-large French brush made of ceramic with a smell of citrus fruits, an extra-large Afghan shirt made of plastic with a smell of leather, a small Iranian plier made of ceramic with a smell of leather, an extra-small British bowl made of glass with a smell of coffee, a medium Polish pencil made of ceramic with a smell of wet dog, an extra-small Spanish clock made of plastic with a smell of citrus fruits, a large French calculator made of concrete with a smell of burning wood, an extra-small German sunglasses made of ceramic with a smell of pine needles, a large Iranian ring made of glass with a smell of pine needles, an extra-extra-small Mexican bottle made of ceramic with a smell of rose, an extra-extra-large Afghan bicycle made of concrete with a smell of citrus fruits, a large Canadian earring made of glass with a smell of pine needles, an extra-large British watch made of concrete with a smell of garlic, a large Mexican vase made of plastic with a smell of vinegar, a small Mexican canvas made of steel with a smell of chocolate, an extra-extra-small Polish pen made of ceramic with a smell of coconut, an extra-large Polish spoon made of steel with a smell of gasoline, a medium Italian pen made of plastic with a smell of coconut, an extra-large Russian key made of steel with a smell of pine needles, a small Afghan hammer made of glass with a smell of coffee, a medium Italian canvas made of ceramic with a smell of vinegar, an extra-large Brazilian marker made of ceramic with a smell of burning wood, a small British wallet made of glass with a smell of vanilla, an extra-large German box made of ceramic with a smell of leather, a large Polish watch made of plastic with a smell of garlic, an extra-extra-small French scissors made of plastic with a smell of wet dog, an extra-extra-small British clock made of steel with a smell of popcorn, an extra-extra-large Turkish book made of concrete with a smell of citrus fruits, an extra-small Japanese canvas made of plastic with a smell of popcorn. The color of the items was respectively as follows: the first 2 were beige, the next 2 were black, the next 1 was brown, the next 1 was crimson, the next 2 were cyan, the next 1 was gray, the next 1 was green, the next 2 were indigo, the next 3 were ivory, the next 3 were khaki, the next 2 were magenta, the next 3 were maroon, the next 1 was orange, the next 1 was pink, the next 2 were purple, the next 4 were red, the next 4 were teal, the next 3 were turquoise, the next 5 were violet, the next 2 were white, the next 2 were yellow. Then, for any item of color magenta in my collection, my mom gave me another one of the same item, but with a gray color, Canadian origin, small size, pine needles smell, and made of glass. Then, for any item with a smell of freshly cut grass, pine needles, or coconut in my new collection, my cousin gifted me another one of that item with the same properties but with her favorite smell (different from the three smells above). After this, my new collection had: 3 item(s) with coffee smell, 3 item(s) with wet dog smell, 6 item(s) with leather smell, 13 item(s) with gasoline smell, 4 item(s) with burning wood smell, 1 item(s) with baking bread smell, 3 item(s) with chocolate smell, 4 item(s) with rose smell, 7 item(s) with pine needles smell, 3 item(s) with coconut smell, 3 item(s) with garlic smell, 4 item(s) with citrus fruits smell, 2 item(s) with vinegar smell, 1 item(s) with vanilla smell, 2 item(s) with popcorn smell. Then, my teacher took any item from my new collection whose color started with \"b\" or whose name had 6 characters and threw it away. Then, my uncle threw away any Russian item in my new collection, but gave me two of the same item, one Afghan and one Polish (all other properties the same). Then, my fiance compared my new collection with my initial collection and if there was an item in my new collection with a (smell, color) pair that was different from the (smell, color) pair of all items in my initial collection, she added another copy of that to my new collection.  Then, I lost one of the large items in my collection.  In my current collection, how many items have the following attributes: color is not pink, size is not large, material is not plastic, smell is not wet dog, and origin is not Japanese? If the exact number cannot be computed, the answer must be \"unknown\".
</example_4>
<example_5>
I had a collection of 50 weird items that went through a few changes. Initially, I had an extra-extra-small Polish chair made of ceramic with a smell of garlic, an extra-extra-large Portuguese knife made of steel with a smell of vanilla, a large Iranian bowl made of glass with a smell of pine needles, a medium Brazilian bag made of glass with a smell of leather, a small Iranian drill made of concrete with a smell of vanilla, a medium British shirt made of plastic with a smell of leather, a medium Italian key made of plastic with a smell of popcorn, an extra-extra-large Brazilian wrench made of steel with a smell of gasoline, an extra-small Spanish necklace made of concrete with a smell of lavender, an extra-extra-small Chinese speaker made of steel with a smell of rose, an extra-extra-large Spanish drill made of steel with a smell of citrus fruits, a medium Russian screwdriver made of plastic with a smell of lavender, a medium Japanese chair made of glass with a smell of baking bread, an extra-small British box made of concrete with a smell of vinegar, a medium Afghan dog made of plastic with a smell of citrus fruits, an extra-large Brazilian guitar made of glass with a smell of citrus fruits, an extra-large Japanese banana made of concrete with a smell of garlic, an extra-small Japanese bird made of glass with a smell of coconut, an extra-large British bird made of ceramic with a smell of wet dog, a small Chinese bicycle made of plastic with a smell of vinegar, an extra-small French car made of concrete with a smell of coconut, an extra-extra-large German drill made of plastic with a smell of coconut, a small French shoe made of steel with a smell of citrus fruits, an extra-extra-small French earring made of concrete with a smell of vinegar, an extra-extra-small Canadian book made of concrete with a smell of garlic, an extra-small Chinese bicycle made of glass with a smell of pine needles, a large Italian apple made of glass with a smell of garlic, a medium British bicycle made of plastic with a smell of freshly cut grass, a large Italian pencil made of concrete with a smell of coffee, an extra-large Chinese hat made of ceramic with a smell of leather, a medium Turkish sunglasses made of ceramic with a smell of pine needles, a medium German fork made of concrete with a smell of gasoline, an extra-large Spanish knife made of steel with a smell of popcorn, an extra-small Italian ball made of steel with a smell of garlic, an extra-extra-small Italian trash can made of concrete with a smell of popcorn, a medium Afghan screwdriver made of plastic with a smell of baking bread, an extra-large Russian bowl made of plastic with a smell of popcorn, an extra-small Russian marker made of plastic with a smell of baking bread, an extra-small Portuguese bed made of ceramic with a smell of rose, a small Italian table made of ceramic with a smell of baking bread, an extra-small Portuguese plier made of plastic with a smell of popcorn, an extra-large German vase made of ceramic with a smell of chocolate, an extra-extra-small French canvas made of glass with a smell of wet dog, an extra-extra-small Mexican bowl made of steel with a smell of rose, an extra-small French umbrella made of plastic with a smell of freshly cut grass, an extra-extra-large British jar made of plastic with a smell of freshly cut grass, an extra-extra-small British speaker made of concrete with a smell of chocolate, a small Afghan table made of glass with a smell of freshly cut grass, a medium British ring made of ceramic with a smell of vanilla, an extra-extra-large Russian saw made of ceramic with a smell of burning wood. The color of the items was respectively as follows: the first 3 were beige, the next 2 were black, the next 3 were blue, the next 3 were brown, the next 2 were cyan, the next 1 was gold, the next 2 were green, the next 2 were indigo, the next 4 were ivory, the next 1 was magenta, the next 4 were maroon, the next 3 were orange, the next 4 were pink, the next 1 was purple, the next 1 was red, the next 3 were silver, the next 3 were turquoise, the next 1 was violet, the next 4 were white, the next 3 were yellow. Then, my uncle threw away any Portuguese item in my collection, but gave me two of the same item, one Iranian and one Turkish (all other properties the same). Then, my teacher took any item from my new collection whose color started with \"b\" or whose name had 6 characters and threw it away. Then, for any item of color yellow in my new collection, my mom gave me another one of the same item, but with a gold color, Iranian origin, extra-extra-large size, chocolate smell, and made of concrete. Then my dad threw away all item of a certain color from my new collection. After this, my new collection only had: 2 cyan item(s), 4 gold item(s), 1 indigo item(s), 4 ivory item(s), 1 magenta item(s), 4 maroon item(s), 2 orange item(s), 4 pink item(s), 1 purple item(s), 1 red item(s), 3 silver item(s), 4 turquoise item(s), 4 white item(s), 3 yellow item(s). Then, my fiance compared my new collection with my initial collection and if there was an item in my new collection with a (smell, color) pair that was different from the (smell, color) pair of all items in my initial collection, she added another copy of that to my new collection.  Then, I lost one of the large items in my collection.  In my current collection, how many items have the following attributes: either color is turquoise, or size is large, or material is glass, or smell is garlic, or origin is Chinese? If the exact number cannot be computed, the answer must be \"unknown\".
</example_5>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from internbootcamp.bootcamp.base import Basebootcamp
import json
import random
import re
from copy import deepcopy

class Bbehobjectpropertiesbootcamp(Basebootcamp):
    def __init__(self, min_items=3 ,max_items=50, max_steps=5):
        """
        初始化谜题训练场，配置生成参数
        
        参数:
            max_items: 初始物品数量上限（默认50）
            max_steps: 操作步骤数量（默认5）
        """
        self.min_items = min_items
        self.max_items = max_items
        self.max_steps = max_steps
        
        # 定义可能属性值域
        self.sizes = ["extra-extra-small", "extra-small", "small", "medium", 
                     "large", "extra-large", "extra-extra-large"]
        self.origins = ['Spanish', 'Turkish', 'French', 'Russian', 'British',
                       'Mexican', 'Afghan', 'Portuguese', 'Canadian', 'Japanese',
                       'Iranian', 'Italian', 'German', 'Chinese', 'Polish', 'Brazilian']
        self.materials = ["plastic", "steel", "glass", "ceramic", "concrete"]
        self.smells = ['vinegar', 'pine needles', 'freshly cut grass', 'coffee',
                      'rose', 'citrus fruits', 'lavender', 'burning wood', 'garlic',
                      'gasoline', 'coconut', 'popcorn', 'wet dog', 'leather',
                      'baking bread', 'chocolate', 'vanilla']
        self.names = ['calculator', 'hat', 'jar', 'plier', 'sunglasses', 'drill',
                     'canvas', 'ring', 'screwdriver', 'camera', 'bird', 'ruler',
                     'hammer', 'umbrella', 'shoe', 'cup', 'fork', 'key', 'house',
                     'brush', 'book', 'candle', 'bicycle', 'vase', 'clock', 'sofa',
                     'trash can', 'marker', 'bottle', 'banana']

    def case_generator(self):
        # 生成初始物品集合
        case = self._generate_initial_case()
        # 应用多层修改步骤
        case = self._apply_operations(case)
        # 生成最终问题条件
        case["question"] = self._generate_question(case["final_objects"])
        # 计算正确答案
        case["correct_answer"] = self._compute_correct_answer(case["final_objects"], case["question"])
        return case

    @staticmethod
    def prompt_func(case):
        # 构建问题提示文本
        prompt = ["I had a collection of {} weird items that went through a few changes. Initially:".format(len(case["initial_objects"]))]
        
        # 添加初始物品描述
        for i, obj in enumerate(case["initial_objects"], 1):
            prompt.append("{}. a {} {} {} made of {} with a smell of {}.".format(
                i, obj['size'], obj['origin'], obj['name'], obj['material'], obj['smell']))
        
        # 添加颜色分布描述
        color_desc = ["The color of the items was respectively as follows:"]
        count = 0
        for block in case["color_blocks"]:
            count += block["count"]
            color_desc.append("the {} {} were {}".format(
                "next" if count > block["count"] else "first",
                block["count"],
                block["color"]
            ))
        prompt.append(" ".join(color_desc).replace(" 1 were", " 1 was") + ".")
        
        # 添加操作步骤描述
        for step in case["steps"]:
            prompt.append(step["description"])
        
        # 添加最终问题
        conditions = []
        for attr, cond in case["question"].items():
            if cond["type"] == "negated":
                conditions.append("{} is not {}".format(attr, cond["value"]))
            elif cond["type"] == "or":
                conditions.append("either " + " or ".join(cond["values"]))
        prompt.append("In my current collection, how many items have the following attributes: {}? "
                      "If the exact number cannot be computed, the answer must be 'unknown'.\n"
                      "Please put your final answer within [answer][/answer] tags.".format(", ".join(conditions)))
        
        return "\n".join(prompt)

    @staticmethod
    def extract_output(output):
        # 使用非贪婪匹配查找所有答案块
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        # 提取最后一个答案块并清除前后空格
        return matches[-1].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 直接比较提取答案和预先计算的正确答案
        try:
            return int(solution) == int(identity["correct_answer"])
        except:
            try:
                return solution.lower() == identity["correct_answer"].lower()
            except:
                return False
    # 以下是辅助方法
    def _generate_initial_case(self):
        # 生成颜色分布
        n = random.randint(self.min_items, self.max_items)
        colors = ["beige", "blue", "crimson", "cyan", "gold", "gray", "green",
                 "indigo", "khaki", "magenta", "maroon", "orange", "pink", 
                 "purple", "teal", "violet", "white", "yellow"]
        
        color_blocks = []
        remaining = n
        while remaining > 0:
            color = random.choice(colors)
            count = random.randint(1, min(5, remaining))
            color_blocks.append({"color": color, "count": count})
            remaining -= count
        
        # 生成物品对象
        objects = []
        color_idx = 0
        current_count = 0
        for _ in range(n):
            if current_count >= color_blocks[color_idx]["count"]:
                color_idx += 1
                current_count = 0
            
            obj = {
                "size": random.choice(self.sizes),
                "origin": random.choice(self.origins),
                "material": random.choice(self.materials),
                "smell": random.choice(self.smells),
                "name": random.choice(self.names),
                "color": color_blocks[color_idx]["color"]
            }
            objects.append(obj)
            current_count += 1
        
        return {
            "initial_objects": objects,
            "color_blocks": color_blocks,
            "steps": [],
            "final_objects": deepcopy(objects),
            "correct_answer": None
        }

    def _apply_operations(self, case):
        # 生成随机操作步骤
        for _ in range(self.max_steps):
            case = self._apply_random_operation(case)
        return case

    def _apply_random_operation(self, case):
        # 随机选择一个操作类型并应用
        operation_type = random.choice(["add_copies", "replace_size", "modify_material"])
        
        if operation_type == "add_copies":
            return self._add_color_copies(case)
        elif operation_type == "replace_size":
            return self._replace_size(case)
        else:
            return self._modify_material(case)

    def _add_color_copies(self, case):
        # 添加指定颜色的副本
        target_color = random.choice(list({b["color"] for b in case["color_blocks"]}))
        new_color = random.choice([c for c in self.smells if c != target_color])
        
        new_objects = []
        for obj in case["final_objects"]:
            new_objects.append(obj)
            if obj["color"] == target_color:
                new_obj = deepcopy(obj)
                new_obj.update({
                    "color": new_color,
                    "origin": random.choice(self.origins),
                    "size": random.choice(self.sizes)
                })
                new_objects.append(new_obj)
        
        case["final_objects"] = new_objects
        
        description_tplts = []
        description_tplts.append("Then, my relative added copies of all {} items, changing their color to {}.")
        description_tplts.append("Then, copies of all {} items were added, changing their color to {}.")
        description_tplts.append("Next, all {} items were duplicated, altering their color to {}.")
        description_tplts.append("After that, copies of all {} items were created, modifying their color to {}.")
        description_tplts.append("Subsequently, all {} items were replicated, transforming their color to {}.")
        description_tplts.append("Following this, copies of all {} items were made, shifting their color to {}.")
        description_tplts.append("Then, all {} items were copied, updating their color to {}.")
        description_tplts.append("Next, the color of all {} items was changed to {} by creating copies.")
        description_tplts.append("After that, the color of all {} items was updated to {} through duplication.")
        description_tplts.append("Subsequently, all {} items were cloned, with their color adjusted to {}.")
        description_tplts.append("Following this, the color of all {} items was modified to {} by making copies.") 
        case["steps"].append({
            "type": "add_copies",
            "description": random.choice(description_tplts).format(
                target_color, new_color)
        })
        return case

    def _replace_size(self, case):
        # 替换某些对象的尺寸
        final_objects = case["final_objects"]
        
        # 确保至少有一个对象的尺寸被替换
        existing_sizes = {obj["size"] for obj in final_objects}
        if not existing_sizes:
            raise ValueError("No objects found to replace size.")
        
        # 随机选择一个目标尺寸
        target_size = random.choice(list(existing_sizes))
        
        # 确保新尺寸与目标尺寸不同
        available_sizes = [s for s in self.sizes if s != target_size]
        if not available_sizes:
            raise ValueError("No available sizes to replace with.")
        new_size = random.choice(available_sizes)
        
        # 更新对象尺寸
        updated_objects = []
        replaced = False  # 标记是否进行了替换
        for obj in final_objects:
            if obj["size"] == target_size:
                new_obj = deepcopy(obj)
                new_obj["size"] = new_size
                updated_objects.append(new_obj)
                replaced = True
            else:
                updated_objects.append(obj)
        
        # 如果没有替换任何对象，抛出异常
        if not replaced:
            raise ValueError(f"No objects found with size '{target_size}' to replace.")
        
        # 更新案例数据
        case["final_objects"] = updated_objects
        case["steps"].append({
            "type": "replace_size",
            "description": "Then, I replaced all {} items with {} ones.".format(target_size, new_size)
        })
        return case

    def _modify_material(self, case):
        # 修改某些对象的材质
        existing_materials = [obj["material"] for obj in case["final_objects"]]
        target_material = random.choice(existing_materials)
        new_material = random.choice([m for m in self.materials if m != target_material])
        
        updated_objects = []
        for obj in case["final_objects"]:
            if obj["material"] == target_material:
                new_obj = deepcopy(obj)
                new_obj["material"] = new_material
                updated_objects.append(new_obj)
            else:
                updated_objects.append(obj)
        
        case["final_objects"] = updated_objects
        case["steps"].append({
            "type": "modify_material",
            "description": "Then, I changed all {} items to be made of {}.".format(target_material, new_material)
        })
        return case

    def _generate_question(self, objects):
        # 生成随机问题条件
        attrs = ["color", "size", "material", "smell", "origin"]
        selected = random.sample(attrs, random.randint(2,len(attrs)))
        
        question_type = random.choice(["negated", "or"])
        conditions = {}
        for attr in selected:
            values = list(set(obj[attr] for obj in objects))
            target = random.choice(values)
            if question_type == "negated":
                conditions[attr] = {
                    "type": question_type,
                    "value": target
                }
            elif question_type == "or":
                conditions[attr] = {
                    "type": question_type,
                    "values": random.sample(values, random.randint(1, len(values)))
                }
        return conditions
    
    def _compute_correct_answer(self, objects, question):
        """
        根据问题条件计算正确答案
        """
        def matches_condition(obj, condition):
            attr, cond = condition
            if cond["type"] == "negated":
                return obj[attr] != cond["value"]
            elif cond["type"] == "or":
                return obj[attr] in cond["values"]
            return False
        
        # 筛选符合条件的对象
        valid_objects = []
        for obj in objects:
            if all(matches_condition(obj, (attr, cond)) for attr, cond in question.items()):
                valid_objects.append(obj)
        
        # 返回符合条件的对象数量
        return len(valid_objects)
    
if __name__ == "__main__":
    ## 单元测试
    bootcamp = Bbehobjectpropertiesbootcamp(min_items=10 ,max_items=12,max_steps=5)
    
    # 测试 case_generator 是否能生成有效的案例
    case = bootcamp.case_generator()
    print(json.dumps(case, indent=4))
    
    # 测试 prompt_func 是否能生成正确的提示文本
    prompt = bootcamp.prompt_func(case)
    print(prompt)
    
    # 测试 extract_output 是否能正确提取答案
    mock_output = "Some text [answer]42[/answer] more text."
    extracted_answer = bootcamp.extract_output(mock_output)
    assert extracted_answer == "42", f"Expected '42', got '{extracted_answer}'"
    
    # 测试 _replace_size 和 _modify_material 是否正常工作
    initial_sizes = {obj["size"] for obj in case["final_objects"]}
    case = bootcamp._replace_size(case)
    updated_sizes = {obj["size"] for obj in case["final_objects"]}
    assert updated_sizes != initial_sizes, "Sizes should have been modified by _replace_size"
    
    # 测试 _modify_material
    initial_materials = {obj["material"] for obj in case["final_objects"]}
    case = bootcamp._modify_material(case)
    updated_materials = {obj["material"] for obj in case["final_objects"]}
    assert updated_materials != initial_materials, "Materials should have been modified by _modify_material"
    
    # 测试 _verify_correction
    answer = case['correct_answer']
    model_output = f"""So, the correct_answer is like [answer]{answer}[/answer]."""
    print(model_output)
    assert bootcamp.verify_score(model_output, case), "Correction should be verified correctly"
    

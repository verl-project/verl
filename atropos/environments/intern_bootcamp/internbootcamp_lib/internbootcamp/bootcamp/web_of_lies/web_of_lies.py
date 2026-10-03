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
# BBEH Web of Lies

In this task, whether a specific person P1 tells the truth of lies is provided
as input. Then, for other people, it is specified what they say about the truth
value of some other person. This forms a chain-like structure that can be
started from P1 and continued to find whether each of the people tells the
truth or lies.

We used two different variants for this task. The first variant comes from the
*Web of Lies V2* from [LiveBench](https://arxiv.org/abs/2406.19314). In this
variant, complexity has been added to the task by specifying where each person
is, and then having sentences such as *The person at the cafe says the person
at the zoo lies*. The second version is created by us. In this version, we add
cyclic cases whose truth value remains unknown, but one can still infer
something about them and continue the chain. For example, consider a cyclic case
such as *Person1 says Person2 tells the truth. Person2 says Person1 tells the
truth.* In this case, we cannot determine whether Person1 or Person2 tell the
truth or lie (so their truthfulness remains unknown). However, if we have
another sentence *Person3 says either both Person1 and Person2 lie or both tell
the truth*, we can determine that Person3 tells the truth. In both variants of
the problems, we ask about the truthfulness of three of the people in the chain,
so the random chance performance for the LiveBench subset is 1/8 since the
truthfulness of each of the three people can be either yes or no, and 1/27 for
our new set given that the values can also be unknown.

[/Description]
<example_1>
In this question, assume each person either always tells the truth or always lies. Mindy says Olga lies. Mustafa says Odell lies. Melanie says Melinda lies. Yoland says only one of Shalonda, Osbert and Mindy lies. Odin says Ozias tells the truth. Margaret says Miranda lies. Jaymie says Conception lies. Rashida says Yoland lies. Mona says McKenzie lies. Delbert says only one of Jim, Odette and Melanie lies. Kristian says exactly one of Audrie, Mona and McKenzie tells the truth.Orhan says Mark tells the truth. Shalonda says either all three of Jaymie, Orhan and Mark lie, or two of them tell the truth. Bernita tells the truth. Jim says Inga, Monica and Margaret all tell the truth. Ozias says Odin tells the truth. Maurice says Oliver lies. Odell says Mustafa lies. Monica says Miranda tells the truth. Audrie says Bernita, Maya and Maurice all tell the truth. Osbert says Olga tells the truth. Onyx says Hritik tells the truth. Dallas says Antwan lies. Vina says Dallas lies. Helene says exactly one of Delbert, Milo and Mariana tells the truth.Milo says Mariana lies. McKenzie says Mona lies. Inga says exactly one of Vina, Odell and Mustafa tells the truth.Conception says only one of Kristian, Onyx and Marina lies. Mark says Orhan tells the truth. Antwan says either exactly one of Rashida, Odin and Ozias tells the truth, or all three of them. Mariana says Milo lies. Marina says Hritik lies. Maya says Oliver tells the truth. Odette says Melinda tells the truth. Do Vina, Melanie, and Helene tell the truth? Your answer should be a list of three words separated by commas. Each of them must be either yes (if the person tells the truth), no (if the person lies), or unknown (if the person's truth cannot be determined). For example, <unknown, no, yes> or <yes, yes, yes>.
</example_1>
<example_2>
In this question, assume each person either always tells the truth or always lies. Melody says Oswald tells the truth. Phoebe says Shaunda, Ophelia and Osvaldo all tell the truth. Orhan says Minerva lies. Shalonda says Gwenn tells the truth. Miller says Margaret tells the truth. Delfina tells the truth. Ocean says Octavius lies. Osmond says Octavia lies. Amberly says Kristian tells the truth. Minerva says Orhan lies. Oakleigh says Olympia lies. Gwenn says either exactly two of Jerry, Margaret and Miller tell the truth or none of them. Olympia says Oakleigh lies. Orin says Mark lies. Owyn says Mustafa lies. Yoland says either all three of Fidel, Melody and Oswald lie, or two of them tell the truth. Mariana says Nicole lies. Tamika says exactly two of Shalonda, Nicole and Mariana tells the truth. Oriana says Octavia tells the truth. Alejandro says Lorine tells the truth. Antwan says only one of Fletcher, Obadiah and Merlin lies. Merlin says Misty lies. Jaymie says Alejandro, Oriana and Osmond all tell the truth. Fletcher says exactly two of Jaymie, Olympia and Oakleigh tells the truth. Ophelia says Melvin tells the truth. Dallas says Amberly lies. Fidel says exactly one of Delfina, Orin and Mark tell the truth. Shaunda says Crista lies. Margaret says Miller tells the truth. Ryan says exactly two of Delbert, Mustafa and Owyn tell the truth. Delbert says Phoebe lies. Obadiah says Misty tells the truth. Mustafa says Owyn lies. Kristian says exactly two of Tamika, Minerva and Orhan tells the truth. Nicole says Mariana lies. Lorine says exactly two of Yoland, Octavius and Ocean tell the truth. Octavius says Ocean lies. Jerry says Ryan tells the truth. Oswald says Melody tells the truth. Crista says Antwan tells the truth. Osvaldo says Melvin lies. Mark says Orin lies. Do Kristian, Orin, and Dallas tell the truth? Your answer should be a list of three words separated by commas. Each of them must be either yes (if the person tells the truth), no (if the person lies), or unknown (if the person's truth cannot be determined). For example, <unknown, no, yes> or <yes, yes, yes>.
</example_2>
<example_3>
In this question, assume each person either always tells the truth or always lies. Sima says exactly one of Fidel, Murphy and Otis tells the truth.Molly says Milo tells the truth. Murphy says Otis lies. Madison says Oriana tells the truth. Jim says exactly one of Inga, Olympia and Miriam tells the truth.Oona says Micah lies. Octavia says Mark lies. Otis says Murphy lies. Jaymie says either all three of Jamey, Madison and Oriana tell the truth or only one of them. Phoebe says only one of Alejandro, Bahar and Osborn tells the truth. Marsha says Muhammad lies. Teressa says only one of Sal, Maurice and Oscar lies. Audrie tells the truth. Olympia says Miriam lies. Oakleigh says Mallory lies. Alejandro says Delbert, John and Octavia all tell the truth. Inga says Kandi tells the truth. John says Mark tells the truth. Margaret says Kate lies. Rashida says Teressa lies. Matteo says Marvin lies. Marlon says Oswald lies. Lorine says Rashida lies. Jamey says only one of Tamika, Maya and Melvin tells the truth. Shaunda says only one of Jim, Odin and Matteo lies. Octavian says Kate tells the truth. Ofelia says Micah tells the truth. Orin says Misty lies. Maximilian says Oakley lies. Maya says Ottoline tells the truth. Yoland says exactly two of Ka, Oakley and Maximilian tells the truth. Opal says Osian tells the truth. Osian says Opal tells the truth. Elanor says exactly two of Conception, Oakleigh and Mallory tell the truth. Misty says Orin lies. Crista says either exactly one of Yoland, Natasha and Mina tells the truth, or all three of them. Natasha says Mina tells the truth. Conception says Audrie tells the truth. Maurice says Maxim tells the truth. Oswin says Muhammad tells the truth. Sal says only one of Elanor, Ofelia and Oona lies. Delbert says only one of Jerry, Oswin and Marsha lies. Ka says Phoebe lies. Alexis says either all three of Jaymie, Osian and Opal tell the truth or only one of them. Fidel says exactly two of Crista, Misty and Orin tells the truth. Oakley says Maximilian lies. Bahar says Maverick tells the truth. Mallory says Oakleigh lies. Delfina says Lorine, Octavian and Margaret all tell the truth. Melvin says Ottoline lies. Kandi says either exactly two of Alexis, Milo and Molly tell the truth or none of them. Tamika says Delfina tells the truth. Osborn says Maverick lies. Miriam says Olympia lies. Jerry says exactly two of Shaunda, Oswald and Marlon tell the truth. Milo says Molly tells the truth. Oscar says Maxim lies. Oswald says Marlon lies. Odin says Marvin tells the truth. Mina says Natasha tells the truth. Oriana says Madison tells the truth. Do Jaymie, Alexis, and Sima tell the truth? Your answer should be a list of three words separated by commas. Each of them must be either yes (if the person tells the truth), no (if the person lies), or unknown (if the person's truth cannot be determined). For example, <unknown, no, yes> or <yes, yes, yes>.
</example_3>
<example_4>
In this question, assume each person either always tells the truth or always lies. Alexis says only one of Delfina, Marshall and Olivia lies. Crista says exactly two of Kandi, Matthew and Maverick tells the truth. Jim says Shaunda, Miller and Misty all tell the truth. Ozias says Oleg tells the truth. Phoebe says Dallas lies. Opal says Odell tells the truth. Olivia says Marlon lies. Rashida tells the truth. Matteo says Ovid tells the truth. Maverick says Matthew lies. Odell says Opal tells the truth. Oden says Mario tells the truth. Antwan says exactly one of Rashida, Otis and Melissa tell the truth. Otis says Melissa lies. Misty says Bahar lies. Mike says Milton tells the truth. Milton says Mike tells the truth. Delfina says either exactly one of Teressa, Oleg and Ozias tells the truth, or all three of them. Kandi says Ka tells the truth. Oleg says Ozias tells the truth. Orion says Madison tells the truth. Vina says either exactly two of Leda, Mario and Oden tell the truth or none of them. Teressa says only one of Antwan, Matteo and Osric lies. Marshall says Marlon tells the truth. Leda says either all three of Elanor, Opal and Odell tell the truth or only one of them. Mario says Oden tells the truth. Shaunda says Alexis lies. Madison says Orion tells the truth. Melissa says Otis lies. Matthew says Maverick lies. Miller says Bahar tells the truth. Osric says Ovid lies. Dallas says Vina lies. Ka says either exactly one of Phoebe, Madison and Orion tells the truth, or all three of them. Elanor says either all three of Jim, Mike and Milton lie, or two of them tell the truth. Do Phoebe, Ozias, and Crista tell the truth? Your answer should be a list of three words separated by commas. Each of them must be either yes (if the person tells the truth), no (if the person lies), or unknown (if the person's truth cannot be determined). For example, <unknown, no, yes> or <yes, yes, yes>.
</example_4>
<example_5>
In this question, assume each person either always tells the truth or always lies. Beatriz is at the vineyard. The person at the city hall says the person at the school lies. The person at the ice skating rink says the person at the school tells the truth. Quan is at the ice skating rink. The person at the amusement park says the person at the school tells the truth. Olivia is at the restaurant. The person at the vineyard says the person at the botanical garden tells the truth. The person at the vineyard says the person at the campground tells the truth. Charlie is at the campground. Soren is at the school. Grace is at the barbershop. The person at the school says the person at the vineyard tells the truth. The person at the barbershop says the person at the campground tells the truth. Mateo is at the amusement park. Tala is at the botanical garden. The person at the campground tells the truth. The person at the restaurant tells the truth. Devika is at the city hall. The person at the amusement park thinks their neighbor is telling the truth. The person at the amusement park thinks their friend is lying. Does the person at the amusement park tell the truth? Does the person at the school tell the truth? Does the person at the vineyard tell the truth? Your answer should be a list of three words, yes or no (for example, yes, no, yes).
</example_5>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from internbootcamp.bootcamp.base import Basebootcamp
import random
import re
from typing import Dict, List, Optional, Tuple, Set
from faker import Faker


# 状态推导引擎
class LogicSolver:
    def __init__(self, people: List[str], initial: Tuple[str, bool]):
        self.people = set(people)
        self.states = {p: None for p in people}
        self.dependencies = {p: [] for p in people}
        self.states[initial[0]] = initial[1]
        self.init_people = initial[0]

    def add_statement(self, stmt: Dict):
        if stmt["type"] == "single":
            # 经典依赖关系
            speaker = stmt["speaker"]
            target = stmt["target"]
            assertion = stmt["assertion"]
            self.dependencies[speaker].append((target, assertion))
        else:
            # 逻辑关系预处理
            speaker = stmt["speaker"]
            group = stmt["group"]
            expected = stmt["assertion"]
            logic_type = stmt["type"]
            self.dependencies[speaker].append((group, logic_type, expected))

    def _eval_logic(
        self, group: List[str], logic: str, assertion: bool
    ) -> Optional[bool]:
        statuses = [self.states[p] for p in group]
        # if any(s is None for s in statuses):
        #     return None  # 存在未知状态
        have_none = any(s is None for s in statuses)
        truths = sum([s == assertion for s in statuses])
        no_cnt = sum([s == (not assertion) for s in statuses])
        group_size = len(group)
        if logic == "exact_one":
            if truths > 1 or group_size - no_cnt < 1:
                return False
            else:
                if have_none:
                    return None
                else:
                    return truths == 1
        elif logic == "exact_two":
            if truths > 2 or group_size - no_cnt < 2:
                return False
            else:
                if have_none:
                    return None
                else:
                    return truths == 2
        elif logic == "all_true":
            if no_cnt > 0:
                return False
            else:
                if have_none:
                    return None
                else:
                    truths == group_size
        elif logic == "all_false":
            if truths > 0:
                return False
            else:
                if have_none:
                    return None
                else:
                    truths == 0
        return None

    def _apply_one_logic(self, group, logic, expected, speaker_state):
        statuses = [self.states[p] for p in group]
        group_size = len(group)
        exact_expected = [s == expected for s in statuses]
        cnt = sum(exact_expected)
        exact_not_expected = [s == (not expected) for s in statuses]
        no_cnt = sum(exact_not_expected)
        none_list = [p for p in group if self.states[p] is None]
        if logic == "exact_one":
            if speaker_state:
                if cnt == 1:
                    for p in none_list:
                        self.states[p] = not expected
                elif group_size - no_cnt == 1:
                    for p in none_list:
                        self.states[p] = expected
            else:
                if cnt == 1 and len(none_list) == 1:
                    for p in none_list:
                        self.states[p] = expected
                elif cnt == 0 and len(none_list) == 1:
                    for p in none_list:
                        self.states[p] = not expected
        elif logic == "exact_two":
            if speaker_state:
                if cnt == 2:
                    for p in none_list:
                        self.states[p] = not expected
                elif group_size - no_cnt == 2:
                    for p in none_list:
                        self.states[p] = expected
            else:
                if cnt == 2 and len(none_list) == 1:
                    for p in none_list:
                        self.states[p] = expected
                elif cnt == 1 and len(none_list) == 1:
                    for p in none_list:
                        self.states[p] = not expected
        elif logic == "all_true":
            if speaker_state:
                for p in group:
                    if self.states[p] is None:
                        self.states[p] = expected
                    elif self.states[p] != expected:
                        raise ValueError
            else:
                if group_size - cnt == 1 and len(none_list) == 1:
                    for p in none_list:
                        self.states[p] = not expected
        elif logic == "all_false":
            if speaker_state:
                for p in group:
                    if self.states[p] is None:
                        self.states[p] = not expected
                    elif self.states[p] == expected:
                        raise ValueError
            else:
                if group_size - no_cnt == 1 and len(none_list) == 1:
                    for p in none_list:
                        self.states[p] = expected

    # if we have determined the speaker's correctness, then apply all of the depencies
    def _apply_logics(self, speaker):
        pre_state = copy.deepcopy(self.states)
        specker_state = self.states[speaker]
        for dep in self.dependencies[speaker]:
            if len(dep) == 3:
                group, logic, expected = dep
                self._apply_one_logic(group, logic, expected, specker_state)

        return pre_state != self.states

    def _reverse_apply_one_logic(self, group, logic, expected, speaker):
        statuses = [self.states[p] for p in group]
        have_none = any(s is None for s in statuses)
        group_size = len(group)
        exact_expected = [s == expected for s in statuses]
        cnt = sum(exact_expected)
        exact_not_expected = [s == (not expected) for s in statuses]
        no_cnt = sum(exact_not_expected)
        none_list = [p for p in group if self.states[p] is None]

        if logic == "exact_one":
            if not have_none and cnt == 1:
                self.states[speaker] = True
            elif cnt > 1:
                self.states[speaker] = False
        elif logic == "exact_two":
            if not have_none and cnt == 2:
                self.states[speaker] = True
            elif cnt > 2:
                self.states[speaker] = False
        elif logic == "all_true":
            if cnt == group_size:
                self.states[speaker] = True
            elif no_cnt > 0:
                self.states[speaker] = False
        elif logic == "all_false":
            if no_cnt == group_size:
                self.states[speaker] = True
            elif cnt > 0:
                self.states[speaker] = False

    def _reverse_apply_logics(self, speaker):
        assert (
            self.states[speaker] is None
        ), "In reverse apply, the state of speaker should be None"
        pre_state = copy.deepcopy(self.states)
        for dep in self.dependencies[speaker]:
            if len(dep) == 3:
                group, logic, expected = dep
                self._reverse_apply_one_logic(group, logic, expected, speaker)
        return pre_state != self.states

    def solve(self) -> Dict[str, Optional[bool]]:
        changed = True
        # 先apply initial
        # self._apply_logics(self.init_people)
        while changed:
            changed = False
            for speaker in self.people:
                if self.states[speaker] is None:
                    # try to reverse applying
                    reverse_changed = self._reverse_apply_logics(speaker)
                    if reverse_changed:
                        changed = reverse_changed
                    else:
                        continue
                apply_changed = self._apply_logics(speaker)
                if apply_changed:
                    changed = True
                for dep in self.dependencies[speaker]:
                    if len(dep) == 2:  # 简单陈述
                        target, assertion = dep
                        expected = assertion if self.states[speaker] else not assertion
                        if self.states[target] is None:
                            self.states[target] = expected
                            changed = True

                        elif self.states[target] != expected:
                            raise ValueError("Contradiction detected")
                    else:  # 逻辑陈述
                        group, logic, expected = dep
                        actual = self._eval_logic(group, logic, expected)
                        if actual is None:
                            continue  # 存在未知状态
                        if actual != self.states[speaker]:
                            raise ValueError("Contradiction in logic statement")

        return self.states


class BbehWebOfLiesbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_num_people = params.get("max_num_people", 12)
        self.min_num_people = params.get("min_num_people", 8)
        self.fake = Faker()

    def case_generator(self) -> Dict:
        num_people = random.randint(self.min_num_people, self.max_num_people)
        initial_known = random.choice([True, False])
        names = [self.fake.name() for _ in range(num_people)]
        initial_person = random.choice(names)

        logic_types = ["single", "exact_one", "exact_two", "all_true", "all_false"]

        while True:
            case = self._one_case(initial_person, initial_known, names, logic_types)
            res = self.call_prover(case)
            if res:
                case["solution"] = res[0]
                state = res[1]
                not_none_cnt = sum([state[n] is not None for n in state.keys()])
                if not_none_cnt / num_people > 0.5:
                    break

        random.shuffle(case["statements"])
        return case

    def call_prover(self, identity):
        try:
            solver = LogicSolver(
                identity["people"],
                (identity["initial_person"], identity["initial_state"]),
            )
            for stmt in identity["statements"]:
                solver.add_statement(stmt)
            states = solver.solve()
        except ValueError:
            return None  # 存在矛盾，生成无效案例
        # 生成预期答案
        expected = []
        for q in identity["questions"]:
            status = states.get(q)
            if status is None:
                expected.append("unknown")
            else:
                expected.append("yes" if status else "no")

        # return solution == expected
        return expected, states

    def _one_case(self, initial_person, initial_known, names, logic_types):
        num_people = len(names)
        statements = []
        # 生成初始确定链
        states = {name: None for name in names}
        states[initial_person] = initial_known
        processed = set([initial_person])
        queue = [initial_person]

        while queue:
            current = queue.pop(0)
            available = [n for n in names if n not in processed]
            for _ in range(random.randint(1, 2)):  # 生成1-2个子节点
                if not available:
                    break
                target = random.choice(available)
                available.remove(target)
                processed.add(target)

                # 生成简单陈述或逻辑陈述
                stmt_type = random.choices(
                    logic_types, weights=[0.2, 0.2, 0.2, 0.2, 0.2], k=1
                )[0]

                if stmt_type == "single":
                    assertion = random.choice([True, False])
                    statements.append(
                        {
                            "type": "single",
                            "speaker": current,
                            "target": target,
                            "assertion": assertion,
                        }
                    )
                    # 推导确定状态
                    states[target] = assertion if states[current] else not assertion
                else:
                    # 生成逻辑陈述需要至少3人
                    others = [n for n in names if n not in (current, target)]
                    if len(others) < 2:
                        continue
                    group = random.sample(others, 2) + [target]
                    statements.append(
                        {
                            "type": stmt_type,
                            "speaker": current,
                            "group": group,
                            "assertion": random.choice([True, False]),
                        }
                    )
                queue.append(target)

        # 添加复杂逻辑陈述
        for _ in range(int(num_people * 1.5)):
            speaker = random.choice(names)
            # if speaker == initial_person:
            #     stmt_type = random.choice(logic_types[1:3])
            # else:
            stmt_type = random.choice(logic_types[1:])
            group_size = 3 if stmt_type in ["exact_one", "exact_two"] else 2
            candidates = [n for n in names if n != speaker]
            if len(candidates) < group_size:
                continue
            group = random.sample(candidates, group_size)

            statements.append(
                {
                    "type": stmt_type,
                    "speaker": speaker,
                    "group": group,
                    "assertion": random.choice([True, False]),
                }
            )

        # 生成问题目标
        questions = random.sample([n for n in names if n != initial_person], 3)

        return {
            "people": names,
            "statements": statements,
            "questions": questions,
            "initial_person": initial_person,
            "initial_state": initial_known,
        }

    @staticmethod
    def prompt_func(question_case: Dict) -> str:
        rules = [
            "In this question, assume each person either always tells the truth or always lies."
        ]

        statements = [
            f"What you know exactly is that {question_case['initial_person']} "
            f"{'tells the truth' if question_case['initial_state'] else 'lies'}."
        ]

        for stmt in question_case["statements"]:
            if stmt["type"] == "single":
                verb = "tells the truth" if stmt["assertion"] else "lies"
                s = f"{stmt['speaker']} says {stmt['target']} {verb}."
            else:
                logic_map = {
                    "exact_one": f"exactly one of {', '.join(stmt['group'])}",
                    "exact_two": f"exactly two of {', '.join(stmt['group'])}",
                    "all_true": f"all of {', '.join(stmt['group'])}",
                    "all_false": f"none of {', '.join(stmt['group'])}",
                }
                truth_status = "tell the truth" if stmt["assertion"] else "lie"
                s = f"{stmt['speaker']} says {logic_map[stmt['type']]} {truth_status}."
            statements.append(s)

        return " ".join(
            [
                "In this question, assume each person either always tells the truth or always lies.",
                *statements,
                f"Do {', '.join(question_case['questions'])} tell the truth?",
                "Your answer should be a list of three words separated by commas. Each of them must be either yes (if the person tells the truth), no (if the person lies), or unknown (if the person's truth cannot be determined).",
                "Please reason step by step and put your final answer within [answer] [/answer] tags.",
                "For example:\n[answer]yes, unknown, no[/answer]\n",
                "Now generate your solution.",
            ]
        )

    @staticmethod
    def extract_output(output: str) -> Optional[List[str]]:
        pattern = r"""
            \[answer\]            # 开始标签
            \s*                   # 允许空白
            (?:<)?                # 可选尖括号
            \s*                   # 允许空白
            ((?:yes|no|unknown)   # 有效值
            (?:\s*,\s*(?:yes|no|unknown)){2})  # 必须包含三个值
            \s*                   # 允许空白
            (?:>)?                # 可选尖括号
            \s*                   # 允许空白
            \[/answer\]           # 结束标签
        """
        matches = re.findall(pattern, output, flags=re.X | re.I)
        if not matches:
            return None

        last_match = matches[-1].strip().lower()
        parts = [p.strip() for p in last_match.split(",")]
        if len(parts) != 3 or any(p not in {"yes", "no", "unknown"} for p in parts):
            return None
        return parts

    @classmethod
    def _verify_correction(cls, solution: List[str], identity: Dict) -> bool:
        return solution == identity["solution"]


import copy

if __name__ == "__main__":
    bootcamp = BbehWebOfLiesbootcamp()
    case = bootcamp.case_generator()
    print(bootcamp.prompt_func(case))
    example_answer = "some text [answer]yes, no, yes[/answer] some text"
    print(f"answer: {example_answer}")
    score = bootcamp.verify_score(example_answer, case)
    print(f"score: {score}")
    print(f"ground_truth: {case['solution']}")

import copy
import re
import random
from typing import Dict, List, Set, Tuple
from internbootcamp.bootcamp.base import Basebootcamp


Entities = [
    # Set1 animals
    'cat', 'dog', 'pig', 'parrot', 'eagle', 'squirrel', 'penguin', 'lion', 'tiger', 'donkey',
    'leopard', 'cheetah', 'grizzly bear', 'polar bear', 'sun bear', 'panda bear', 'black bear',
    'turtle', 'crocodile', 'elephant', 'panther', 'cow', 'rabbit', 'hare', 'buffalo', 'baboon',
    'sheep', 'whale', 'jellyfish', 'carp', 'goldfish', 'viperfish', 'starfish', 'catfish',
    'oscar', 'zander', 'sea bass', 'swordfish', 'salmon', 'halibut', 'blobfish', 'doctorfish',
    'tilapia', 'kangaroo', 'octopus', 'phoenix', 'aardvark', 'amberjack', 'eel', 'hummingbird',
    'canary', 'hippopotamus', 'snail', 'caterpillar', 'mosquito', 'bat', 'ferret', 'gecko',
    'kudu', 'moose', 'cockroach', 'cricket', 'grasshopper', 'meerkat', 'spider', 'lobster',
    'squid', 'puffin', 'raven', 'kiwi', 'koala', 'wolverine',
    # Set2 animals
    'akita', 'bear', 'camel', 'coyote', 'snake', 'monkey', 'leopard', 'fish', 'ostrich', 'pigeon',
    'dolphin', 'frog', 'goat', 'goose', 'wolf', 'gorilla', 'beaver', 'lizard', 'flamingo', 'swan',
    'elk', 'duck', 'reindeer', 'bison', 'shark', 'mouse', 'owl', 'llama', 'cobra', 'zebra',
    'otter', 'crab', 'peafowl', 'rhino', 'dinosaur', 'dove', 'badger', 'chinchilla', 'cougar',
    'crow', 'seal', 'worm', 'ant', 'bee', 'butterfly', 'dragonfly', 'dragon', 'gadwall', 'mule',
    'liger', 'german shepherd', 'bulldog', 'husky', 'poodle', 'chihuahua', 'dachshund', 'basenji',
    'dalmatian', 'mermaid', 'seahorse', 'fangtooth', 'dugong', 'walrus', 'vampire', 'stork',
    'swallow', 'songbird', 'woodpecker', 'starling', 'mannikin', 'pelikan', 'beetle', 'finch'
]

# List of predicates/relations between entities
Predicates = [
    # Set1 predicates
    'owe money to',
    'give a magnifier to', 
    'learn the basics of resource management from',
    'know the defensive plans of',
    'show all her cards to',
    'prepare armor for',
    'sing a victory song for',
    'need support from',
    'respect',
    'raise a peace flag for',
    'become',
    'an enemy of',
    'roll the dice for',
    'hold the same number of points as',
    'offer a job to',
    'wink at',
    'steal five points from',
    'knock down the fortress of',
    'burn the warehouse of',
    'eat the food of',
    'attack the green fields whose owner is',
    'proceed to the spot that is right after the spot of',
    'remove one of the pieces of',
    # Set2 predicates
    'tear down the castle that belongs to',
    'bring an oil tank for',
    'reveal a secret to',
    'enjoy the company of',
    'neglect',
    'want to see',
    'swear to',
    'refuse to help',
    'manage to convince',
    'call',
    'stop the victory of',
    'dance with',
    'shout at',
    'smile at',
    'pay money to',
    'unite with',
    'hug',
    'destroy the wall constructed by',
    'create one castle for',
    'disarm',
    'acquire a photograph of',
    'borrow one of the weapons of',
    'fall on a square of',
    'suspect the truthfulness of',
    'invest in the company whose owner is',
    'leave the houses occupied by',
    'hide the cards that she has from',
    'swim in the pool next to the house of',
    'negotiate a deal with',
    'trade one of its pieces with',
    'build a power plant near the green fields of',
    'take over the emperor of',
    'capture the king of',
    'surrender to'
]


class Predicate:
    def __init__(self, name: str, negated: bool = False):
        self.name = name
        self.negated = negated

    def __str__(self):
        if self.negated:
            return f"does not {self.name}"
        return self.name
    
    def negate(self):
        return Predicate(self.name, not self.negated)
        
    def __eq__(self, other):
        return self.name == other.name and self.negated == other.negated
    
    def __hash__(self):
        return hash((self.name, self.negated))
    
    def __repr__(self):
        return f"Predicate(name={self.name}, negated={self.negated})"


class Statement:
    def __init__(self, subject, predicate, object):
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def __str__(self):
        return f"{self.subject} {self.predicate} {self.object}"
        
    def negate(self):
        """
        返回当前陈述的否定形式。
        
        返回:
            Statement: 一个新的Statement对象，表示当前陈述的否定
        """
        # 创建一个新的否定陈述
        return Statement(self.subject, self.predicate.negate(), self.object)

    def __eq__(self, other):
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object
    
    def __hash__(self):
        return hash((self.subject, self.predicate, self.object, self.negated))
    
    def __repr__(self):
        return f"Statement(subject={repr(self.subject)}, predicate={repr(self.predicate)}, object={repr(self.object)})"
    

class Rule:
    def __init__(self, type_id: int, conditions: List[Statement], conclusion: Statement, rule_id: int=-1):
        self.type_id = type_id
        self.conditions = conditions
        self.conclusion = conclusion
        self.rule_id = rule_id

    def __str__(self):
        if self.type_id == 0:
            assert len(self.conditions) == 1, repr(self)
            _, p1, e1 = self.conditions[0].subject, self.conditions[0].predicate, self.conditions[0].object
            _, p2, e2 = self.conclusion.subject, self.conclusion.predicate, self.conclusion.object
            return f"If any animal {p1} {e1}, then it {p2} {e2}"
        elif self.type_id == 1:
            assert len(self.conditions) == 2, repr(self)
            _, p1, e1 = self.conditions[0].subject, self.conditions[0].predicate, self.conditions[0].object
            _, p2, e2 = self.conditions[1].subject, self.conditions[1].predicate, self.conditions[1].object
            _, p3, e3 = self.conclusion.subject, self.conclusion.predicate, self.conclusion.object
            return f"If any animal {p1} {e1} and {p2} {e2}, then it {p3} {e3}"
        elif self.type_id == 2:
            assert len(self.conditions) == 1, repr(self)
            e1, p1, e2 = self.conditions[0].subject, self.conditions[0].predicate, self.conditions[0].object
            e2, p2, e3 = self.conclusion.subject, self.conclusion.predicate, self.conclusion.object
            return f"If {e1} {p1} {e2}, then {e2} {p2} {e3}"
        elif self.type_id == 3:
            assert len(self.conditions) == 2, repr(self)
            e1, p1, e2 = self.conditions[0].subject, self.conditions[0].predicate, self.conditions[0].object
            e3, p2, e2_again = self.conditions[1].subject, self.conditions[1].predicate, self.conditions[1].object
            e2, p3, e4 = self.conclusion.subject, self.conclusion.predicate, self.conclusion.object
            return f"If {e1} {p1} {e2} and {e3} {p2} {e2}, then {e2} {p3} {e4}"
        elif self.type_id == 4:
            assert len(self.conditions) == 1, repr(self)
            _, p1, e1 = self.conditions[0].subject, self.conditions[0].predicate, self.conditions[0].object
            e2, p2, e3 = self.conclusion.subject, self.conclusion.predicate, self.conclusion.object
            return f"If there exists an animal which {p1} {e1}, then {e2} {p2} {e3}"
        else:
            raise ValueError(f"Invalid type_id: {self.type_id}")
        
    def to_dict(self):
        d = dict(
            type_id=self.type_id,
            conditions=self.conditions,
            conclusion=self.conclusion,
            rule_id=self.rule_id,
            text=str(self),
        )
        return str(d)
    
    def __repr__(self):
        return f"Rule(type_id={self.type_id}, conditions={repr(self.conditions)}, conclusion={repr(self.conclusion)}, rule_id={self.rule_id})"


class RuleGenerator:
    def __init__(self, prob_negate: float = 0.5):
        self.prob_negate = prob_negate
        self.generator_fn = {
            0: self.generate_from_question_0,
            1: self.generate_from_question_1,
            2: self.generate_from_question_2,
            3: self.generate_from_question_3,
            4: self.generate_from_question_4,
        }

    def generate_from_question(self, q: Statement, type_id: int=-1) -> Tuple[List[Statement], Rule]:
        if type_id == -1:
            type_id = random.randint(0, 4)
        states = self.generator_fn[type_id](q)
        return states, Rule(type_id, copy.deepcopy(states), q)
        
    def sample_predicate(self, n: int = 1) -> List[str]:
        ps = [Predicate(p) for p in random.sample(Predicates, n)]
        ps = [p if random.random() < self.prob_negate else p.negate() for p in ps]
        if len(ps) == 1:
            return ps[0]
        return ps
    
    def sample_subject(self, n: int = 1) -> List[str]:
        es = random.sample(Entities, n)
        if len(es) == 1:
            return es[0]
        return es
    
    def sample_object(self, n: int = 1) -> List[str]:
        es = random.sample(Entities, n)
        if len(es) == 1:
            return es[0]
        return es
    
    def generate_from_question_0(self, q: Statement) -> List[Statement]:
        x, p2, e2 = q.subject, q.predicate, q.object
        p1 = self.sample_predicate()
        e1 = self.sample_object()
        return [Statement(x, p1, e1)]
    
    def generate_from_question_1(self, q: Statement) -> List[Statement]:
        x, p3, e3 = q.subject, q.predicate, q.object
        p1, p2 = self.sample_predicate(2)
        e1, e2 = self.sample_object(2)
        return [Statement(x, p1, e1), Statement(x, p2, e2)]
    
    def generate_from_question_2(self, q: Statement) -> List[Statement]:
        e2, p2, e3 = q.subject, q.predicate, q.object
        p1 = self.sample_predicate()
        e1 = self.sample_object()
        return [Statement(e1, p1, e2)]
    
    def generate_from_question_3(self, q: Statement) -> List[Statement]:
        e2, p3, e4 = q.subject, q.predicate, q.object
        e1, e3 = self.sample_object(2)
        p1, p2 = self.sample_predicate(2)
        return [Statement(e1, p1, e2), Statement(e3, p2, e2)]
    
    def generate_from_question_4(self, q: Statement) -> List[Statement]:
        e2, p2, e3 = q.subject, q.predicate, q.object
        x, e1 = self.sample_subject(2)
        p1 = self.sample_predicate()
        return [Statement(x, p1, e1)]




class Bbehboardgameqabootcamp(Basebootcamp):
    def __init__(self, max_depth: int = 3, prob_conflict: float = 0.5, prob_conflict_type: float = 0.5, prob_negate: float = 0.5):
        """
        参数增强:
        - max_depth: 最大推理深度
        - prob_conflict: 冲突规则概率
        - prob_conflict_type: 冲突类型概率
        - prob_negate: 否定概率
        """
        self.max_depth = max_depth
        self.prob_conflict = prob_conflict
        self.prob_conflict_type = prob_conflict_type
        self.prob_negate = prob_negate
        self.rule_generator = RuleGenerator(prob_negate)

    def generate_theory(self, case: Dict, q: Statement, depth: int = 0):
        def add_rule(rule: Rule):
            rule.rule_id = len(case["rules"])
            case["rules"].append(rule)

        if case['need_remove']:
            if random.random() < 1/(depth+1):
                case['need_remove'] = False
                return

        if depth == 0:
            case["facts"].append(q)
            return case
        
        states, rule = self.rule_generator.generate_from_question(q)
        add_rule(rule)
        if random.random() < self.prob_conflict:
            neg_q = q.negate()
            new_states, new_rule = self.rule_generator.generate_from_question(neg_q)
            add_rule(new_rule)
            if random.random() < self.prob_conflict_type:
                states += new_states
                case["preference_graph"][rule.rule_id] = [new_rule.rule_id]
            else:
                random.shuffle(new_states)
                states += new_states[1:]
                case["preference_graph"][new_rule.rule_id] = [rule.rule_id]

        for state in states:
            self.generate_theory(case, state, depth - 1)


    def case_generator(self) -> Dict:
        """生成一条数据"""
        depth = random.randint(1, self.max_depth)
        e1, p1, e2 = self.rule_generator.sample_subject(), self.rule_generator.sample_predicate(), self.rule_generator.sample_object()
        q = Statement(e1, p1, e2)
        case = {
            "facts": [],
            "rules": [],
            "preference_graph": {},  # 用邻接表表示优先级
            "question": None,
            'depth': depth,
            'answer': None,
        }
        answer = random.choice(['proved', 'disproved', 'unknown'])
        if answer == 'unknown':
            case['need_remove'] = True
        else:
            case['need_remove'] = False
        if answer == 'disproved':
            q = q.negate()
        case['question'] = self.construct_question(q)
        case['answer'] = answer
        self.generate_theory(case, q, depth)
        case['facts'] = [str(f) for f in case['facts']]
        case['rules'] = {r.rule_id: str(r) for r in case['rules']}
        return case

    def construct_question(self, q: Statement) -> str:
        return f"does the {q.subject} {q.predicate} the {q.object}?"

    @staticmethod
    def prompt_func(question_case) -> str:
        """生成符合示例格式的提示"""
        components = [
            "A few players are playing a boardgame. The current state of the game is as follows:",
            *[f"- {fact}" for fact in question_case["facts"]],
            "\nThe rules of the game are as follows:",
            *[f"{k}: {v}" for k,v in question_case["rules"].items()],
            "\nRule priorities (higher priority first):",
            *[f"> {sup} has precedence over {inf}" for sup in question_case["preference_graph"] for inf in question_case["preference_graph"][sup]],
            f"\nQuestion: {question_case['question']}",
            "\nAnswer must be placed within [answer] tags, exactly one of: proved, disproved, unknown."
        ]
        return "\n".join(components)

    @staticmethod
    def extract_output(output: str) -> str:
        """鲁棒的答案提取"""
        # 处理多标签和嵌套情况
        candidates = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL|re.IGNORECASE)
        if not candidates:
            # 回退模式：查找最后一个符合的答案
            candidates = re.findall(r'\b(proved|disproved|unknown)\b', output, re.IGNORECASE)
        
        if candidates:
            final = candidates[-1].strip().lower()
            if final in {'proved', 'disproved', 'unknown'}:
                return final
        return None

    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict) -> bool:
        """实现完整的逻辑验证"""
        answer = identity["answer"]
        model_pred = cls.extract_output(solution)
        return answer == model_pred


if __name__ == "__main__":
    bootcamp = BbehBoardgameQabootcamp(5)
    case = bootcamp.case_generator()
    print(case)
    print(bootcamp.prompt_func(case))
import random
import re
from typing import List, Dict, Tuple
from bootcamp import Basebootcamp


class Bbehshuffobjectbootcamp(Basebootcamp):
    def __init__(self, n=3, context_type=None, object_pool_size=10):
        self.n = n
        self.context_type = context_type or random.choice([
            "game of catch",
            "gift exchange",
            "soccer match",
            "book exchange",
            "square dance"
        ])
        
        # Initialize object pools for different contexts
        self.object_pools = {
            "game of catch": [color for color in ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"]],
            "gift exchange": [color for color in ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"]],
            "soccer match": ["goalkeeper", "defender", "midfielder", "forward", "striker", "left wing", "right wing", "center back", "sweeper", "libero"],
            "book exchange": ["Catch-22", "1984", "To Kill a Mockingbird", "The Great Gatsby", "Brave New World", 
                            "Pride and Prejudice", "The Catcher in the Rye", "Moby Dick", "War and Peace", "The Hobbit"],
            "square dance": [f"partner {chr(ord('A') + i)}" for i in range(10)]
        }
        
        # Trim object pools to specified size
        for k in self.object_pools:
            self.object_pools[k] = self.object_pools[k][:object_pool_size]

    def _generate_people_names(self) -> List[str]:
        names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
        return names[:self.n]

    def _generate_valid_swaps(self, people: List[str]) -> List[Tuple[str, str]]:
        while True:
            swaps = []
            previous_pair = None
            
            for _ in range(self.n):
                # Generate non-consecutive pairs
                while True:
                    a, b = random.sample(people, 2)
                    current_pair = tuple(sorted([a, b]))
                    if current_pair != previous_pair:
                        break
                swaps.append(current_pair)
                previous_pair = current_pair

            # Verify all people involved
            involved = set(p for pair in swaps for p in pair)
            if involved == set(people):
                return swaps

    def case_generator(self) -> Dict:
        people = self._generate_people_names()
        context_type = self.context_type
        object_pool = self.object_pools[context_type]
        objects = random.sample(object_pool, self.n)
        
        # Generate initial assignments
        initial_assignments = {p: o for p, o in zip(people, objects)}
        
        # Generate valid swap sequence
        swaps = self._generate_valid_swaps(people)
        
        # Apply swaps to get final state
        final_state = initial_assignments.copy()
        for a, b in swaps:
            final_state[a], final_state[b] = final_state[b], final_state[a]
        
        # Select question target
        question_person = random.choice(people)
        
        return {
            "people": people,
            "context_type": context_type,
            "object_pool": object_pool,  # 添加object_pool到生成的用例中
            "initial_assignments": initial_assignments,
            "swaps": swaps,
            "question_person": question_person,
            "correct_answer": final_state[question_person]
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        templates = {
            "game of catch": (
                "In a game of catch, {people_list} each have a ball of different colors. "
                "Initially: {initial_desc}.\n"
                "Swaps:\n{swaps_desc}\n"
                "After all swaps, which ball does {target} have? "
                "Please solve this task step by step and put your answer(the color) within <answer></answer>.\n"
            ),
            "gift exchange": (
                "During a gift exchange: {people_list} received gifts. "
                "Initial gifts: {initial_desc}.\n"
                "Swaps:\n{swaps_desc}\n"
                "After swapping, what gift does {target} have? "
                "Please solve this task step by step and put your answer(the color) within <answer></answer>.\n"
            ),
            "soccer match": (
                "In a soccer match positions: {people_list}. "
                "Initial positions: {initial_desc}.\n"
                "Position swaps:\n{swaps_desc}\n"
                "What position does {target} play finally? "
                "Please solve this task step by step and put your answer(the soccer) within <answer></answer>.\n"
            ),
            "book exchange": (
                "Friends swapped books: {people_list}. "
                "Initial books: {initial_desc}.\n"
                "Book swaps:\n{swaps_desc}\n"
                "What book does {target} end up with? "
                "Please solve this task step by step and put your answer(the book) within <answer></answer>.\n"
            ),
            "square dance": (
                "Square dance partners: {people_list}. "
                "Initial partners: {initial_desc}.\n"
                "Partner swaps:\n{swaps_desc}\n"
                "Who is {target}'s final partner? "
                "Please solve this task step by step and put your answer(the final partner's name) within <answer></answer>.\n"
            )
        }

        verbs = {
            "game of catch": "swap balls",
            "gift exchange": "swap gifts",
            "soccer match": "swap positions",
            "book exchange": "swap books",
            "square dance": "swap partners"
        }

        # Format components
        people = question_case["people"]
        people_list = ", ".join(people[:-1]) + f' and {people[-1]}' if len(people) > 1 else people[0]
        
        initial_desc = [f"{p} has {o}" for p, o in question_case["initial_assignments"].items()]
        initial_desc = "; ".join(initial_desc)
        
        swaps_desc = []
        for i, (a, b) in enumerate(question_case["swaps"], 1):
            swaps_desc.append(f"{i}. {a} and {b} {verbs[question_case['context_type']]}.")
        swaps_desc = "\n".join(swaps_desc)
        
        return templates[question_case["context_type"]].format(
            people_list=people_list,
            initial_desc=initial_desc,
            swaps_desc=swaps_desc,
            target=question_case["question_person"]
        )

    @staticmethod
    def extract_output(output: str) -> str:
        matches = re.findall(r'<answer>(.*?)</answer>', output, re.DOTALL)
        # context_type = self.context_type
        # object_pools = self.object_pools[context_type]

        # output_lower = matches.lower()
        # print(f"out_l:{output_lower}")
        # for book in object_pools:
        #     if book.lower() in output_lower:
        #         return book.lower()
        # return None
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict) -> bool:
        # 注意：此处假设solution是通过extract_output(output, identity)获取的
        return solution.strip().lower() == identity["correct_answer"].strip().lower()
from libs.wordladder.solving.puzzle import Puzzle
from libs.wordladder.solving.solution import Solution, CandidateSolution
from libs.wordladder.solving.word_distance_map import WordDistanceMap
from libs.wordladder.words.word import Word


class Solver(object):
    def __init__(self, puzzle: Puzzle):
        self.puzzle: Puzzle = puzzle
        self.start_word: Word = puzzle.start_word
        self.end_word: Word = puzzle.end_word
        self.explored_count: int = 0
        self.solutions: list[Solution] = []
        self.reversed: bool = False
        self.maximum_ladder_length: int = 0
        self.end_distances = None

    def solve(self, maximum_ladder_length: int) -> list[Solution]:
        self.maximum_ladder_length = maximum_ladder_length
        self.start_word: Word = self.puzzle.start_word
        self.end_word: Word = self.puzzle.end_word
        self.reversed = False
        self.explored_count = 0
        self.solutions.clear()

        diffs = self.start_word - self.end_word
        if diffs == 0:
            self.solutions.append(Solution(self.start_word))
            return self.solutions
        elif diffs == 1:
            self.solutions.append(Solution(self.start_word, self.end_word))
            if self.maximum_ladder_length == 2:
                return self.solutions
            elif self.maximum_ladder_length == 3:
                self._short_circuit_ladder_length_3()
                return self.solutions
        elif diffs == 2 and self.maximum_ladder_length == 3:
            self._short_circuit_ladder_length_3()
            return self.solutions

        self.reversed = len(self.start_word.linked_words) > len(self.end_word.linked_words)
        if self.reversed:
            self.start_word = self.puzzle.end_word
            self.end_word = self.puzzle.start_word

        self.end_distances = WordDistanceMap(self.end_word, self.maximum_ladder_length - 1)
        for linked_word in self.start_word.linked_words:
            if self.end_distances.reachable(linked_word, self.maximum_ladder_length):
                self._solve(CandidateSolution(self.start_word, linked_word))
        return self.solutions

    def _solve(self, candidate: CandidateSolution):
        self.explored_count += 1
        last_word = candidate.last_word
        if last_word == self.end_word:
            self.solutions.append(candidate.as_solution(self.reversed))
            return
        if len(candidate) < self.maximum_ladder_length:
            new_max = self.maximum_ladder_length - len(candidate)
            for linked_word in last_word.linked_words:
                if not candidate.seen(linked_word) and self.end_distances.reachable(linked_word, new_max):
                    self._solve(candidate.spawn(linked_word))

    def _short_circuit_ladder_length_3(self):
        common: set[str] = set()
        for word in self.start_word.linked_words:
            common.add(str(word))
        for word in self.end_word.linked_words:
            if str(word) in common:
                self.solutions.append(Solution(self.start_word, word, self.end_word))

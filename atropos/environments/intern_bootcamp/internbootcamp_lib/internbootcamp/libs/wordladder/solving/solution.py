from __future__ import annotations
from libs.wordladder.words.word import Word


class Solution(object):
    __slots__ = ['ladder']

    def __init__(self, *words: Word):
        self.ladder = []
        for word in words:
            self.ladder.append(word)

    def __str__(self):
        return '[' + ','.join([str(word) for word in self.ladder]) + ']'

    def __len__(self):
        return len(self.ladder)

    def __getitem__(self, index) -> Word:
        return self.ladder[index]

    # supports soring
    def __lt__(self, other):
        if not isinstance(other, Solution) or other is None or len(self) < len(other):
            return True
        elif len(self) > len(other):
            return False
        for i in range(len(self)):
            if str(self.ladder[i]) < str(other.ladder[i]):
                return True
            elif str(self.ladder[i]) > str(other.ladder[i]):
                return False
        return False


class CandidateSolution(object):
    __slots__ = ['ladder', 'seen_words']

    def __init__(self, *words: Word):
        self.ladder = []
        for word in words:
            self.ladder.append(word)
        self.seen_words = set()
        for word in self.ladder:
            self.seen_words.add(str(word))

    def seen(self, word: Word) -> bool:
        return str(word) in self.seen_words

    def spawn(self, next_word: Word) -> CandidateSolution:
        result = CandidateSolution(*self.ladder)
        for s in self.seen_words:
            result.seen_words.add(s)
        result.seen_words.add(str(next_word))
        result.ladder.append(next_word)
        return result

    def as_solution(self, reverse: bool) -> Solution:
        if reverse:
            return Solution(*list(reversed(self.ladder)))
        return Solution(*self.ladder)

    @property
    def last_word(self) -> Word:
        return self.ladder[len(self.ladder) - 1]

    def __len__(self):
        return len(self.ladder)

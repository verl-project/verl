from libs.wordladder.solving.word_distance_map import WordDistanceMap
from libs.wordladder.words.word import Word


class Puzzle(object):
    def __init__(self, start_word: Word, end_word: Word):
        self.start_word = start_word
        self.end_word = end_word

    def calculate_minimum_ladder_length(self):
        start: Word = self.start_word
        end: Word = self.end_word
        diffs = start - end
        if diffs == 0 or diffs == 1:
            return diffs + 1
        elif diffs == 2:
            common = set()
            for word in start.linked_words:
                common.add(str(word))
            for word in end.linked_words:
                if str(word) in common:
                    return 3
        if len(start.linked_words) > len(end.linked_words):
            start = self.end_word
            end = self.start_word
        return WordDistanceMap(start)[end]


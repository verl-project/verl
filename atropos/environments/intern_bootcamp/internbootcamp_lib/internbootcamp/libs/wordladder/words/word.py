class Word(object):
    __slots__ = ['actual_word', 'linked_words']
    def __init__(self, actual_word: str):
        self.actual_word = actual_word.upper()
        self.linked_words = []

    @property
    def variations(self):
        vs: list[str] = []
        for i in range(len(self.actual_word)):
            letters = list(self.actual_word)
            letters[i] = '_'
            vs.append(''.join(letters))
        return vs

    def add_link(self, linked_word):
        self.linked_words.append(linked_word)

    @property
    def is_island(self):
        return len(self.linked_words) == 0

    def __sub__(self, other):
        if not isinstance(other, Word):
            return None
        diffs = 0
        for i in range(len(self.actual_word)):
            if self.actual_word[i:i + 1] != other.actual_word[i:i + 1]:
                diffs += 1
        return diffs

    def __str__(self):
        return self.actual_word

    def __contains__(self, item):
        if isinstance(item, Word):
            return item in self.linked_words
        return False
import os.path as path

from libs.wordladder.words.word import Word

FILE_PREFIX = "/resources/dictionary-"
FILE_SUFFIX = "-letter-words.txt"


class Dictionary(object):
    __dictionary_cache = {}

    def __init__(self, word_length):
        self.__word_length = word_length
        if word_length in Dictionary.__dictionary_cache:
            self.__words = Dictionary.__dictionary_cache[word_length]
        else:
            self.__words = {}
            self.__load()
            Dictionary.__dictionary_cache[word_length] = self.__words

    def __load(self):
        filename = path.dirname(path.abspath(__file__)) + FILE_PREFIX + str(self.__word_length) + FILE_SUFFIX
        links_builder = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                word = Word(line[:-1])
                self.__words[str(word)] = word
                for variant in word.variations:
                    existing = links_builder.get(variant)
                    if existing is None:
                        existing = []
                        links_builder[variant] = existing
                    for linkedWord in existing:
                        linkedWord.add_link(word)
                        word.add_link(linkedWord)
                    existing.append(word)

    @property
    def word_length(self):
        return self.__word_length

    def __len__(self):
        return len(self.__words)

    def __getitem__(self, word: str):
        if not isinstance(word, str):
            return None
        return self.__words.get(word.upper())



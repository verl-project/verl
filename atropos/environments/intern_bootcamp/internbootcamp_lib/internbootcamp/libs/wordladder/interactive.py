from libs.wordladder.solving.puzzle import Puzzle
from libs.wordladder.solving.solver import Solver
from libs.wordladder.words.dictionary import Dictionary
from libs.wordladder.words.word import Word
from time import perf_counter


APP_NAME = "WordLadder"
PROMPT_PREFIX = APP_NAME + "> "
MINIMUM_WORD_LENGTH = 2
MAXIMUM_WORD_LENGTH = 15
MINIMUM_LADDER_LENGTH = 1
MAXIMUM_LADDER_LENGTH = 20

STEPS = [
    "Enter start word: ",
    "Enter final word: ",
    'Maximum ladder length? [%d-%d, or return]: ' % (MINIMUM_LADDER_LENGTH, MAXIMUM_LADDER_LENGTH)
]


class Interactive(object):
    def __init__(self, args):
        # not using these yet
        self.args = args
        self.on_step: int = 0
        self.dictionary: Dictionary = None
        self.dictionary_load_time: int = 0
        self.start_word: Word = None
        self.end_word: Word = None
        self.maximum_ladder_length: int = -1



    def run(self):
        again = True
        while again:
            while self.on_step < len(STEPS):
                self._process_input()
            self.on_step = 0
            self._solve()
            print()

            # 无须手动again
            # inp = input('%sRun again? [y/n]: ' % PROMPT_PREFIX)
            # again = inp == "y" or inp == "Y"
            # print()

    def _process_input(self):
        inp = input('%s%s' % (PROMPT_PREFIX, STEPS[self.on_step]))
        ok = False
        if self.on_step == 0:
            ok = self._set_start_word(inp)
        elif self.on_step == 1:
            ok = self._set_end_word(inp)
        elif self.on_step == 2:
            ok = self._set_maximum_ladder_length(inp)
        if ok:
            self.on_step += 1

    def _set_start_word(self, inp: str) -> bool:
        if len(inp) < MINIMUM_WORD_LENGTH or len(inp) > MAXIMUM_WORD_LENGTH:
            print(red('            Please enter a word with between %d and %d characters' % (MINIMUM_WORD_LENGTH, MAXIMUM_WORD_LENGTH)))
            return False
        self._load_dictionary(len(inp))
        self.start_word = self._validate_word(inp)
        return self.start_word is not None

    def _set_end_word(self, inp: str) -> bool:
        if len(inp) != self.dictionary.word_length:
            print(red("            Final word length must match start word length!"))
            return False
        self.end_word = self._validate_word(inp)
        return self.end_word is not None

    def _validate_word(self, inp: str) -> Word or None: # type: ignore
        word: Word = self.dictionary[inp]
        if word is None:
            print(red('            Word \'%s\' does not exist!' % inp))
            return
        elif word.is_island:
            print(red('            Word \'%s\' is an island word (cannot change single letter to form another word)' % inp))
            return
        return word

    def _set_maximum_ladder_length(self, inp: str) -> bool:
        if len(inp) == 0:
            print(green("            No answer - assuming auto calc of minimum ladder length"))
            self.maximum_ladder_length = -1
            return True
        try:
            inp_int = int(inp)
            if inp_int < MINIMUM_LADDER_LENGTH or inp_int > MAXIMUM_LADDER_LENGTH:
                raise TypeError
            self.maximum_ladder_length = inp_int
            return True
        except TypeError:
            print(red('            Invalid input (please enter a integer between %d and %d)' % (MINIMUM_LADDER_LENGTH, MAXIMUM_LADDER_LENGTH)))
        return False

    def _load_dictionary(self, word_length: int):
        start = perf_counter()
        self.dictionary = Dictionary(word_length)
        self.dictionary_load_time = (perf_counter() - start) * 1000

    def _solve(self):
        print('Took %s to load dictionary' % green('%.2fms' % self.dictionary_load_time))
        puzzle = Puzzle(self.start_word, self.end_word)
        if self.maximum_ladder_length == -1:
            start = perf_counter()
            min_ladder = puzzle.calculate_minimum_ladder_length()
            took = (perf_counter() - start) * 1000
            if min_ladder is None:
                print(red('Cannot solve \'%s\' to \'%s\' (took %.2fms to determine that)' % (self.start_word, self.end_word, took)))
                return
            self.maximum_ladder_length = min_ladder
            print('Took %s to determine minimum ladder length of %s' % (green('%.2fms' % took), green('%d' % min_ladder)))
        solver = Solver(puzzle)
        start = perf_counter()
        solutions = solver.solve(self.maximum_ladder_length)
        took = (perf_counter() - start) * 1000
        if len(solutions) == 0:
            print(red('Took %.2fms to find no solutions (explored %d solutions)' % (took, solver.explored_count)))
            return
        print('Took %s to find %s solutions (explored %s solutions)'
              % (green('%.2fms' % took), green(len(solutions)), green(solver.explored_count)))
        self._display_solutions(solutions)

    def _display_solutions(self, solutions):
        solutions.sort()
        page_start: int = 0
        length: int = len(solutions)
        while page_start < (length - 1):
            inp = input(
                '%sList%s solutions? (Enter \'n\' for no, \'y\' or return for next 10, \'all\' for all or how many): '
                % (PROMPT_PREFIX, " more" if page_start > 0 else ""))
            limit = 10
            if inp == "n" or inp == "N":
                return
            elif inp == "all":
                limit = length
            elif len(inp) > 0 and inp != "y" and inp != "Y":
                try:
                    inp_int = int(inp)
                    if inp_int > 0:
                        limit = inp_int
                except TypeError:
                    limit = 10
            for page_start in range(page_start, min(page_start + limit, length)):
                print('%d/%d %s' % (page_start + 1, length, green(solutions[page_start])))


TERMINAL_COLOUR_RED = "\u001b[31m"
TERMINAL_COLOUR_GREEN = "\u001b[32m"
TERMINAL_COLOUR_BLACK = "\u001b[0m"


def green(msg):
    return TERMINAL_COLOUR_GREEN + str(msg) + TERMINAL_COLOUR_BLACK


def red(msg):
    return TERMINAL_COLOUR_RED + str(msg) + TERMINAL_COLOUR_BLACK

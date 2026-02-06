from random import randint, randrange, random
from libs.futoshiki.Futoshiki import FutoshikiPuzzle
from libs.futoshiki.config import ConfigReader
from libs.futoshiki.log import output_log

rand_puzzle = FutoshikiPuzzle.empty_array_returner(5, 5, 'range')
creader = ConfigReader()
config:dict = creader._load_config()

class FutoshikiGenerator(FutoshikiPuzzle):
    @staticmethod
    def generate_logic(size, threshold=0.1):
        logic = []
        for j in range(2*size - 1):
            line = []
            # if j is odd then it's an up or down line
            if (j+1) % 2 == 0:  # j odd
                for i in range(size):
                    a = random()
                    if a >= threshold and a <= 1 - threshold:
                        line.append('')
                    elif a < threshold:
                        line.append(u"\u2228")
                    elif a > 1 - threshold:
                        line.append(u"\u2227")
            else:  # Even line
                for i in range(size - 1):
                    a = random()
                    if a >= threshold and a <= 1 - threshold:
                        line.append('')
                    elif a < threshold:
                        line.append('<')
                    elif a > 1 - threshold:
                        line.append('>')
            logic.append(line)
        return logic

    def generate_numbers(self, max_brute_force_level=2):
        self.index += 1
        if config["DEBUG"]:
            print("目前迭代次数 {}，最大迭代次数 {}".format(self.index, self.max_iter))
        
        # 超过最大迭代次数
        if self.index >= self.max_iter:
            raise StopIteration("超过最大迭代次数，无法生成有效谜题")
        
        try:
            self.solve()
        except KeyError:
            # 重新生成逻辑并递归调用
            self.generated_logic = self.generate_logic(self.size)
            return self.generate_numbers(max_brute_force_level)
        
        if self.solved:
            if config["DEBUG"]:
                print("Humans can solve this puzzle without numbers:")
                print(self.puzzle_printer(self.solvable_puzzle_numbers, self.puzzle_logic))
                print("正常生成成功")
            return self.solvable_puzzle_numbers, self.puzzle_logic
        else:
            self.brute_force(max_brute_force_level=max_brute_force_level, dlog = self.zzlog)
            if self.solved:
                if config["DEBUG"]:
                    print("Found a set of numbers the computer could solve:")
                    print(self.puzzle_printer(self.solvable_puzzle_numbers, self.puzzle_logic))
                    print("暴力破解生成成功")
                return self.solvable_puzzle_numbers, self.puzzle_logic
            else:
                # 重新生成逻辑并递归调用
                self.generated_logic = self.generate_logic(self.size)
                return self.generate_numbers(max_brute_force_level)


    def __init__(self, size,iter_depth ,threshold=None):
        self.size = size
        self.index = 0
        self.max_iter = int(pow(iter_depth,self.size))
        self.generated_logic = self.generate_logic(size)
        self.zzlog = output_log()
        super().__init__(FutoshikiPuzzle.empty_array_returner(size, 1, 0), self.generated_logic,self.zzlog,gen_mode = True)




# l = FutoshikiGenerator.generate_logic(5, 0.1)
# p = FutoshikiPuzzle.empty_array_returner(5, 1, 0)
# # FutoshikiPuzzle.puzzle_printer(p, l)
# fp = FutoshikiPuzzle(p, l)

def test():
    fp = FutoshikiGenerator(5,config["gen_depth"])
    fp.generate_numbers(1)

def generate_futoshiki_puzzle(size: int = 5):
    is_gen_success = False
    puzzle = None
    logic = None
    
    while not is_gen_success:
        try:
            fp = FutoshikiGenerator(size, config["gen_depth"])
            result = fp.generate_numbers(1)
            if result:
                puzzle, logic = result
                is_gen_success = True
        except StopIteration as e:
            if config["DEBUG"]:
                print("遇到 StopIteration 异常，重新尝试生成谜题...")
            continue  # 捕获异常后重新循环
        except Exception as e:
            print(f"遇到意外错误: {e}")
            break  # 避免死循环，退出
    
    return puzzle, logic



# large_fp = FutoshikiGenerator(9, 0.5)
# large_fp.generate_numbers()
# all_solved = True
# while all_solved:
#     fp = FutoshikiGenerator(5)
#     fp.generate_numbers()
#     all_solved = fp.solved
# if not fp.solved:
#     print("Unable to solve puzzle:")
#     print(fp)

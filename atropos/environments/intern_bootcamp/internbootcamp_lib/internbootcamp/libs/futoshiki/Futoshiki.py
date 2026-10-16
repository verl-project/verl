from copy import deepcopy
import multiprocessing as mp

from libs.futoshiki.config import ConfigReader
from libs.futoshiki.log import output_log

creader = ConfigReader()
config:dict = creader._load_config()

class FutoshikiPuzzle:
    """Object representation for a Futoshiki puzzle."""
    @staticmethod
    def number_line_to_string(numbers, logic):
        top = '\u250C\u2500\u2510 '*len(numbers)

        middle = ''
        for k in range(len(numbers)):
            if numbers[k] == 0:
                num = ''
            else:
                num = numbers[k]
            try:
                log = logic[k]
            except IndexError:
                log = ''
            middle += '\u2502{num:1}\u2502{log:1}'.format(num=num, log=log)

        bottom = '\u2514\u2500\u2518 '*len(numbers)

        line = top + '\n' + middle + '\n' + bottom
        return line

    @staticmethod
    def puzzle_printer(puzzle, logic):
        p_to_print = ''
        for i, l_line in enumerate(logic):
            if (i+1) % 2:  # number lines
                line = FutoshikiPuzzle.number_line_to_string(puzzle[i//2], l_line)
                p_to_print += line + '\n'

            else:  # logic lines
                line = ''
                for l in range(len(l_line)-1):
                    line += '{:^3} '.format(l_line[l])
                line += '{:^3}'.format(l_line[-1])
                p_to_print += line + '\n'
        return p_to_print

    @staticmethod
    def empty_array_returner(puzzle_size, item_size, contents):
        e = []
        for j in range(puzzle_size):
            line = []
            for i in range(puzzle_size):
                if item_size == 1:
                    item = contents
                else:
                    item = []
                    for k in range(item_size):
                        if contents is None:
                            item.append(contents)
                        elif contents == "range":
                            # print("Returning range using dimension sizes to infer length")
                            item.append(k+1)
                line.append(item)
            e.append(line)
        return e

    def get_neighbours(self, j, i):
        """Return the cell values around the current cell."""
        neighbours = [None]*4
        try:
            neighbours[0] = self.solution[j][i+1]
        except IndexError:
            pass
        try:
            neighbours[1] = self.solution[j+1][i]
        except IndexError:
            pass
        try:
            neighbours[2] = self.solution[j][i-1]
        except IndexError:
            pass
        try:
            neighbours[3] = self.solution[j-1][i]
        except IndexError:
            pass

        return neighbours

    def single_cell_tester(self, j, i, only_check_two=False):
        """Test the current cell value against neighbouring logic. 'Only check two' allows for the fact that checking only right and down for all squares will be equally valid and quicker if all cells are being checked."""
        neighbours = self.get_neighbours(j, i)
        cell_number = self.solution[j][i]
        nearby_logic = self.logic_matrix[j][i]
        result = True
        if nearby_logic[0] == '>' and neighbours[0] is not None:
            result = result & (cell_number > neighbours[0])
        if nearby_logic[1] == u"\u2228" and neighbours[1] is not None:
            result = result & (cell_number > neighbours[1])
        if nearby_logic[2] == '<' and neighbours[2] is not None and not only_check_two:
            result = result & (cell_number > neighbours[2])
        if nearby_logic[3] == u"\u2227"and neighbours[3] is not None and not only_check_two:
            result = result & (cell_number > neighbours[3])

        if nearby_logic[0] == '<'and neighbours[0] is not None and neighbours[0] != 0:
            result = result & (cell_number < neighbours[0])
        if nearby_logic[1] == u"\u2227" and neighbours[1] is not None and neighbours[1] != 0:
            result = result & (cell_number < neighbours[1])
        if nearby_logic[2] == '>' and neighbours[2] is not None and neighbours[2] != 0 and not only_check_two:
            result = result & (cell_number < neighbours[2])
        if nearby_logic[3] == u"\u2228" and neighbours[3] is not None and neighbours[3] != 0 and not only_check_two:
            result = result & (cell_number < neighbours[3])

        return result

    def valid_solution(self):
        """Checks if the current solution is valid, and returns problem areas if not."""
        all_cells_valid = True
        zero_flag = False
        all_lines_filled = True
        logic_failures = []
        for j, line in enumerate(self.solution):
            for i, n in enumerate(line):
                if n == 0:
                    # any zeroes mean not completed puzzle
                    # print('Not Complete (Zeroes Present) Checking Logic')
                    zero_flag = True
                    self.solved = False
                else:
                    # Applying the tests to the cell
                    cell_valid = self.single_cell_tester(j, i, only_check_two=True)
                    if not cell_valid:
                        # print('Logic Failure caused by cell: {},{} (line, index)'.format(j, i))
                        logic_failures.append((j, i))
                        all_cells_valid = False
                        self.slveod = False
            l = line.copy()
            l.sort()
            # Check the line sorted is the same as an array of 1->size of puzzle
            if l != [k+1 for k in range(self.size)]:
                # print(
                #     "The line {} does not have solely the numbers 1-{} (inclusive), therefore is invalid".format(
                #         self.solution.index(line), self.size))
                all_lines_filled = False
                self.solved = False
        if all_cells_valid and all_lines_filled and not zero_flag:
            if not self.is_gen:
                self.log.log("经过验证，所有内容合法，并且全部格子已解出，解题完毕" + "\n")
            if config["DEBUG"]:
                print("经过验证，所有内容合法，并且全部格子已解出，解题完毕")
            self.solved = True
        return (all_cells_valid, all_lines_filled, zero_flag), logic_failures

    def _logic_find(self):
        """Turn a human-readable logic input to a computer-usable logic matrix where each cell 'knows' the logic around it."""
        # Puzzle size is always the same as the length of the second line of logic.
        for j in range(self.size):
            for i in range(self.size):
                # Finding logic to test this cell against
                try:
                    # right
                    self.logic_matrix[j][i][0] = self.puzzle_logic[j*2][i]
                except IndexError:
                    self.logic_matrix[j][i][0] = None
                try:
                    # below
                    self.logic_matrix[j][i][1] = self.puzzle_logic[(j*2)+1][i]
                except IndexError:
                    self.logic_matrix[j][i][1] = None

                try:
                    # left
                    if i-1 < 0:
                        self.logic_matrix[j][i][2] = None
                    else:
                        self.logic_matrix[j][i][2] = self.puzzle_logic[j*2][i-1]
                except IndexError:
                    self.logic_matrix[j][i][2] = None
                try:
                    # above
                    if (j*2)-1 < 0:
                        self.logic_matrix[j][i][3] = None
                    else:
                        self.logic_matrix[j][i][3] = self.puzzle_logic[(j*2)-1][i]
                except IndexError:
                    self.logic_matrix[j][i][3] = None

    def _cell_lookup_update(self):
        """ Box lookup is a dictionary of box locations, indexed by number of possible values for each location"""
        for m in range(1, self.size+1):
            self.cell_lookup[m] = []
        for j in range(self.size):
            for i in range(self.size):
                self.cell_lookup[len(self.possible_values[j][i])].append((j, i))

    def _possible_value_update(self):
        """Use current solved values to reduce possible values in lines/columns"""
        # if not self.is_gen:
        #     self.log.log("接下来更新所有格子的可能值" + "\n")
        for j in range(self.size):
            for i in range(self.size):
                if self.solution[j][i] != 0:
                    self.possible_values[j][i] = [self.solution[j][i]]
                    row_values_to_change = [k for k in range(self.size) if k != i]
                    col_values_to_change = [k for k in range(self.size) if k != j]

                    for k in row_values_to_change:
                        try:
                            self.possible_values[j][k].remove(self.solution[j][i])
                        except (ValueError, IndexError):
                            pass
                    for l in col_values_to_change:
                        try:
                            self.possible_values[l][i].remove(self.solution[j][i])
                        except (ValueError, IndexError):
                            pass

        self._cell_lookup_update()

    def _solution_update(self):
        """Update solution inputting locations with a single possible value"""
        for j in range(self.size):
            for i in range(self.size):
                if self.solution[j][i] == 0 and len(self.possible_values[j][i]) == 1:
                    self.solution[j][i] = self.possible_values[j][i][0]
                    if not self.is_gen:
                        self.log.log(f"现在可以填充单元格 ({j}, {i})了，因为它只有一个可能的值：{self.possible_values[j][i][0]}." + "\n")
                else:
                    pass
        self._possible_value_update()

    # Solution Algorithms:

    def _only_possible_location(self, j, i):
        """Check whether the current cell is the only location one of its possible values can take."""
        # 获取单元格所在行和列的所有索引（排除单元格的位置）
        row_indices_to_check = [k for k in range(self.size) if k != i]
        col_indices_to_check = [k for k in range(self.size) if k != j]
        
        # 遍历单元格的所有可能值
        for val in self.possible_values[j][i]:
            # 检查该值是否在当前行和列的其他位置出现
            another_val_in_row = False
            for k in row_indices_to_check:
                if val in self.possible_values[j][k]:
                    another_val_in_row = True
                    break

            another_val_in_col = False
            for l in col_indices_to_check:
                if val in self.possible_values[l][i]:
                    another_val_in_col = True
                    break

            # 如果该值在行和列的其他位置都没有出现，则将该值设为单元格的唯一可能值
            if (not another_val_in_col) or (not another_val_in_row):
                self.possible_values[j][i] = [val]
                self._solution_update()
                if not self.is_gen:
                    self.log.log(f"单元格 ({j}, {i}) 的唯一可能值为 {val}，更新该单元格的值。" + "\n")


    def _recursive_less_than(self, j, i):
        """Follow 'less than' signs until maximum cell is found, and recurse possible value changes (ie all poss values less than max in adjacent cell) back to the original cell the function is initially called from."""
        # 单元格的值需要小于相邻单元格的值
        less_than_other_cell = False
        
        # 如果单元格的逻辑是“小于”符号
        if self.logic_matrix[j][i][0] == '<':
            less_than_other_cell = True
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 与右侧单元格比较，检查是否满足“小于”条件。" + "\n")
            self._recursive_less_than(j, i+1)
            
            try:
                # 更新单元格的可能值，确保单元格的值小于右侧单元格的最大值
                self.possible_values[j][i] = [x for x in self.possible_values[j][i] if x < max(self.possible_values[j][i+1])]
                self._solution_update()
            except ValueError:
                # 如果右侧单元格没有有效的值可供比较，忽略这个错误
                if not self.is_gen:
                    self.log.log(f"右侧单元格 ({j}, {i+1}) 没有有效的值可供比较，跳过此步骤。" + "\n")

        # 检查单元格的逻辑是否是“与”符号（逻辑与）
        if self.logic_matrix[j][i][1] == u"\u2227":
            less_than_other_cell = True
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 与下方单元格比较，检查是否满足“小于”条件。" + "\n")
            self._recursive_less_than(j+1, i)
            
            try:
                # 更新单元格的可能值，确保单元格的值小于下方单元格的最大值
                self.possible_values[j][i] = [x for x in self.possible_values[j][i] if x < max(self.possible_values[j+1][i])]
                self._solution_update()
            except ValueError:
                # 如果下方单元格没有有效的值可供比较，忽略这个错误
                if not self.is_gen:
                    self.log.log(f"下方单元格 ({j+1}, {i}) 没有有效的值可供比较，跳过此步骤。" + "\n")

        # 检查单元格的逻辑是否是“大于”符号
        if self.logic_matrix[j][i][2] == '>':
            less_than_other_cell = True
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 与左侧单元格比较，检查是否满足“小于”条件。" + "\n")
            self._recursive_less_than(j, i-1)
            
            try:
                # 更新单元格的可能值，确保单元格的值小于左侧单元格的最大值
                self.possible_values[j][i] = [x for x in self.possible_values[j][i] if x < max(self.possible_values[j][i-1])]
                self._solution_update()
            except ValueError:
                # 如果左侧单元格没有有效的值可供比较，忽略这个错误
                if not self.is_gen:
                    self.log.log(f"左侧单元格 ({j}, {i-1}) 没有有效的值可供比较，跳过此步骤。" + "\n")

        # 检查单元格的逻辑是否是“或”符号（逻辑或）
        if self.logic_matrix[j][i][3] == u"\u2228":
            less_than_other_cell = True
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 与上方单元格比较，检查是否满足“小于”条件。" + "\n")
            self._recursive_less_than(j-1, i)
            
            try:
                # 更新单元格的可能值，确保单元格的值小于上方单元格的最大值
                self.possible_values[j][i] = [x for x in self.possible_values[j][i] if x < max(self.possible_values[j-1][i])]
                self._solution_update()
            except ValueError:
                # 如果上方单元格没有有效的值可供比较，忽略这个错误
                if not self.is_gen:
                    self.log.log(f"上方单元格 ({j-1}, {i}) 没有有效的值可供比较，跳过此步骤。" + "\n")

        # 如果单元格没有逻辑符号（即所有位置都为空），则什么也不做
        if all(x == None or x == '' for x in self.logic_matrix[j][i]):
            return

        # 如果单元格没有符合条件的操作（即没有“小于”条件），则结束递归
        elif not less_than_other_cell:
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 没有满足“小于”条件的相邻单元格。" + "\n")
            return


    def _recursive_more_than(self, j, i):
        """Follow 'more than' signs until minimum cell is found, and recurse possible value changes (ie all poss values more than min in adjacent cell) back to the original cell the function is initially called from."""
        # 单元格的值需要大于相邻单元格的值
        more_than_other_cell = False
        
        # 如果单元格的逻辑是“大于”符号
        if self.logic_matrix[j][i][0] == '>':
            more_than_other_cell = True
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 与右侧单元格比较，检查是否满足“大于”条件。" + "\n")
            self._recursive_more_than(j, i+1)
            
            try:
                # 更新单元格的可能值，确保单元格的值大于右侧单元格的最小值
                self.possible_values[j][i] = [x for x in self.possible_values[j][i] if x > min(self.possible_values[j][i+1])]
                self._solution_update()
            except ValueError:
                # 如果右侧单元格没有有效的值可供比较，忽略这个错误
                if not self.is_gen:
                    self.log.log(f"右侧单元格 ({j}, {i+1}) 没有有效的值可供比较，跳过此步骤。" + "\n")

        # 检查单元格的逻辑是否是“或”符号（逻辑或）
        if self.logic_matrix[j][i][1] == u"\u2228":
            more_than_other_cell = True
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 与下方单元格比较，检查是否满足“大于”条件。" + "\n")
            self._recursive_more_than(j+1, i)
            
            try:
                # 更新单元格的可能值，确保单元格的值大于下方单元格的最小值
                self.possible_values[j][i] = [x for x in self.possible_values[j][i] if x > min(self.possible_values[j+1][i])]
                self._solution_update()
            except ValueError:
                # 如果下方单元格没有有效的值可供比较，忽略这个错误
                if not self.is_gen:
                    self.log.log(f"下方单元格 ({j+1}, {i}) 没有有效的值可供比较，跳过此步骤。" + "\n")

        # 检查单元格的逻辑是否是“小于”符号
        if self.logic_matrix[j][i][2] == '<':
            more_than_other_cell = True
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 与左侧单元格比较，检查是否满足“大于”条件。" + "\n")
            self._recursive_more_than(j, i-1)
            
            try:
                # 更新单元格的可能值，确保单元格的值大于左侧单元格的最小值
                self.possible_values[j][i] = [x for x in self.possible_values[j][i] if x > min(self.possible_values[j][i-1])]
                self._solution_update()
            except ValueError:
                # 如果左侧单元格没有有效的值可供比较，忽略这个错误
                if not self.is_gen:
                    self.log.log(f"左侧单元格 ({j}, {i-1}) 没有有效的值可供比较，跳过此步骤。" + "\n")

        # 检查单元格的逻辑是否是“与”符号（逻辑与）
        if self.logic_matrix[j][i][3] == u"\u2227":
            more_than_other_cell = True
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 与上方单元格比较，检查是否满足“大于”条件。" + "\n")
            self._recursive_more_than(j-1, i)
            
            try:
                # 更新单元格的可能值，确保单元格的值大于上方单元格的最小值
                self.possible_values[j][i] = [x for x in self.possible_values[j][i] if x > min(self.possible_values[j-1][i])]
                self._solution_update()
            except ValueError:
                # 如果上方单元格没有有效的值可供比较，忽略这个错误
                if not self.is_gen:
                    self.log.log(f"上方单元格 ({j-1}, {i}) 没有有效的值可供比较，跳过此步骤。" + "\n")

        # 如果单元格没有逻辑符号（即所有位置都为空），则什么也不做
        if all(x == None or x == '' for x in self.logic_matrix[j][i]):
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 没有逻辑符号，无需进行比较。" + "\n")
            return

        # 如果单元格没有符合条件的操作（即没有“大于”条件），则结束递归
        elif not more_than_other_cell:
            if not self.is_gen:
                self.log.log(f"单元格 ({j}, {i}) 没有满足“大于”条件的相邻单元格。" + "\n")
            return


    def _line_match(self, line, match_list):
        """Find locations in a line (row/column) where the possible values are the same, and if matches are found then remove the relevant possible values from all the other locations."""
        # 查找与匹配列表相同的值在行（或列）中的位置
        line_matches = [index for index, val in enumerate(line) if val == match_list]
        
        # 如果找到了完全匹配
        if len(line_matches) == len(match_list):
            # 查找不匹配的位置
            non_matched_line_indices = [x for x in range(self.size) if x not in line_matches]
            
            for index in non_matched_line_indices:
                try:
                    # 移除这些位置的所有匹配值
                    for num in match_list:
                        line[index].remove(num)
                except ValueError:
                    # 如果某个值不存在，继续执行
                    continue
            self._solution_update()

        # Now all exact matches are removed, look for partial matches:
        if len(match_list) == 2 and len(line_matches) == 1:
            rest_of_line = list(filter(lambda a: a != match_list, deepcopy(line)))
            third_values_attempted = []
            for item in rest_of_line:
                if len(item) == 3 and all(x in item for x in match_list):
                    # all of the match list is contained in new item
                    third_val = [x for x in deepcopy(item) if x not in match_list]
                    if third_val[0] in third_values_attempted:
                        pass
                    else:
                        third_values_attempted.append(third_val[0])
                        new_match_value = deepcopy(match_list)
                        new_match_value.append(third_val[0])
                        new_match_value.sort()

                        new_line_matches = [index for index,
                                            val in enumerate(line) if (val == new_match_value) or (val == match_list)]

                        if len(new_line_matches) == 3:
                            non_matched_line_indices = [x for x in range(
                                self.size) if x not in new_line_matches]
                            for index in non_matched_line_indices:
                                for num in new_match_value:
                                    try:
                                        line[index].remove(num)
                                    except ValueError:
                                        continue
                            self._solution_update()

        return line

    def _find_matches(self, j, i):
        """Run the line matching function on the row and column of the current cell."""
        match_list = self.possible_values[j][i]
        if not self.is_gen:
            self.log.log(f"现在检查单元格 ({j}, {i}) 的可能值与行列中其他值的是否相同，如果有确定的数字，则它不再可能出现在同行、同列，从其同行、同列删除所有可能值。" + "\n")
        
        # 检查单元格所在行的匹配情况
        row = self.possible_values[j].copy()
        row = self._line_match(row, match_list)
        
        # 检查单元格所在列的匹配情况
        col = [x[i] for x in self.possible_values.copy()]
        col = self._line_match(col, match_list)


    def _corner_rule(self, j, i):
        match_list = self.possible_values[j][i]
        if len(match_list) != 2:
            return
        else:
            row = self.possible_values[j].copy()
            col = [x[i] for x in self.possible_values.copy()]
            row_matches = [index for index, val in enumerate(row) if val == match_list]
            if len(row_matches) != 2:
                return
            else:
                i1, i2 = row_matches[0], row_matches[1]
                if ((self.logic_matrix[j][i1][1] == u"\u2227" and self.logic_matrix[j][i2][3] == u"\u2228") and self.possible_values[j+1][i1] == self.possible_values[j-1][i2]):
                    # print("Both pairs less than same pair - applying corner rule 1")
                    num_to_remove = max(self.possible_values[j+1][i1])
                    try:
                        self.possible_values[j+1][i2].remove(num_to_remove)
                        self._solution_update()
                    except ValueError:
                        pass
                    try:
                        self.possible_values[j-1][i1].remove(num_to_remove)
                        self._solution_update()
                    except ValueError:
                        pass

                if (self.logic_matrix[j][i1][3] == u"\u2228" and self.logic_matrix[j][i2][1] == u"\u2227" and self.possible_values[j-1][i1] == self.possible_values[j+1][i2]):
                    # print("Both pairs less than same pair - applying corner rule 2")
                    num_to_remove = max(self.possible_values[j-1][i1])
                    try:
                        self.possible_values[j+1][i1].remove(num_to_remove)
                        self._solution_update()
                    except ValueError:
                        pass
                    try:
                        self.possible_values[j-1][i2].remove(num_to_remove)
                        self._solution_update()
                    except ValueError:
                        pass
            return

    def not_solved_print(self):
        if not self.is_gen:
            self.log.log("解题失败，没能求出唯一解，但是当前的所有格子情况是：" + "\n")
        if config["DEBUG"]:
            print("Not solved, but current state of solution:")
            print(FutoshikiPuzzle.puzzle_printer(self.solution, self.puzzle_logic))
            print("And current possible values:")
            for line in self.possible_values:
                print(line)

        

    def solve(self , max_iterations=20 ):
        """Solve the Futoshiki Puzzle using the algorithms implemented in this class."""
        if not self.is_gen:
            self.log.log("现在开始求解此futoshiki谜题" + "\n")
        if config["DEBUG"]:
            print("Solving Futoshiki")
            print(FutoshikiPuzzle.puzzle_printer(self.puzzle_numbers, self.puzzle_logic))
        t = 0

        while t < max_iterations and len(self.cell_lookup[1]) < self.size * self.size:
            for j in range(self.size):
                for i in range(self.size):
                    # print("Cell: {}, {}".format(cell[0], cell[1]))
                    self._recursive_more_than(j, i)
                    self._recursive_less_than(j, i)
                    self._only_possible_location(j, i)
                    self._find_matches(j, i)
                    self._corner_rule(j, i)

                    if not self.is_gen:
                        self.log.log("更新完毕，现在棋盘上的所有格可能的值是：" + "\n")
                        for line in self.possible_values:
                            self.log.log(str(line) + "\n")
                    if config["DEBUG"]:
                        print("这一步棋盘上的所有可能性是：")
                        for line in self.possible_values:
                            print(line)
            t += 1


        self.valid_solution()

        if len(self.cell_lookup[1]) == self.size * self.size and self.solved:
            if not self.is_gen:
                self.log.log("此futoshiki求解完成，经历了{}次迭代。".format(t) + "\n")
            if config["DEBUG"]:
                print("Solution found after {} iterations:".format(t))
                print(self.puzzle_printer(self.solution, self.puzzle_logic))
            self.solvable_puzzle_numbers = self.puzzle_numbers
        else:
            self.not_solved_print()
        return str(self.possible_values)

    def brute_force(self, dlog:output_log,max_brute_force_level=None):
        self.log = dlog
        """Try each possible value for each cell until a solution is found."""
        if max_brute_force_level is None:
            max_brute_force_level = self.size
        if self.solved:
            print("Problem is solved, no need for brute force")
            return
        else:
            # print("Brute Forcing at depth: {}".format(self.brute_force_level))
            for m in range(2, self.size):
                # Parallelize this setup using multiprocessing
                for cell in self.cell_lookup[m]:
                    if not self.is_gen:
                        self.log.log("\n接下来对坐标为[{},{}]的数字进行暴力破解，将逐步尝试这个单元格的所有可能性".format(
                        cell[0], cell[1]) + "\n")
                    if config["DEBUG"]:
                        print()
                        print("接下来对坐标为[{},{}]的数字进行暴力破解，将逐步尝试这个单元格的所有可能性".format(
                        cell[0], cell[1]))
                    for index in range(len(self.possible_values[cell[0]][cell[1]])):
                        test_puzzle = deepcopy(self.solution)
                        test_value = self.possible_values[cell[0]][cell[1]][index] 
                        test_puzzle[cell[0]][cell[1]
                                             ] = self.possible_values[cell[0]][cell[1]][index]
                        t = FutoshikiPuzzle(test_puzzle, self.puzzle_logic,self.log)
                        t.brute_force_level = self.brute_force_level + 1
                        try:
                            t.solve()
                            if not self.is_gen:
                                if not t.solved or len(t.possible_values[cell[0]][cell[1]]) > 1:  # 判断是否有多个解或没有解
                                    self.log.log("假设此格内容为{}时，无法解出唯一答案，仍有格子存在不止一种可能，接下来进行下一次尝试".format(test_value) + "\n")  
                                    if config["DEBUG"]:
                                        print("假设此格内容为{}时，无法解出唯一答案，仍有格子存在不止一种可能，接下来进行下一次尝试".format(test_value))
                        except KeyError:
                            # Must have zero possible values, meaning this index is an invalid option
                            pass
                        if t.solved:
                            if config["DEBUG"]:
                                print("Problem solved through brute forcing at depth {}".format(
                                self.brute_force_level))
                            self.solvable_puzzle_numbers = t.solvable_puzzle_numbers
                            self.solution = t.solution
                            self._solution_update()
                            self.solved = True
                            return
                        else:
                            if t.brute_force_level < max_brute_force_level:
                                try:
                                    t.brute_force(max_brute_force_level,dlog=self.log)
                                    if t.solved:
                                        if config["DEBUG"]:
                                            print("Problem solved through brute forcing at depth {}".format(
                                            self.brute_force_level))
                                        self.solvable_puzzle_numbers = t.solvable_puzzle_numbers
                                        self.solution = t.solution
                                        self._solution_update()
                                        self.solved = True
                                        return
                                    else:
                                        pass
                                except KeyError:
                                    # Must have zero possible values
                                    pass
                            else:
                                pass
        return

    def __init__(self, initial_puzzle_numbers:str, puzzle_logic, alog:output_log,gen_mode = True):
        self.puzzle_numbers = deepcopy(initial_puzzle_numbers)
        self.solvable_puzzle_numbers = None
        self.puzzle_logic = deepcopy(puzzle_logic)
        self.size = len(initial_puzzle_numbers[0])
        self.solution = deepcopy(initial_puzzle_numbers)
        self.solved = False
        self.brute_force_level = 0
        self.possible_values = self.empty_array_returner(self.size, self.size, 'range')
        self.logic_matrix = self.empty_array_returner(self.size, 4, None)
        self.cell_lookup = {}

        self.is_gen = gen_mode
        self.log = alog

        self._logic_find()
        self._solution_update()

    def __repr__(self):
        out = "Futoshiki Puzzle:\n"
        out += self.puzzle_printer(self.puzzle_numbers, self.puzzle_logic) + "\n"
        if self.solved:
            out += "Solution found:\n"
            out += self.puzzle_printer(self.solution, self.puzzle_logic) + "\n"
        return out

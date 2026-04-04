from copy import deepcopy
from libs.futoshiki.Futoshiki import FutoshikiPuzzle
from libs.futoshiki.log import output_log

############################
# Functional Below this line
############################


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
                        item.append(k + 1)
            line.append(item)
        e.append(line)
    return e


def puzzle_printer(puzzle, logic):
    for i, l_line in enumerate(logic):
        if (i + 1) % 2:  # number lines
            line = ""
            for j in range(len(puzzle[0]) - 1):
                line += "{n}{l:1}".format(n=puzzle[i // 2][j], l=l_line[j])
            line += "{}".format(puzzle[i // 2][-1])
            print(line)

        else:  # logic lines
            line = ""
            for l in range(len(l_line) - 1):
                line += "{:1} ".format(l_line[l])
            line += "{}".format(l_line[-1])
            print(line)


def logic_finder(logic):
    # Puzzle size is always the same as the length of the second line of logic.
    puzzle_size = len(logic[1])
    logic_matrix = empty_array_returner(puzzle_size, 4, None)
    for j in range(puzzle_size):
        for i in range(puzzle_size):
            # Finding logic to test this cell against
            try:
                # right
                logic_matrix[j][i][0] = logic[j * 2][i]
            except IndexError:
                logic_matrix[j][i][0] = None
            try:
                # below
                logic_matrix[j][i][1] = logic[(j * 2) + 1][i]
            except IndexError:
                logic_matrix[j][i][1] = None

            try:
                # left
                if i - 1 < 0:
                    logic_matrix[j][i][2] = None
                else:
                    logic_matrix[j][i][2] = logic[j * 2][i - 1]
            except IndexError:
                logic_matrix[j][i][2] = None
            try:
                # above
                if (j * 2) - 1 < 0:
                    logic_matrix[j][i][3] = None
                else:
                    logic_matrix[j][i][3] = logic[(j * 2) - 1][i]
            except IndexError:
                logic_matrix[j][i][3] = None
    return logic_matrix


def get_neighbours(puzzle, j, i):
    neighbours = [None] * 4
    try:
        neighbours[0] = puzzle[j][i + 1]
    except IndexError:
        pass
    try:
        neighbours[1] = puzzle[j + 1][i]
    except IndexError:
        pass
    try:
        neighbours[2] = puzzle[j][i - 1]
    except IndexError:
        pass
    try:
        neighbours[3] = puzzle[j - 1][i]
    except IndexError:
        pass

    return neighbours


def single_cell_tester(cell_number, neighbours, nearby_logic, only_check_two=False):
    # "Only check two" allows for the fact that checking only right and down for all squares will be equally valid and quicker.
    result = True
    if nearby_logic[0] == ">" and neighbours[0] is not None:
        result = result & (cell_number > neighbours[0])
    if nearby_logic[1] == u"\u2228" and neighbours[1] is not None:
        result = result & (cell_number > neighbours[1])
    if nearby_logic[2] == "<" and neighbours[2] is not None and not only_check_two:
        result = result & (cell_number > neighbours[2])
    if (
        nearby_logic[3] == u"\u2227"
        and neighbours[3] is not None
        and not only_check_two
    ):
        result = result & (cell_number > neighbours[3])

    if nearby_logic[0] == "<" and neighbours[0] is not None and neighbours[0] != 0:
        result = result & (cell_number < neighbours[0])
    if (
        nearby_logic[1] == u"\u2227"
        and neighbours[1] is not None
        and neighbours[1] != 0
    ):
        result = result & (cell_number < neighbours[1])
    if (
        nearby_logic[2] == ">"
        and neighbours[2] is not None
        and neighbours[2] != 0
        and not only_check_two
    ):
        result = result & (cell_number < neighbours[2])
    if (
        nearby_logic[3] == u"\u2228"
        and neighbours[3] is not None
        and neighbours[3] != 0
        and not only_check_two
    ):
        result = result & (cell_number < neighbours[3])

    return result


def get_possible_neighbours(p, j, i, index=0):
    neighbours = [None] * 4
    try:
        neighbours[0] = p[j][i + 1][index]
    except IndexError:
        pass
    try:
        neighbours[1] = p[j + 1][i][index]
    except IndexError:
        pass
    try:
        neighbours[2] = p[j][i - 1][index]
    except IndexError:
        pass
    try:
        neighbours[3] = p[j - 1][i][index]
    except IndexError:
        pass

    return neighbours


def valid(puzzle, logic):
    tests_to_be_performed = logic_finder(logic)
    all_cells_valid = True
    zero_flag = False
    all_lines_filled = True
    logic_failures = []
    for j, line in enumerate(puzzle):
        for i, n in enumerate(line):
            if n == 0:
                # any zeroes mean not completed puzzle
                print("Not Complete (Zeroes Present) Checking Logic")
                zero_flag = True
            else:
                # Applying the tests to the cell
                cell_valid = single_cell_tester(
                    cell_number=puzzle[j][i],
                    neighbours=get_neighbours(puzzle, j, i),
                    nearby_logic=tests_to_be_performed[j][i],
                    only_check_two=True,
                )
            if not cell_valid:
                print("Logic Failure caused by cell: {},{} (line, index)".format(j, i))
                logic_failures.append((j, i))
                all_cells_valid = False
        l = line.copy()
        l.sort()
        # Check the line sorted is the same as an array of 1->size of puzzle
        if l != [k + 1 for k in range(len(puzzle[0]))]:
            print(
                "The line {} does not have solely the numbers 1-{} (inclusive), therefore is invalid".format(
                    puzzle.index(line), len(puzzle[0])
                )
            )
            all_lines_filled = False

    return (all_cells_valid, all_lines_filled, zero_flag), logic_failures


def recursive_more_than(logic_matrix, p, j, i):
    nearby_logic = logic_matrix[j][i]
    # print("Currently in {},{}".format(j, i))
    # CHECK WHETHER RECURSION COULD BE MOVED UP
    # next cell is less than current cell
    if nearby_logic[0] == ">":
        try:
            p[j][i] = [x for x in p[j][i] if x > min(p[j][i + 1])]
        except ValueError:
            pass
        recursive_more_than(logic_matrix, p, j, i + 1)

    if nearby_logic[1] == u"\u2228":
        try:
            # p[j][i].remove((min(p[j+1][i])))
            p[j][i] = [x for x in p[j][i] if x > min(p[j + 1][i])]
        except ValueError:
            pass
        recursive_more_than(logic_matrix, p, j + 1, i)
    if nearby_logic[2] == "<":
        try:
            # p[j][i].remove((min(p[j][i-1])))
            p[j][i] = [x for x in p[j][i] if x > min(p[j][i - 1])]
        except ValueError:
            pass
        recursive_more_than(logic_matrix, p, j, i - 1)
    if nearby_logic[3] == u"\u2227":
        try:
            # p[j][i].remove((min(p[j-1][i])))
            p[j][i] = [x for x in p[j][i] if x > min(p[j - 1][i])]
        except ValueError:
            pass
        recursive_more_than(logic_matrix, p, j - 1, i)
    return p


def recursive_less_than(logic_matrix, p, j, i):
    nearby_logic = logic_matrix[j][i]
    # print("Currently in {},{}".format(j, i))
    # next cell is greater than current cell
    if nearby_logic[0] == "<":
        try:
            p[j][i] = [x for x in p[j][i] if x < max(p[j][i + 1])]
        except ValueError:
            pass
        recursive_less_than(logic_matrix, p, j, i + 1)

    if nearby_logic[1] == u"\u2227":
        try:
            p[j][i] = [x for x in p[j][i] if x < max(p[j + 1][i])]
        except ValueError:
            pass
        recursive_less_than(logic_matrix, p, j + 1, i)
    if nearby_logic[2] == ">":
        try:
            p[j][i] = [x for x in p[j][i] if x < max(p[j][i - 1])]
        except ValueError:
            pass
        recursive_less_than(logic_matrix, p, j, i - 1)
    if nearby_logic[3] == u"\u2228":
        try:
            p[j][i] = [x for x in p[j][i] if x < max(p[j - 1][i])]
        except ValueError:
            pass
        recursive_less_than(logic_matrix, p, j - 1, i)
    return p


def line_match(row, match_list):
    row_matches = [index for index, val in enumerate(row) if val == match_list]
    if len(row_matches) == len(match_list):
        non_matched_row_indices = [x for x in range(len(row)) if x not in row_matches]
        for index in non_matched_row_indices:
            try:
                for num in match_list:
                    row[index].remove(num)
            except ValueError:
                continue

    # Now all exact matches are removed, look for partial matches:
    if len(match_list) == 2 and len(row_matches) == 1:
        # print("Partial matching: match_list {} and \n row:{} ".format(match_list, row))
        rest_of_row = list(filter(lambda a: a != match_list, deepcopy(row)))
        third_values_attempted = []
        for item in rest_of_row:
            if len(item) == 3 and all(
                x in item for x in match_list
            ):  # all of the match list is contained

                ic = [x for x in deepcopy(item) if x not in match_list]
                # ic.remove(match_list[0])
                # ic.remove(match_list[1])
                # print("Checking {}, Third value in item = {}".format(item, ic[0]))
                if ic[0] in third_values_attempted:
                    pass
                else:
                    third_values_attempted.append(ic[0])
                    new_match_value = deepcopy(match_list)
                    new_match_value.append(ic[0])
                    new_match_value.sort()

                    new_row_matches = [
                        index
                        for index, val in enumerate(row)
                        if (val == new_match_value) or (val == match_list)
                    ]

                    if len(new_row_matches) == 3:
                        # print("Partial matching match_list: {} extended to: {} for \nrow: {}".format(
                        # match_list, new_match_value, row))

                        # print("Found {} matches, removing {} from row ".format(
                        # len(new_row_matches), new_match_value))
                        non_matched_row_indices = [
                            x for x in range(len(row)) if x not in new_row_matches
                        ]
                        for index in non_matched_row_indices:
                            for num in new_match_value:
                                try:
                                    row[index].remove(num)
                                except ValueError:
                                    continue
                        # print("New row: {}".format(row))

    return row


def only_possible_location(line, index):
    other_indices = [k for k in range(len(line))]
    other_indices.remove(index)

    poss_values = line[index]

    for val in poss_values:
        other_occurences = 0
        for i in other_indices:
            if val in line[i]:
                # Count up if there is another possible location
                other_occurences += 1
        if other_occurences == 0:
            # If none then this place is the only one you can put this value
            line[index] = [val]
    return line


def poss_locations(pc, j, i):
    row = pc[j].copy()
    row = only_possible_location(row, i)
    pc[j] = row
    col = [x[i] for x in pc.copy()]
    col = only_possible_location(col, j)
    for m in range(len(pc[0])):
        pc[m][i] = col[m]
    return pc


def find_matches(pc, j, i):
    match_list = pc[j][i]
    # Match the row first
    row = pc[j].copy()
    row = line_match(row, match_list)
    # Match the column next
    col = [x[i] for x in pc.copy()]
    col = line_match(col, match_list)
    # As shallow copies have been used, the possibility matrix will have been altered
    return pc


def corner_rule(logic_matrix, pc, j, i):
    match_list = pc[j][i]
    if len(match_list) != 2:
        return pc
    else:
        row = pc[j].copy()
        col = [x[i] for x in pc.copy()]
        row_matches = [index for index, val in enumerate(row) if val == match_list]
        if len(row_matches) != 2:
            return pc
        else:
            i1, i2 = row_matches[0], row_matches[1]
            if (
                logic_matrix[j][i1][1] == u"\u2227"
                and logic_matrix[j][i2][3] == u"\u2228"
            ) and pc[j + 1][i1] == pc[j - 1][i2]:
                # print("Both pairs less than same pair - applying corner rule 1")
                num_to_remove = max(pc[j + 1][i1])
                try:
                    pc[j + 1][i2].remove(num_to_remove)
                except ValueError:
                    pass
                try:
                    pc[j - 1][i1].remove(num_to_remove)
                except ValueError:
                    pass
            if (
                logic_matrix[j][i1][3] == u"\u2228"
                and logic_matrix[j][i2][1] == u"\u2227"
                and pc[j - 1][i1] == pc[j + 1][i2]
            ):
                # print("Both pairs less than same pair - applying corner rule 2")
                num_to_remove = max(pc[j - 1][i1])
                try:
                    pc[j + 1][i1].remove(num_to_remove)
                except ValueError:
                    pass
                try:
                    pc[j - 1][i2].remove(num_to_remove)
                except ValueError:
                    pass

        # col_matches = [index for index, val in enumerate(col) if val == match_list]

        return pc


def possible_values(puzzle, logic, p):
    logic_layout = logic_finder(logic)
    pc = deepcopy(p)
    for j, line in enumerate(puzzle):
        for i, number in enumerate(line):
            if number != 0 or len(pc[j][i]) == 1:
                if number == 0:
                    no = pc[j][i][0]
                else:
                    no = number
                # print('Removing {} from column and line due to appearance at {}, {}'.format(number, j, i))
                other_line = [k for k in range(len(pc[0]))]
                other_line.remove(i)

                for k in other_line:
                    try:
                        # print('{},{}'.format(l, i))
                        pc[j][k].remove(no)
                    except (ValueError, IndexError):
                        pass
                other_column = [k for k in range(len(pc[0]))]
                other_column.remove(j)
                for l in other_column:
                    try:
                        pc[l][i].remove(no)
                    except (ValueError, IndexError):
                        pass
                pc[j][i] = [no]
            else:
                # Removing the maxima from neighbours in 'less than' locations, and minima from 'more than' locations
                pc = recursive_more_than(logic_layout, pc, j, i)
                # current cell is less than the next cell along
                pc = recursive_less_than(logic_layout, pc, j, i)
                if len(pc[j][i]) == 2 or len(pc[j][i]) == 3:
                    pc = find_matches(pc, j, i)
                pc = poss_locations(pc, j, i)
                pc = corner_rule(logic_layout, pc, j, i)

    return pc


def brute_force(puzzle, logic, p):
    logic_matrix = logic_finder(logic)
    new_p = deepcopy(p)
    new_puzzle = deepcopy(puzzle)
    # Box lookup is a dictionary of box locations, indexed by number of possible values for each location
    box_lookup = {}
    # We want boxes with two values, up to boxes with any possible value (ie size of puzzle)
    for m in range(2, len(p[0]) + 1):
        box_lookup[m] = []

    for j in range(len(new_puzzle[0])):
        for i in range(len(new_puzzle[0])):
            if len(p[j][i]) == 1:
                new_puzzle[j][i] = new_p[j][i][0]
            else:
                box_lookup[len(new_p[j][i])].append((j, i))
    # print(box_lookup)
    solved = False
    for m in range(2, len(p[0]) + 1):
        for box in box_lookup[m]:
            print("Attempting brute force at cell {},{}".format(box[0], box[1]))
            for index in range(len(new_p[box[0]][box[1]])):
                test_puzzle = new_puzzle
                # Try value at index as the value for this box:
                test_puzzle[box[0]][box[1]] = new_p[box[0]][box[1]][index]
                pc, t = consistent_past_values(test_puzzle, logic)
                # pc, t, solved = solve(test_puzzle, logic, brute_force_if_needed=True)
                solved = True
                # # TODO fix this bit up so that it brute forces all options below a current option, if all of those don't work then it removes the current option.
                # if not solved:
                #     new_p[box[0]][box[1]].pop(index)
                #     # remove option that does not solve puzzle
                # #####################################################
                # # Old stuff down here

                solution = []
                for j in range(len(new_puzzle[0])):
                    line = []
                    for i in range(len(new_puzzle[0])):
                        if len(pc[j][i]) == 1:
                            line.append(pc[j][i][0])
                        else:
                            solved = False
                            break
                    solution.append(line)
                #####################################################
                if solved and valid(solution, logic):
                    # print("Problem solved through brute forcing")
                    # puzzle_printer(solution, logic)
                    return pc
                else:
                    continue
    print("Not able to solve by brute forcing, you might have to figure it out mate.")
    return new_p


def sol_print(p):
    for line in p:
        print(line)


def consistent_past_values(puzzle, logic):
    p = empty_array_returner(len(puzzle[0]), len(puzzle[0]), "range")
    p_prev = None
    t = 0

    while p != p_prev and t < 15:
        p_prev = p
        p = possible_values(puzzle, logic, p_prev)
        t += 1

    return p, t


def solve(puzzle, logic, brute_force_if_needed=False):
    print("Finding a solution to the puzzle:")
    puzzle_printer(puzzle, logic)
    p, t = consistent_past_values(puzzle, logic)
    solved = True
    solution = []
    for j in range(len(puzzle[0])):
        line = []
        for i in range(len(puzzle[0])):
            if len(p[j][i]) == 1:
                line.append(p[j][i][0])
            else:
                solved = False
                break
        solution.append(line)

    if solved and valid(solution, logic):
        # print("Problem solved after {} iterations!".format(t))
        # puzzle_printer(solution, logic)
        return p, t, solved
    else:
        print("Problem not solved after {} iterations, possible values:".format(t))
        sol_print(p)
        if brute_force_if_needed:
            print("Brute Forcing")
            p = brute_force(puzzle, logic, p)
        else:
            brute_q = input("Do you want to brute force a solution? (y/n)")
            if brute_q == "y":
                p = brute_force(puzzle, logic, p)
        return p, t, solved


# p1 = FutoshikiPuzzle(puzzle1, logic1)
# # print(p1)
# p1.solve()
# print(p1.solution)
# for x in p1.possible_values:
#     print(x)
# solve(puzzle1, logic1)
puzzle11 = [
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0]
]

# '/\': u"\u2227"
# '\/': u"\u2228"

logic11 = [
    ["", "", "", ""],
    ["", "", "", "", ""],
    ["", "", "", ""],
    ["", "", "", "", ""],
    ["", "", ">", ""],
    ["", "", "", u"\u2228", ""],
    ["", ">", "", ""],
    [u"\u2227", "", "", "", ""],
    ["", ">", ">", ""],
    ["", "", "", "", ""]
]

def test():
    p = FutoshikiPuzzle(puzzle11, logic11)
    p.solve(output_log())

def solve_futoshiki(puzzle, logic):
    # 确保 puzzle 和 logic 不为 None
    if puzzle is None:
        output_str = "error"
        solution = "出现错误，跳过此条训练数据"
        print(output_str)
        return output_str, solution
        raise ValueError("puzzle cannot be None")
    if logic is None:
        output_str = "error"
        solution = "出现错误，跳过此条训练数据"
        print(output_str)
        return output_str, solution
        raise ValueError("logic cannot be None")
    

    # 检查 puzzle 和 logic 的类型，确保它们符合预期
    if not isinstance(puzzle, list) or not isinstance(logic, list):
        raise TypeError("puzzle and logic must be of type list")
    steps = output_log()  # 创建日志对象
    p = FutoshikiPuzzle(initial_puzzle_numbers=puzzle, puzzle_logic=logic, alog=steps,gen_mode=False)
    solution = p.solve()
    output_str = steps.get_log()
    print(output_str)
    return output_str, solution



import re
from collections import defaultdict
from itertools import permutations

def validate_calcudoku_puzzle(puzzle_spec, solution):
    """
    Given:
    puzzle_spec: list of strings, each string has n group labels or clues separated by whitespace.
    E.g., "J+12 F/4 5 2 E*3 E".
    solution: 2D list of ints (n×n), e.g. [[6,4,5,2,1,3], [5,1,4,3,2,6], ...]
    Returns: (bool, str)
    - bool indicating whether the solution is fully valid or not.
    - str is "" if valid, or an explanation of the first error found.

    Steps to validate:
    1) Parse puzzle spec to find:
        - The puzzle size n
        - group_label -> list of cells
        - group_label -> (op, target), if any

    2) Check solution dimension matches puzzle dimension.

    3) Check each row, each column for distinct 1..n.

    4) For each group, check if the group's constraint is satisfied:
        op "+" => sum of group cells == target
        op "*" => product of group cells == target
        op "-" => for some permutation p of group cells, p[0] - p[1] - ... = target
        op "/" => for some permutation p of group cells, p[0] / p[1] / ... = target (within an epsilon for float)
    """

    # ---------------------------------------------------------------------------
    # Step 1: parse puzzle spec
    # ---------------------------------------------------------------------------
    n = len(puzzle_spec)
    # basic check: each row in puzzle_spec should have n tokens
    puzzle_grid = []
    for r, line in enumerate(puzzle_spec):
        tokens = line.strip().split()
        if len(tokens) != n:
            return False, f"Row {r} of puzzle_spec has {len(tokens)} cells, expected {n}"
        puzzle_grid.append(tokens)

    # group_label -> dict with {"cells": [...], "op": None or str, "target": None or int}
    groups = defaultdict(lambda: {"cells": [], "op": None, "target": None})

    # We'll read each cell's label or clue. 
    # If it includes operation+target (like "J+12"), we store that in the group info.
    for r in range(n):
        for c in range(n):
            cell_str = puzzle_grid[r][c]
            # We assume the puzzle format: 
            #   either "X" (just a label) 
            #   or "X+12", "X-3", "X*60", "X/4", etc.
            #   or possibly "XY+12" if label has multiple letters
            # We'll parse label, operation, and target.

            match = re.match(r"^([A-Za-z]+)([+\-\*/])(\d+)$", cell_str)
            if match:
                g_label = match.group(1)
                op = match.group(2)
                tgt = int(match.group(3))
                groups[g_label]["op"] = op
                groups[g_label]["target"] = tgt
                groups[g_label]["cells"].append((r, c))
            else:
                # just a label
                g_label = cell_str
                groups[g_label]["cells"].append((r, c))

    # gather group data in a simpler structure
    group_list = []
    for g_label, info in groups.items():
        op = info["op"]
        target = info["target"]
        cells = info["cells"]
        group_list.append((g_label, op, target, cells))

    # ---------------------------------------------------------------------------
    # 2) Check solution dimension
    # ---------------------------------------------------------------------------
    if len(solution) != n:
        return False, f"Solution row count {len(solution)} != puzzle dimension {n}"
    for r in range(n):
        if len(solution[r]) != n:
            return False, f"Solution row {r} has {len(solution[r])} cols, expected {n}"

    # ---------------------------------------------------------------------------
    # 3) Check row & column distinctness
    # Each row/col must contain numbers 1..n exactly once
    # ---------------------------------------------------------------------------
    # Check row distinctness
    for r in range(n):
        row_vals = solution[r]
        if len(set(row_vals)) != n:
            return False, f"Row {r} has repeated values: {row_vals}"
        # optionally check they are exactly 1..n
        for val in row_vals:
            if val < 1 or val > n:
                return False, f"Row {r} has out-of-range value {val}"

    # Check column distinctness
    for c in range(n):
        col_vals = [solution[r][c] for r in range(n)]
        if len(set(col_vals)) != n:
            return False, f"Column {c} has repeated values: {col_vals}"
        for val in col_vals:
            if val < 1 or val > n:
                return False, f"Column {c} has out-of-range value {val}"

    # ---------------------------------------------------------------------------
    # 4) Check group operation constraints
    # ---------------------------------------------------------------------------
    for g_label, op, target, cells in group_list:
        # Gather the solution values for these cells
        vals = [solution[r][c] for (r, c) in cells]

        # If the group has no op/target (like single-cell group sometimes?), 
        # then it's typically the puzzle format that it does have an op & target 
        # on at least one cell. For a single-cell group, one possibility is "A+5" or similar.
        # If we truly have no op, we'll just skip – or treat it as passing automatically.
        if op is None or target is None:
            # If the group is single cell, we can interpret that as the target = that cell's value with op = '+'
            # or we can just skip. Here, let's skip. Or we can check that if group has 1 cell, the value is that cell.
            if len(cells) == 1:
                # It's presumably correct. 
                # If you need a stricter logic, uncomment:
                # if vals[0] != target: return False, f"Single-cell group {g_label} mismatch"
                pass
            continue

        if op == '+':
            if sum(vals) != target:
                return False, f"Group {g_label} sum {sum(vals)} != target {target}"
        elif op == '*':
            prod = 1
            for v in vals:
                prod *= v
            if prod != target:
                return False, f"Group {g_label} product {prod} != target {target}"
        elif op == '-':
            # We look for any permutation p of vals s.t. p[0] - p[1] - ... = target
            found = False
            for perm in permutations(vals):
                result = perm[0]
                for x in perm[1:]:
                    result -= x
                if result == target:
                    found = True
                    break
            if not found:
                return False, f"Group {g_label} cannot satisfy difference = {target} with values {vals}"
        elif op == '/':
            # We look for any permutation p of vals s.t. p[0] / p[1] / ... = target
            found = False
            for perm in permutations(vals):
                result = float(perm[0])
                ok = True
                for x in perm[1:]:
                    # check dividing by zero not possible here if x>0
                    # but just in case
                    if x == 0:
                        ok = False
                        break
                    result /= float(x)
                # check if close
                if ok and abs(result - target) < 1e-9:
                    found = True
                    break
            if not found:
                return False, f"Group {g_label} cannot satisfy ratio = {target} with values {vals}"
        else:
            return False, f"Group {g_label} has unsupported op {op}"

    # If we reach here, everything passed
    return True, ""

if __name__ == "__main__":


    # Example puzzle (the same puzzle_spec might be from a generator).
    puzzle_spec_example = [
        "P+4 O+1 T+5 G-2 E+3 R+5",
        "A-2 Q/3 Q G F+5 R",
        "A A H+2 N+8 N K+4",
        "L-1 B*48 B N C+6 S+1",
        "L J+10 B I*10 I I",
        "J J D+8 D D M+6"
    ]
    # Suppose n=4 here. A= group 1, B= group 2, etc.
    # This puzzle_spec is purely an illustrative example, likely incomplete as a real puzzle.

    # And an example solution that we want to check:
    solution_example = [
        [4, 1, 5, 6, 3, 2],
        [1 ,2, 6, 4, 5, 3],
        [3, 6, 2, 5, 1, 4],
        [5, 4, 3, 2, 6, 1],
        [6, 3, 4, 1, 2, 5],
        [2, 5, 1, 3, 4, 6],
    ]

    is_valid, reason = validate_calcudoku_puzzle(puzzle_spec_example, solution_example)
    if is_valid:
        print("Solution is valid!")
    else:
        print("Solution is invalid:", reason)
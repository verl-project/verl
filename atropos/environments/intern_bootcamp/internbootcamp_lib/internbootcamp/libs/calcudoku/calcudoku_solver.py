#!/usr/bin/env python3

"""
Calcudoku (a variant sometimes known under the trademark KenKen) solver.

Given a puzzle "spec" (a grid of strings, one row per line, each cell
describing its group and/or an initial digit), this code will parse
the puzzle, then solve it via backtracking (DFS).

Usage example (puzzle spec is the example puzzle from the prompt):

    puzzle_spec = [
        "J+12    F/4     5       2       E*3     E",
        "J       F       H+7     H       I*60    I",
        "J       D+12    G+28    G       G       I",
        "3       D       K-1     K       G       G",
        "4       D       D       B+11    G       A*2",
        "C+10    C       C       B       4       A"
    ]

    solver = CalcudokuSolver(puzzle_spec)
    solver.solve()
    solver.print_solution()
"""

import re
from collections import defaultdict

class CalcudokuSolver:
    def __init__(self, puzzle_spec):
        """
        Initialize solver with a puzzle specification.

        puzzle_spec is a list of strings, each representing a row.
        Each row has N cells (for N x N puzzle), separated by spaces.

        Example (6x6):
            [
              "J+12    F/4     5       2       E*3     E",
              "J       F       H+7     H       I*60    I",
              ...
            ]
        """

        # Store puzzle size by length of the puzzle_spec
        self.size = len(puzzle_spec)

        # We'll parse each cell to figure out:
        #   1) which group it belongs to (group label)
        #   2) if that cell includes an operation + target
        #   3) any initial digit given for that cell
        # Then we'll unify group information (operation, target, set of cells).
        self.groups = {}   # group_label -> {"op": "+/-/*//", "target": int, "cells": [(r,c), ...]}
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        # Each cell in self.grid will hold an integer once solved (1..size), or 0 if not assigned.

        # We'll also store partial group definitions if a cell has e.g. J+12
        # Later, all cells sharing group "J" must share the same op/target.
        # partial_groups[group_label] => {'op': str, 'target': int}
        self.partial_groups = defaultdict(lambda: {"op": None, "target": None})

        # Parse the puzzle spec line by line
        for r, line in enumerate(puzzle_spec):
            # Split on whitespace, respecting that each row has exactly self.size cells
            cells = line.strip().split()
            if len(cells) != self.size:
                raise ValueError(f"Row {r} does not have {self.size} cells: {cells}")

            for c, cell_str in enumerate(cells):
                # cell_str might look like "J+12", "F/4", "5", "2", "E*3", "E", or e.g. "3"
                # We can have:
                #   - a leading digit that is the initial value (e.g. "5")
                #   - a group label + optional operation + optional target (e.g. "J+12", "F/4", "E*3", or just "J")
                # Sometimes they are combined (e.g. "3" alone might be *just* an initial digit, or it's an initial digit
                # that also happens to be a group label if it doesn't match a group pattern. 
                # We'll handle the standard patterns using a regex.
                
                # Attempt to parse out the initial digit
                # We'll consider the possibility that there's an integer alone or at the start,
                # vs. a group label plus operation plus target.
                # A robust approach is to use a regex that looks for patterns like:
                #   group_label (one or more letters)
                #   optional: operation symbol [+|\-|*|/]
                #   optional: digits for target
                #   or just a single integer

                # We'll store an integer 0 if cell is not pre-filled.

                initial_digit = 0
                group_label = None
                op = None
                target = None

                # Check if the entire cell is just an integer
                if re.fullmatch(r"\d+", cell_str):
                    # It's purely an integer => initial digit
                    initial_digit = int(cell_str)
                else:
                    # Look for an integer at the start
                    # e.g. "3" might appear, or "4" might appear, or "5"
                    match_int_first = re.match(r"^(\d+)$", cell_str)
                    if match_int_first:
                        # purely digit
                        initial_digit = int(match_int_first.group(1))
                    else:
                        # Not purely digit, so let's see if there's an embedded digit or group + operation
                        # We'll attempt a more general pattern
                        # group label can be letters (one or more)
                        # optionally, there's an operation symbol and a target
                        # or the cell might be just letters (i.e. no operation or target).
                        match_group = re.match(r"^([A-Za-z]+)([+\-\*/])?(\d+)?$", cell_str)
                        if match_group:
                            g_label = match_group.group(1)
                            g_op = match_group.group(2)
                            g_target = match_group.group(3)
                            group_label = g_label
                            if g_op:
                                op = g_op
                            if g_target:
                                target = int(g_target)
                        else:
                            # Another possibility is that there's a leading digit plus a group pattern, e.g. "5J+12"
                            # But the example puzzle doesn't seem to do that. If that were needed, we'd parse accordingly.
                            # We'll just handle the example puzzle format.
                            # If we get here, it might be a single digit or something else. Let's see if it's purely digit:
                            # We already covered that. So we'll raise an error if it doesn't match expected format.
                            # But let's check if there's a leading digit plus group pattern.
                            match_leading_digit = re.match(r"^(\d+)([A-Za-z]+)([+\-\*/])?(\d+)?$", cell_str)
                            if match_leading_digit:
                                d_val = match_leading_digit.group(1)
                                g_label = match_leading_digit.group(2)
                                g_op = match_leading_digit.group(3)
                                g_target = match_leading_digit.group(4)

                                initial_digit = int(d_val)
                                group_label = g_label
                                if g_op:
                                    op = g_op
                                if g_target:
                                    target = int(g_target)
                            else:
                                # If none of these patterns match, let's see if it's purely a digit followed by nothing
                                # or if there's a stray trailing digit...
                                # We'll just raise an error because we don't expect that format in the example puzzle.
                                raise ValueError(f"Cell '{cell_str}' doesn't match known puzzle spec formats.")

                # If it's still possible there's a group label+op+target in the same cell as a digit,
                # we'd handle that above. In the example puzzle, the given digits (mostly "3" or "4" etc.)
                # appear in separate cells or after an operation. We have covered those patterns.

                # If we haven't got group_label/op/target from the above, maybe there's another pattern:
                # purely a digit, or purely a group label. Let's see if there's a leftover.
                if group_label is None:
                    # Maybe it's just letters
                    match_letters = re.match(r"^[A-Za-z]+$", cell_str)
                    if match_letters:
                        group_label = cell_str
                    else:
                        # If not, we already set initial_digit if it was purely a digit.
                        # so possibly we are done.
                        pass

                # Store initial_digit in self.grid
                self.grid[r][c] = initial_digit

                if group_label:  # We have a group for this cell
                    # Add this cell to that group in groups dictionary
                    if group_label not in self.groups:
                        self.groups[group_label] = {
                            "op": None,
                            "target": None,
                            "cells": []
                        }
                    self.groups[group_label]["cells"].append((r, c))

                    # If this cell indicated an operation & target, note that
                    if op and target:
                        # If the group was never assigned an operation or target, do so
                        if self.groups[group_label]["op"] is None and self.groups[group_label]["target"] is None:
                            self.groups[group_label]["op"] = op
                            self.groups[group_label]["target"] = target
                        else:
                            # If there's already a mismatch, that means puzzle spec is conflicting
                            if (self.groups[group_label]["op"] != op or
                                self.groups[group_label]["target"] != target):
                                raise ValueError(
                                    f"Conflict in group {group_label}: multiple ops/targets found."
                                )
                    # It's also possible the group operation+target is found in another cell of the same group
                    # so we won't do anything if this cell has no op/target. We'll unify them at the end.
                # else: the cell has no group label => For Calcudoku, typically every cell belongs to a group.
                # Possibly the puzzle spec is incomplete or the puzzle is malformed. We'll ignore that for now.

        # Now we unify groups (some might have zero/None op/target because it was specified in a different cell).
        # It's typical that only one cell of each group has the operation/target. Or in the case of a group of size 1,
        # it might be labeled, e.g. "K-1" or something. We assume each group must have exactly one op/target overall.
        # Some puzzle formats only show that op/target in one cell. We'll assume each group is valid for the example.
        for g_label, info in self.groups.items():
            if info["op"] is None or info["target"] is None:
                # In a well-formed puzzle, presumably one of the group's cells had the clue "g_label op target"
                # If truly missing, that might be an error or might be a group of size 1 with no operation (?), 
                # but that wouldn't be standard Calcudoku. We'll raise an error if missing.
                raise ValueError(f"Group '{g_label}' is missing an op or a target: {info}")

        # Data structures to help with row/column constraints while solving:
        # We'll keep track of used_in_row[r][val] = True if val is used in row r
        # We'll keep track of used_in_col[c][val] = True if val is used in col c
        self.used_in_row = [[False]*(self.size+1) for _ in range(self.size)]
        self.used_in_col = [[False]*(self.size+1) for _ in range(self.size)]

        # Fill these in with any initial digits
        for r in range(self.size):
            for c in range(self.size):
                val = self.grid[r][c]
                if val != 0:
                    if self.used_in_row[r][val]:
                        raise ValueError(f"Duplicate initial value {val} in row {r}")
                    if self.used_in_col[c][val]:
                        raise ValueError(f"Duplicate initial value {val} in col {c}")
                    self.used_in_row[r][val] = True
                    self.used_in_col[c][val] = True

        # We'll convert self.groups to a list for convenience:
        # group_list = [(group_label, op, target, [(r,c), ...])]
        self.group_list = []
        for g_label, info in self.groups.items():
            self.group_list.append((g_label, info["op"], info["target"], info["cells"]))

        # Sort group_list by size or something if we want (optional).
        # We'll just keep it as is. The solver doesn't particularly require an order.

        self.solved = False
        self.history = []


    def solve(self):
        """
        Solve the puzzle via backtracking DFS.
        """
        # print(self.grid)
        # print(self.used_in_row)
        # print(self.group_list)
        self.dfs_solve(0, 0)
        if not self.solved:
            return False, None
        else:
            return self.grid, self.history


    def dfs_solve(self, row, col):
        """
        Recursive backtracking approach.
        row, col: current cell to fill.
        """
        if row == self.size:
            # We've assigned all rows successfully
            self.history.append(f"Assign all rows successfully. Find the solution!")
            self.solved = True
            return True

        # Compute next position to go after we fill this cell
        next_col = col + 1
        next_row = row
        if next_col == self.size:
            next_row = row + 1
            next_col = 0
        
        self.history.append(f"Select position ({col},{row}).")
        
        # If cell is already assigned (from initial clues), skip
        if self.grid[row][col] != 0:
            self.history[-1] = self.history[-1] + f"This position already has number {self.grid[row][col]}. Skip this position."
            return self.dfs_solve(next_row, next_col)

        self.history[-1] = self.history[-1] + f"This position is empty now."
        # Try each candidate from 1..size
        for val in range(1, self.size+1):
            # Check if it's valid to place val here
            self.history.append(f"Select position ({col},{row}). Try to fill it with number {val}.")
            if self.used_in_row[row][val] or self.used_in_col[col][val]:
                self.history[-1] = self.history[-1] + f"Number {val} is already used in the current row/column. Retry."
                continue
            
            self.history[-1] = self.history[-1] + f"Number {val} has not been used in the current row and column."
            # Place val
            self.grid[row][col] = val
            self.used_in_row[row][val] = True
            self.used_in_col[col][val] = True

            # Check if group constraint remains valid
            # We'll do a partial check if possible. For + and * there's partial pruning:
            # For - and / there's not straightforward partial pruning, so we only check if the group is fully assigned.
            self.history.append(f"Check if position ({col},{row}) with number {val} keep group constraint valid.")
            is_group_valid, reason = self.group_is_valid(row, col)
            if is_group_valid:
                self.history[-1] = self.history[-1] + f"Position ({col},{row}) with number {val} keep group constraint valid."
                if self.dfs_solve(next_row, next_col):
                    return True
                else:
                    self.history.append(f"Backtrack to Position ({col},{row}).")
            else:
                self.history[-1] = self.history[-1] + f"Due to {reason}. Position ({col},{row}) with number {val} can't keep group constraint valid. Retry."
            # Revert
            self.grid[row][col] = 0
            self.used_in_row[row][val] = False
            self.used_in_col[col][val] = False
        
        self.history.append(f"All possible values at position ({col},{row}) have been tried and no value satisfies the condition. So clear the value of position ({col},{row}) and backtrack to the previous node and retry.")
        return False


    def group_is_valid(self, row, col):
        """
        Check if the group containing (row, col) is still valid with the current partial assignment.

        We'll look up which group (row,col) belongs to. In Calcudoku there's exactly one group per cell,
        so let's find that. We'll then check partial constraints:

        - If the group's operation is '+', we ensure that the sum of assigned cells does not exceed the target,
        and if all cells are assigned, that the sum equals the target.
        - If '*', we ensure that the product of assigned cells does not exceed the target,
        and if all cells are assigned, the product equals the target.
        - If '-' or '/', we only check if all cells in group are assigned; if yes, test if there's a permutation of
        those values that yields the target via difference or ratio. If not all assigned, we can't prune easily.

        Returns:
            (bool, str): A tuple of (is_valid, reason_if_invalid).
                        If is_valid == False, reason_if_invalid is a string describing why.
                        If is_valid == True, reason_if_invalid will be None.
        """

        # Find which group this cell belongs to
        group_label = None
        for g_label, op, target, cells in self.group_list:
            if (row, col) in cells:
                group_label = g_label
                break

        if group_label is None:
            # No group? That would be odd. We'll just say it's valid and provide no reason.
            # (In well-formed puzzles, every cell belongs to exactly one group.)
            return True, None

        # Grab group details
        op, target, group_cells = None, None, None
        for g_label, g_op, g_target, g_cells in self.group_list:
            if g_label == group_label:
                op, target, group_cells = g_op, g_target, g_cells
                break

        # Gather assigned values in this group
        assigned_values = []
        unassigned_count = 0
        for (r, c) in group_cells:
            val = self.grid[r][c]
            if val == 0:
                unassigned_count += 1
            else:
                assigned_values.append(val)

        # --------------------------------------------------------
        # Partial checks for sum or product groups (if not fully assigned)
        # --------------------------------------------------------
        if unassigned_count > 0:
            if op == '+':
                current_sum = sum(assigned_values)
                if current_sum > target:
                    return False, f"Partial sum {current_sum} exceeds target {target}"
            elif op == '*':
                prod = 1
                for v in assigned_values:
                    prod *= v
                if prod > target:
                    return False, f"Partial product {prod} exceeds target {target}"
            # For '-' or '/', partial checks are trickier, so no pruning here.
            return True, None

        # --------------------------------------------------------
        # All cells in this group have been assigned, so do a final check.
        # --------------------------------------------------------
        if op == '+':
            if sum(assigned_values) != target:
                return False, f"Sum of group {sum(assigned_values)} != target {target}"
                    
                
            return True, None

        elif op == '*':
            prod = 1
            for v in assigned_values:
                prod *= v
            if prod != target:
                return False, f"Product of group {prod} != target {target}"
                    
                
            return True, None

        elif op == '-':
            # Check if there's a permutation of assigned_values such that
            # one value minus the others (in sequence) == target
            from itertools import permutations
            for perm in permutations(assigned_values):
                result = perm[0]
                for v in perm[1:]:
                    result -= v
                if result == target:
                    return True, None
            return False, f"No permutation of values {assigned_values} produces difference {target}"
                
            

        elif op == '/':
            # Check if there's a permutation of assigned_values such that
            # one value divided by the others (in sequence) == target
            from itertools import permutations
            for perm in permutations(assigned_values):
                result = float(perm[0])
                for v in perm[1:]:
                    result /= float(v)  # safe, since puzzle values won't be 0
                if abs(result - target) < 1e-9:
                    return True, None
            return False,  f"No permutation of values {assigned_values} produces ratio {target}"
                
            

        else:
            # Unknown operation
            return False, f"Unknown operation '{op}'"


    def print_solution(self):
        """
        Print the solved grid (if solved).
        """
        if not self.solved:
            print("No solution found.")
            return

        print("Solution:")
        for r in range(self.size):
            row_str = " ".join(str(self.grid[r][c]) for c in range(self.size))
            print(row_str)
        #print("\n".join(self.history))

def example_usage():
    """
    Example usage with the puzzle from the prompt.

    The puzzle spec is the one given in the prompt. 
    (This is the 6x6 puzzle with solution.)
    """
    puzzle_spec = [
        "J+12    F/4     5       2       E*3     E",
        "J       F       H+7     H       I*60    I",
        "J       D+12    G+28    G       G       I",
        "3       D       K-1     K       G       G",
        "4       D       D       B+11    G       A*2",
        "C+10    C       C       B       Z+4       A"
    ]
    
    # puzzle_spec = [
    #     "A-1 B+6 C-2 C",
    #     "A B D+5 D",
    #     "E+7 B D F",
    #     "E E F F+9",
    # ]
    
    puzzle_spec = [ "P+4 O+1 T+5 G-2 E+3 R+5",
        "A-2 Q/3 Q G F+5 R",
        "A A H+2 N+8 N K+4",
        "L-1 B*48 B N C+6 S+1",
        "L J+10 B I*10 I I",
        "J J D+8 D D M+6",
    ]


    solver = CalcudokuSolver(puzzle_spec)
    solution, history = solver.solve()
    if solution:
        solver.print_solution()
    else:
        print("No solution found.")

if __name__ == "__main__":
    # Uncomment the following line to run the example puzzle:
    example_usage()

    # Or you can import this module elsewhere and call CalcudokuSolver with your own puzzle spec.
    #pass
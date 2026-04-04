

def validate_campsite_solution(puzzle, row_constraints, col_constraints, solution):
    """
    Checks if 'solution' is a valid solution to the given Campsite puzzle.

    :param puzzle:       2D list (n×m), each cell is 'T' (tree) or 'X' (empty).
    :param row_constraints: list of length n, how many tents should be in each row.
    :param col_constraints: list of length m, how many tents should be in each column.
    :param solution:     2D list (n×m), each cell is 'T', 'X', or 'C' (tent).
    :return: (is_valid, message)
            is_valid (bool) - True if the solution is valid, False otherwise
            message (str)   - Explanation if invalid, or "Valid solution" otherwise.
    """
    n = len(puzzle)
    m = len(puzzle[0])

    # 1) Check the dimensions match
    if len(solution) != n or any(len(row) != m for row in solution):
        return False, "Solution dimension mismatch."

    # 2) Puzzle 'T' must remain 'T' in the solution
    for r in range(n):
        for c in range(m):
            if puzzle[r][c] == 'T' and solution[r][c] != 'T':
                return False, f"Solution changed puzzle tree at row={r}, col={c}."

            # Optionally, disallow new trees in puzzle 'X': 
            # If puzzle[r][c] == 'X' and solution[r][c] == 'T',
            #     you can decide how strict you want to be.
            #
            # Usually, in a Campsite puzzle, the trees are fixed,
            # so we typically disallow adding new 'T' where puzzle was 'X'.
            if puzzle[r][c] == 'X' and solution[r][c] == 'T':
                return False, f"Solution introduced a new tree at row={r}, col={c}."

    # 3) Check row/column counts of tents
    row_counts = [0] * n
    col_counts = [0] * m
    for r in range(n):
        for c in range(m):
            if solution[r][c] == 'C':
                row_counts[r] += 1
                col_counts[c] += 1

    for r in range(n):
        if row_counts[r] != row_constraints[r]:
            return False, f"Row {r} has {row_counts[r]} tents, expected {row_constraints[r]}."

    for c in range(m):
        if col_counts[c] != col_constraints[c]:
            return False, f"Column {c} has {col_counts[c]} tents, expected {col_constraints[c]}."

    # 4) Validate adjacency rules for tents
    #    (a) Each tent is orthogonally adjacent to at least one tree.
    #    (b) No two tents are adjacent in any of the 8 directions.
    directions_orth = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    directions_diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    directions_all = directions_orth + directions_diag

    for r in range(n):
        for c in range(m):
            if solution[r][c] == 'C':
                # (a) Ensure there's a tree in one of the orth. neighbors
                has_tree_neighbor = False
                for dr, dc in directions_orth:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < m and solution[rr][cc] == 'T':
                        has_tree_neighbor = True
                        break
                if not has_tree_neighbor:
                    return False, f"Row={r}, Col={c} tent not adjacent to any tree."

                # (b) Ensure no tent in any of the 8 directions
                for dr, dc in directions_all:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < m:
                        if solution[rr][cc] == 'C':
                            return False, f"Tents at ({r},{c}) and ({rr},{cc}) are adjacent."

    # If all checks pass
    return True, "Valid solution."

if __name__ == "__main__":

    puzzle = [
    ['X','X','X','T'],
    ['T','X','X','X'],
    ['X','X','T','X']
    ]
    row_constraints = [1, 0, 2]
    col_constraints = [1, 1, 0, 1]

    solution = [
    ['C','X','X','T'], # A tent at (0,0)
    ['T','X','X','X'],
    ['X','C','T','C'] # A tent at (2,3)
    ]

    is_valid, msg = validate_campsite_solution(puzzle, row_constraints, col_constraints, solution)
    print(is_valid, msg)
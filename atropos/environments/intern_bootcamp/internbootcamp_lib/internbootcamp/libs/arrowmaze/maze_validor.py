import re

def parse_candidate_path(answer_str):
    """
    Parse a candidate_path string in the format:
       "[[1 0 0,0 0 0,0 0 2]]"
    into a 2D list of integers.
    """
    # 1) Remove the outer brackets "[[" and "]]"
    # 2) Split into rows by comma
    # 3) Each row is space-separated integers
    # Example input: "[[1 0 0,0 0 0,0 0 2]]"
    
    # Trim leading/trailing brackets
    trimmed = answer_str.strip()
    if trimmed.startswith("[["):
        trimmed = trimmed[2:]
    if trimmed.endswith("]]"):
        trimmed = trimmed[:-2]
    
    # Now we have something like: "1 0 0,0 0 0,0 0 2"
    # Split by commas to get rows
    row_strs = trimmed.split(",")
    
    grid_of_ints = []
    for row_str in row_strs:
        # row_str might look like "1 0 0" or "0 0 2"
        row_str = row_str.strip()
        if not row_str:
            continue
        # split by spaces
        vals = row_str.split()
        row_ints = [int(v) for v in vals]
        grid_of_ints.append(row_ints)
    return grid_of_ints

def arrow_maze_validator(
    grid, start_position, answer
):
    """
    Validate whether a candidate_path in puzzle's format (e.g. "[[1 0 0,0 0 0,0 0 2]]")
    is a correct solution to the arrow maze.

    Parameters
    ----------
    grid : list[list[str]]
        A 2D grid of arrow symbols or '○'.
        Example:
         [
           ['→', '↙', '↓'],
           ['↖', '↓', '↙'],
           ['↑', '←', '○'],
         ]
    start_position : (int, int)
        (row, col) of the starting cell.
    answer : list
        The proposed solution in the format "[[...]]"
        0 => not on path
        1 => first visited cell
        2 => second visited cell
        etc.

    Returns
    -------
    bool
        True if the path is valid, False otherwise.
    """

    # Directions dictionary: maps arrow symbol -> (dr, dc)
    DIRECTIONS = {
        '↑':  (-1,  0),
        '↓':  ( 1,  0),
        '←':  ( 0, -1),
        '→':  ( 0,  1),
        '↖':  (-1, -1),
        '↗':  (-1,  1),
        '↙':  ( 1, -1),
        '↘':  ( 1,  1),
    }

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    
    #$candidate_grid = parse_candidate_path(answer_str)
    candidate_grid = answer

    # Sanity check: the candidate_grid should match the same dimensions as 'grid'
    if len(candidate_grid) != rows:
        return False
    for row_vals in candidate_grid:
        if len(row_vals) != cols:
            return False
    
    # 2. Extract the labeled cells: (label, (row, col))
    #    We only care about label > 0
    labeled_cells = []
    for r in range(rows):
        for c in range(cols):
            label = candidate_grid[r][c]
            if label > 0:
                labeled_cells.append((label, (r, c)))
    
    # If no labeled cells, invalid
    if not labeled_cells:
        return False
    
    # 3. Sort by label ascending
    labeled_cells.sort(key=lambda x: x[0])  # sort by label number
    # This gives us an ordered path: [ (1, (r1,c1)), (2, (r2,c2)), ... ]

    # 4. The path in terms of coordinates:
    path = [cell_coord for _, cell_coord in labeled_cells]

    # 5. Check that label "1" is at start_position
    if path[0] != start_position:
        return False

    # 6. Validate each consecutive step in path
    for i in range(len(path) - 1):
        (r1, c1) = path[i]
        (r2, c2) = path[i + 1]

        if not in_bounds(r1, c1) or not in_bounds(r2, c2):
            return False

        # If the current cell is the end symbol '○' but we still have more steps, invalid
        if grid[r1][c1] == '○':
            return False

        # Arrow in the current cell:
        arrow_symbol = grid[r1][c1]
        if arrow_symbol not in DIRECTIONS:
            return False  # not an arrow and not the end symbol

        (dr, dc) = DIRECTIONS[arrow_symbol]
        delta_r = r2 - r1
        delta_c = c2 - c1

        # Must move in a positive integer multiple of (dr, dc).
        if dr == 0 and dc == 0:
            return False  # shouldn't happen with valid arrows

        # Horizontal or vertical
        if dr == 0:
            # vertical movement is zero => must move horizontally
            # check we didn't move in row, must move in col
            if delta_r != 0:
                return False
            # direction must match sign of dc
            if dc > 0 and delta_c <= 0:
                return False
            if dc < 0 and delta_c >= 0:
                return False
        elif dc == 0:
            # horizontal movement is zero => must move in row
            if delta_c != 0:
                return False
            if dr > 0 and delta_r <= 0:
                return False
            if dr < 0 and delta_r >= 0:
                return False
        else:
            # diagonal
            if delta_r == 0 or delta_c == 0:
                return False  # can't be diagonal if one is zero
            if (dr > 0 and delta_r <= 0) or (dr < 0 and delta_r >= 0):
                return False
            if (dc > 0 and delta_c <= 0) or (dc < 0 and delta_c >= 0):
                return False
            # check integer multiples
            if (delta_r % dr) != 0 or (delta_c % dc) != 0:
                return False
            factor_r = delta_r // dr
            factor_c = delta_c // dc
            if factor_r != factor_c or factor_r <= 0:
                return False

    # 7. Check last labeled cell is the '○' cell
    last_r, last_c = path[-1]
    if not in_bounds(last_r, last_c):
        return False
    if grid[last_r][last_c] != '○':
        return False

    # If all checks pass, it's a valid solution
    return True


# ------------------------------
# Example usage:
if __name__ == "__main__":
    # A small 3×3 arrow maze with start=(0, 0):
    # Grid (3x3):
    #   →   ↙   ↓
    #   ↖   ↓   ↙
    #   ↑   ←   ○
    grid = [
        ['→', '↙', '↓'],
        ['↖', '↓', '↙'],
        ['↑', '←', '○'],
    ]
    start_position = (0, 0)

    # Example candidate_path string:
    # "[[1 0 0,0 0 0,0 0 2]]"
    # This claims:
    #   row=0 col=0 => 1 (start)
    #   row=2 col=2 => 2 (end)
    candidate_path_str = "[[1 0 2,0 0 0,0 0 3]]"

    is_valid = arrow_maze_validator(
        grid, start_position, candidate_path_str
    )
    print("Is the candidate path valid?", is_valid)
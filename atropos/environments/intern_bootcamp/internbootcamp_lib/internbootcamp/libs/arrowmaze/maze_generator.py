import random

ARROW_SYMBOLS = {
    (-1,  0): '↑',
    ( 1,  0): '↓',
    ( 0, -1): '←',
    ( 0,  1): '→',
    (-1, -1): '↖',
    (-1,  1): '↗',
    ( 1, -1): '↙',
    ( 1,  1): '↘',
}

# Inverse mapping from symbol back to (dr, dc) if needed:
# REVERSE_ARROW = {v: k for k, v in ARROW_SYMBOLS.items()}

def generate_arrow_maze(n, m, start, end, max_attempts=10000, max_solution_step=20, seed=None):
    """
    Generate an n×m arrow maze with a guaranteed path from `start` to `end`.
    
    Parameters
    ----------
    n : int
        Number of rows.
    m : int
        Number of columns.
    start : (int, int)
        (row, col) of the start cell (0-based).
    end : (int, int)
        (row, col) of the end cell (0-based).
    max_attempts : int
        Maximum attempts to try building a path in case of random backtracking failures.
    
    Returns
    -------
    list[list[str]]
        A 2D grid of strings. Each cell is one of the 8 arrow symbols or '○' for the end cell.
        Guaranteed at least one path under the "move multiple steps" rule.
    
    Raises
    ------
    ValueError
        If a path cannot be generated within `max_attempts`.
    """
    if seed != None:
        random.seed(seed)

    # Basic checks
    if not (0 <= start[0] < n and 0 <= start[1] < m):
        raise ValueError("Start position out of bounds.")
    if not (0 <= end[0] < n and 0 <= end[1] < m):
        raise ValueError("End position out of bounds.")
    if start == end:
        raise ValueError("Start and end cannot be the same cell.")

    # We will first build a path (a list of cells) from start to end using random DFS-like backtracking.
    # Then we will fill the "arrows" along that path to ensure a valid path. 
    # Finally, fill in random arrows for the other cells.

    # For convenience, store the path of cells as a list of (row, col).
    path = []
    global now_step_number 
    
    # We will do a backtracking function. The function attempts to build a path from "current" cell to "end".
    # If it succeeds, it returns True and has the path in the `path`.
    # If it fails, it returns False.

    def in_bounds(r, c):
        return 0 <= r < n and 0 <= c < m

    def backtrack(current):
        """Attempt to build a path from `current` to `end` via random expansions."""
        # Add the current cell to path
        global now_step_number 
        now_step_number += 1
        path.append(current)

        # If current == end, we've made the path successfully
        if current == end:
            return True
        
        if now_step_number > max_solution_step:
            path.pop()
            now_step_number -= 1
            return False

        # Try random directions in a shuffled order
        directions = list(ARROW_SYMBOLS.keys())
        random.shuffle(directions)

        # For each direction, try steps of size 1.. up to max possible in that direction
        for (dr, dc) in directions:
            # The maximum step we can take in this direction so we don't go out of bounds
            max_step = 1
            while True:
                nr = current[0] + (max_step * dr)
                nc = current[1] + (max_step * dc)
                if not in_bounds(nr, nc):
                    break
                max_step += 1
            # Now max_step - 1 is the largest valid step

            if max_step <= 1:
                # We can't move in this direction at all
                continue

            # We can choose any step in range [1, max_step-1]
            step_sizes = list(range(1, max_step))
            random.shuffle(step_sizes)

            for step in step_sizes:
                nr = current[0] + step * dr
                nc = current[1] + step * dc
                # Check if the next cell is not yet in path (avoid immediate loops)
                if (nr, nc) not in path:
                    # Recurse
                    if backtrack((nr, nc)):
                        return True
                # else skip because it's already in path (avoid cycles)

        # If no direction/step led to a solution, backtrack
        path.pop()
        now_step_number -= 1
        return False

    # Try multiple times to build a path (sometimes random choices fail to find a path)
    attempts = 0
    success = False
    while attempts < max_attempts:
        path.clear()
        now_step_number = 0
        if backtrack(start):
            success = True
            break
        attempts += 1

    if not success:
        raise ValueError("Could not generate a path with random backtracking after many attempts.")

    # Now `path` is our sequence of cells from start to end. 
    # Next we build the grid of arrows. We'll mark all cells with random arrows first, then override the path.

    grid = [[None for _ in range(m)] for _ in range(n)]

    # Assign random arrows to every cell initially
    directions_list = list(ARROW_SYMBOLS.values())
    for r in range(n):
        for c in range(m):
            grid[r][c] = random.choice(directions_list)

    # Mark the end cell with '○'
    er, ec = end
    grid[er][ec] = '○'

    # Now override the path cells (except the end). 
    # If path[i] = (r1, c1) leads to path[i+1] = (r2, c2),
    # we find direction = (r2-r1, c2-c1) -> arrow symbol. 
    # We put that symbol in grid[r1][c1]. The last cell in the path is the end cell → '○'.

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        dr = r2 - r1
        dc = c2 - c1
        symbol = ARROW_SYMBOLS.get((dr, dc), None)
        if symbol is None:
            # This should never happen if (dr, dc) is one of the 8 directions.
            # But we might have multi-steps combined. We only store the "macro step" as if it were a single arrow.
            # Because in the puzzle, moving multiple steps in the same direction is allowed in one arrow cell.
            # So the direction is the *normalized* version, i.e., sign of dr/dc if non-zero.
            # For example, if dr=2, dc=-2 => direction is (1, -1) or (1, -1) repeated. 
            # Let's define a quick normalization.
            ndr = 0 if dr == 0 else (dr // abs(dr))
            ndc = 0 if dc == 0 else (dc // abs(dc))
            symbol = ARROW_SYMBOLS.get((ndr, ndc))
        grid[r1][c1] = symbol

    # print("standard path we select:",path)
    
    return grid


# ---------------------------
# Example usage / testing code

if __name__ == "__main__":
    # Example: generate a 6x8 maze, start at (0,0), end at (5,7).
    # You can change these freely.
    n, m = 10, 8
    start = (0, 0)
    end = (5, 7)

    # For reproducibility, remove or change for more randomness
    maze = generate_arrow_maze(n, m, start, end, max_solution_step=15, seed=0)

    # Print the generated maze
    for row in maze:
        print(" ".join(row))
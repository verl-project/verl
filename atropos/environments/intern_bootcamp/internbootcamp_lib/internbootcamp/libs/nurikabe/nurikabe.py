from collections import deque

def solve_nurikabe(grid, verbose=True):
    """
    Nurikabe puzzle solver.
    
    grid: 2D list, each element in {'A', 'X', '0'..'9'}
      - digit: island with size hint
      - 'X': island cell (without specific numeric hint)
      - 'A': sea cell
    verbose: bool, whether to print intermediate solving process.
    
    Returns: a 2D list (same format) representing the solved puzzle.
    
    Requirements:
      1) Exactly one digit (size hint) per island, and the island size == that digit.
      2) All sea cells are connected (single sea), and no 2x2 sea block.
      3) No island lacking a digit hint.
    """

    if verbose:
        print("=== Nurikabe Solver: Start ===")
        print("Puzzle Input:")
        for row in grid:
            print("  " + " ".join(row))
        print()

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Make a modifiable copy of the grid
    solution = [row[:] for row in grid]

    # Identify cells that can be changed (not digit)
    fill_cells = []
    for r in range(rows):
        for c in range(cols):
            if not solution[r][c].isdigit():
                fill_cells.append((r, c))

    if verbose:
        print(f"[Info] There are {len(fill_cells)} fillable cells.\n")

    # Helper functions
    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def is_island(val):
        return val == 'X' or val.isdigit()

    def get_island_region(r, c):
        """ BFS to get connected island region and its digit set """
        queue = deque([(r, c)])
        region = {(r, c)}
        digits = set()

        while queue:
            rr, cc = queue.popleft()
            val = solution[rr][cc]
            if val.isdigit():
                digits.add(val)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = rr + dr, cc + dc
                if in_bounds(nr, nc) and is_island(solution[nr][nc]) and (nr, nc) not in region:
                    region.add((nr, nc))
                    queue.append((nr, nc))
        return region, digits

    def check_islands():
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if is_island(solution[r][c]) and (r, c) not in visited:
                    region, digits = get_island_region(r, c)
                    visited |= region
                    if len(digits) != 1:
                        return False
                    digit_val = next(iter(digits))
                    if len(region) != int(digit_val):
                        return False
        return True

    def check_sea():
        sea_cells = [(r, c) for r in range(rows) for c in range(cols) if solution[r][c] == 'A']
        if not sea_cells:  # if puzzle requires at least some sea
            return False

        visited = set([sea_cells[0]])
        queue = deque([sea_cells[0]])
        while queue:
            rr, cc = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = rr + dr, cc + dc
                if in_bounds(nr, nc) and solution[nr][nc] == 'A' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        if len(visited) != len(sea_cells):  # sea not connected
            return False

        # no 2x2 all sea
        for rr in range(rows - 1):
            for cc in range(cols - 1):
                block = [solution[rr][cc],
                         solution[rr][cc+1],
                         solution[rr+1][cc],
                         solution[rr+1][cc+1]]
                if all(x == 'A' for x in block):
                    return False

        return True

    def is_valid():
        return check_islands() and check_sea()

    def backtrack(idx=0):
        if idx == len(fill_cells):
            # All fillable cells assigned, do final check
            if verbose:
                print("[*] All cells filled, validating solution...")
            if is_valid():
                if verbose:
                    print("[√] Valid solution found!\n")
                return True
            else:
                if verbose:
                    print("[X] Validation failed, backtrack.\n")
                return False

        r, c = fill_cells[idx]
        original_val = solution[r][c]

        # Try 'A' or 'X'
        for candidate in ['A', 'X']:
            solution[r][c] = candidate
            if verbose:
                print(f"[Try] idx={idx}, cell=({r},{c}), set='{candidate}'")
            
            if backtrack(idx + 1):
                return True

            # Undo assignment if not successful
            solution[r][c] = original_val
            if verbose:
                print(f"  -> Backtrack: reset cell=({r},{c}) to '{original_val}'")

        return False

    if verbose:
        print("=== Start Backtracking ===\n")
    success = backtrack(0)
    if not success:
        raise ValueError("\n[Conclusion] Puzzle has no valid solution under given constraints.")

    if verbose:
        print("=== Final Solved Puzzle ===")
        for row in solution:
            print(" ".join(row))

    return solution


# ============ Example test ============ #
if __name__ == "__main__":
    # puzzle = [
    #     ['X','X','X','X','5'],
    #     ['X','X','X','X','X'],
    #     ['1','X','X','4','X'],
    #     ['X','X','X','X','X'],
    #     ['X','X','X','X','1']
    # ]
    puzzle = [
        ['4','X','1','X','X'],
        ['X','X','X','X','3'],
        ['X','X','2','X','X'],
        ['X','X','X','X','X'],
        ['X','X','X','X','X']
    ]
    result = solve_nurikabe(puzzle, verbose=False)
    print("\n[Solved Output]:")
    for row in result:
        print(" ".join(row))
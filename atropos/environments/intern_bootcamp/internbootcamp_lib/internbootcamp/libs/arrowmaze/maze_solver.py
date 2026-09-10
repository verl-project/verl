from collections import deque

def solve_arrow_maze(grid, start=(0, 0)):
    """
    Solve the arrow maze puzzle where each cell has an arrow (↑, ↗, →, ↘, ↓, ↙, ←, ↖),
    or the '○' end symbol. From each cell, you can move 1 or more steps in that cell's
    arrow direction (if it's not the '○' cell).

    Parameters
    ----------
    grid : list[list[str]]
        2D list of strings. Each element is one of:
          '↑', '↗', '→', '↘', '↓', '↙', '←', '↖', or '○' (end).
    start : (int, int)
        (row, column) index of the starting cell.

    Returns
    -------
    str
        A string with the answer in the format:
          "[[r1, r2, r3, ...]]"
        Where each row is space-separated and rows are separated by commas.
        - The position of each cell on the path that is an inflection point
          (including start, any direction change, and the end) is labeled
          in ascending order of encountering the direction changes.
        - Cells not on the path are labeled '0'.
    """

    # Map from arrow symbol to row, col deltas
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
    if rows == 0 or cols == 0:
        raise ValueError("Grid must not be empty.")

    # Locate the end cell (just for validation); not strictly needed,
    # but good to confirm the puzzle has a valid end.
    end_cell = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '○':
                end_cell = (r, c)
                break
        if end_cell is not None:
            break
    if not end_cell:
        raise ValueError("No '○' (end) cell found in the grid.")

    # Check start in bounds
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError("Start position is out of grid bounds.")

    # BFS to find any path from start to end
    queue = deque()
    queue.append(start)
    visited = set()
    visited.add(start)

    history = []
    
    # Parent dict for reconstructing path: parent[(r, c)] = (pr, pc)
    parent = {}

    # Helper to check in-bounds
    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    found_end = False

    while queue and not found_end:
        
        if found_end == False:
            candidates_needed2explore = ""
            for item in queue:
                r,c = item
                candidates_needed2explore = candidates_needed2explore + f"({r},{c}),"
            history.append("The candidate positions need to explore are: "+candidates_needed2explore +"\n")
        
        r, c = queue.popleft()
        if found_end == False:
            history.append(f"select position ({r},{c}) '{grid[r][c]}' to explore.\n")

        if grid[r][c] == '○':
            # Already at end
            found_end = True
            break

        # Current cell arrow
        if grid[r][c] not in DIRECTIONS:
            # If it's an invalid symbol (not arrow, not '○'), skip
            raise ValueError
        dr, dc = DIRECTIONS[grid[r][c]]

        # Try moving k steps in that direction
        step = 1
        while True:
            nr = r + step * dr
            nc = c + step * dc
            if not in_bounds(nr, nc):
                history.append(f"Chose step={step}. Position ({nr},{nc}) is out of the bounds. So let's explore next node.\n")
                # Out of bounds or invalid => stop exploring further steps
                break
            if (nr, nc) not in visited:
                history.append(f"Chose step={step}. Add position ({nr},{nc}) to candidates.")
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))
                if grid[nr][nc] == '○':
                    #history.append((nr, nc, f"checking ({nr},{nc}) and ({nr},{nc}) is the end point"))
                    history[-1] = history[-1]+ f"Check position ({nr},{nc}). Position ({nr},{nc}) is the end point!\n"
                    found_end = True
                    break
                else:
                    #history.append((nr, nc, f"check ({nr},{nc}) and ({nr},{nc}) not the end point"))
                    history[-1] = history[-1]+ f"Check position ({nr},{nc}). Position ({nr},{nc}) is not the end point.\n"
            else:
                history.append(f"Chose step={step}. Position ({nr},{nc}) has been explored. So skip the position ({nr},{nc}).\n")
            
            step += 1

            if found_end:
                break
    
    
    # If we never found the end, puzzle is unsolvable from this start
    if not found_end:
        history.append(f"Fail! Candidates are empty! We explore all posibility but can't find the solution. So No path can be found to the end cell '○'.")
        #raise ValueError("No path found to the end cell '○'.")
        return False, history
    else:
        history.append(f"Find the solution! Now backtrack the full path.\n")

    # Reconstruct path from the end to the start
    path = []
    cur = end_cell
    while cur in parent or cur == start:
        path.append(cur)
        if cur == start:
            history.append("Back to the start node. Here is the final answer:\n")
            break
        cur_for_history = cur
        cur = parent[cur]
        history.append(f"Track ({cur_for_history[0]},{cur_for_history[1]}) and find the parent node is ({cur[0]},{cur[1]}).\n")
    path.reverse()  # Now it goes from start -> end

    # Prepare the result grid (same dimensions), fill with 0
    result = [[0]*cols for _ in range(rows)]

    # Label inflection points:
    #   - Start cell gets the first label (1).
    #   - Every time the direction changes from one cell to the next, we increment the label.
    #   - End cell gets labeled last.
    inflection_label = 0
    prev_dir = None

    for i, (r, c) in enumerate(path):
        symbol = grid[r][c]
        # For the end cell '○', we treat it as a different "direction" to ensure it is labeled
        current_dir = symbol if symbol in DIRECTIONS else '○'

        if current_dir != prev_dir:
            inflection_label += 1
            result[r][c] = inflection_label
            prev_dir = current_dir
        else:
            # It's on the path, but not a new inflection => could set to something else if needed
            # The puzzle examples do NOT label those. We leave them as 0 in the final output
            pass

    # If you do *not* want the end cell labeled, remove the above logic that sets it.

    # Convert result 2D array into the puzzle's required string format
    #   "[[r1, r2, ...], [r1, r2, ...], ...]"
    # However, from the examples, the format seems to be:
    #   "[[1 0 2,0 0 0,0 0 3]]"
    # i.e. each row is "space-separated" and the rows are separated by commas.
    rows_strings = []
    for rr in range(rows):
        row_str = " ".join(str(result[rr][cc]) for cc in range(cols))
        rows_strings.append(row_str)
    final_answer = "[[" + ",".join(rows_strings) + "]]"
    history.append(final_answer)
    
    history = ''.join(history) 
    
    return result, history


# ------------------------------------------------------------------------------
# Example usage:

if __name__ == "__main__":
    # Example 1:
    grid1 = [
        ['→', '↙', '↓'],
        ['↖', '↓', '↙'],
        ['↑', '←', '○'],
    ]
    ans1, history = solve_arrow_maze(grid1, start=(0, 0))
    print("Example 1 Answer:", ans1)
    
    print(history)
    # Expect something like "[[1 0 2,0 0 0,0 0 3]]"
    # depending on how you label inflections

    # Example 2:
    grid2 = [
        ['↘', '↙', '↓'],
        ['↖', '↓', '↙'],
        ['↑', '←', '○'],
    ]
    ans2, history = solve_arrow_maze(grid2, start=(0, 0))
    print("Example 2 Answer:", ans2)
    
    print(history)
    # Expect something like "[[1 0 0,0 0 0,0 0 2]]"
    
    # Example 2:
    grid3 = [
        ['↓', '↙', '↑'],
        ['→', '↖', '↓'],
        ['↗', '→', '○'],
    ]
    ans3, history = solve_arrow_maze(grid3, start=(0, 0))
    print("Example 3 Answer:", ans3)
    
    print(history)
    # Expect something like "[[1 0 0,0 0 0,0 0 2]]"
    
    grid3 = [
        ['↓', '↙', '↑',"↑"],
        ['→', '↖', '↘',"↙"],
        ['↙', '→', '↑',"○"],
    ]
    ans3, history = solve_arrow_maze(grid3, start=(0, 0))
    print("Example 3 Answer:", ans3)
    
    print(history)
    # Expect something like "[[1 0 0,0 0 0,0 0 2]]"
    
  
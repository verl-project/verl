
import random

def generate_campsite(n, m, k, seed=None, random_rate=0.2):
    """
    Generate an n×m Campsite puzzle with at least one valid solution.

    1) Randomly place k tents in the grid with no two tents adjacent 
       orthogonally or diagonally.
    2) Place trees so that each tent is orthogonally adjacent to at least one tree.
       (We can add extra random trees to keep the puzzle interesting.)
    3) Fill remaining cells with 'X'.
    4) Compute row and column constraints, i.e., how many tents in each row/column.
    5) Return the puzzle grid (with 'T' = tree, 'X' = empty) and the row/column constraints.

    :param n: number of rows
    :param m: number of columns
    :param k: number of tents to place
    :param seed: random seed for reproducibility (optional)
    :return: (grid, row_constraints, col_constraints)
             where grid is an n×m list of lists with 'T' or 'X'.
             row_constraints is a list of length n,
             col_constraints is a list of length m.
    """
    if seed is not None:
        random.seed(seed)

    # -------------------------------------------------------------------------
    # STEP 1: Randomly place k tents with no adjacency among them
    # -------------------------------------------------------------------------
    # We'll build our final "solution" using 'C' for tents, 'T' for trees, 'X' for empty.
    solution = [['X'] * m for _ in range(n)]
    
    # Keep track of tent locations
    tent_positions = []
    
    # We want to place k tents so that no two tents touch (8-direction adjacency).
    # We'll keep trying random positions until we place all k (or we exhaust attempts).
    # In a real puzzle generator, you often do more sophisticated logic to ensure
    # you can place all k tents in large grids, but let's keep this straightforward.
    
    # A function to check if a tent can be placed at (r, c)
    def can_place_tent(r, c):
        if solution[r][c] != 'X':
            return False
        # Check 8-direction adjacency
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < m:
                    if solution[rr][cc] == 'C':
                        return False
        return True

    all_cells = [(r, c) for r in range(n) for c in range(m)]
    random.shuffle(all_cells)

    placed = 0
    idx = 0
    # Try to place k tents in random positions
    while placed < k and idx < len(all_cells):
        r, c = all_cells[idx]
        if can_place_tent(r, c):
            solution[r][c] = 'C'
            tent_positions.append((r, c))
            placed += 1
        idx += 1

    # If we failed to place k tents, raise an error
    if placed < k:
        #raise ValueError(f"Could not place {k} tents without adjacency conflicts. Try a smaller k.")
        #return False, f"Could not place {k} tents without adjacency conflicts. Try a smaller k."
        print( f"Could not place {k} tents without adjacency conflicts. This case place {placed} tents")
    
    # -------------------------------------------------------------------------
    # STEP 2: Place trees so that each tent has at least one tree neighbor
    # -------------------------------------------------------------------------
    # For each tent, check if it already has a tree neighbor. If not, place a tree
    # in at least one valid neighbor cell. We can optionally place more trees to
    # make the puzzle more interesting.
    directions_orth = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for (r, c) in tent_positions:
        # Check if there's already at least one tree neighbor
        has_tree_neighbor = False
        for dr, dc in directions_orth:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < m:
                if solution[rr][cc] == 'T':
                    has_tree_neighbor = True
                    break
        
        # If we don't have a tree neighbor, place at least one tree
        if not has_tree_neighbor:
            # Gather valid spots around (r, c) where a tree can be placed
            valid_spots = []
            for dr, dc in directions_orth:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < m:
                    # "T" or "X" or "C" – if there's a tent, we can't override it.
                    if solution[rr][cc] == 'X':
                        valid_spots.append((rr, cc))
            # If valid_spots is empty, we have an impossible puzzle. Normally,
            # it shouldn't be empty because the tent wasn't originally placed
            # adjacent to another tent.
            if not valid_spots:
                raise RuntimeError("No valid spot to place a required tree! Puzzle generation error.")
            # Place a tree at exactly one of these spots (or random choice)
            rr, cc = random.choice(valid_spots)
            solution[rr][cc] = 'T'

    # Optionally: place extra random trees for puzzle variety
    # For demonstration, 20% chance to place a tree on any empty X,
    # so that the puzzle looks less minimal. Comment out if undesired.
    for r in range(n):
        for c in range(m):
            if solution[r][c] == 'X':
                if random.random() < random_rate:
                    solution[r][c] = 'T'

    # -------------------------------------------------------------------------
    # STEP 3: Now we have a "solution" with tents (C), trees (T), and some X's
    #         We want to produce the puzzle's T/X grid (the solver must find
    #         where 'C' should go).
    # -------------------------------------------------------------------------
    # To create the puzzle for the end user, we remove 'C' from the final puzzle
    # and replace them with 'X'. So the puzzle that the user sees only has 'T' and 'X'.

    puzzle = []
    for r in range(n):
        row = []
        for c in range(m):
            if solution[r][c] == 'C':
                # Remove the tent from the puzzle the user sees
                # so the user can solve for the tents
                row.append('X')
            else:
                row.append(solution[r][c])
        puzzle.append(row)

    # -------------------------------------------------------------------------
    # STEP 4: Compute row/col constraints from the actual solution with tents
    # -------------------------------------------------------------------------
    row_constraints = [0] * n
    col_constraints = [0] * m
    for r in range(n):
        for c in range(m):
            if solution[r][c] == 'C':
                row_constraints[r] += 1
                col_constraints[c] += 1

    # Return puzzle (trees + X), row/col constraints,
    # and optionally the *solution* grid if you want to verify.
    return puzzle, row_constraints, col_constraints, solution

# ------------------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Generate a 5×7 puzzle with 6 tents, with a fixed random seed for reproducibility
    gen_puzzle, row_cons, col_cons, solution = generate_campsite(n=5, m=7, k=6, seed=None, random_rate=0.2)
    
    # Print puzzle in a readable way
    print("Puzzle Layout (T/X):")
    for row in gen_puzzle:
        print(" ".join(row))
    
    print("\nRow Constraints:", row_cons)
    print("Col Constraints:", col_cons)

    print("\n(For debugging) One valid solution used to generate the puzzle (C = tent):")
    for row in solution:
        print(" ".join(row))


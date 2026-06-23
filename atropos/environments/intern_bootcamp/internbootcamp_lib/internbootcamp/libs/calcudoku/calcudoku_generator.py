import random
import itertools
from collections import deque

class CalcudokuGenerator:

    def __init__(
        self, 
        n=6, 
        group_size_range=(1, 5), 
        seed=None
    ):
        """
        n: the puzzle dimension (n x n).
        group_size_range: tuple (min_group_size, max_group_size) or any range-like object
                        describing how large or small each group can be.
        seed: optional random seed for reproducibility.
        """
        self.n = n
        self.group_size_range = group_size_range
        if seed is not None:
            random.seed(seed)

        # solution_grid will hold the fully-specified solution
        self.solution_grid = [[0]*n for _ in range(n)]
        # Each cell's group label (for puzzle spec)
        self.cell_group_label = [[None]*n for _ in range(n)]
        # Each group will have an (operation, target_value, cells)
        self.groups_info = []  

    def generate_puzzle(self):
        """
        Generate the puzzle in several steps:
        1) Generate random valid n×n grid with each row & column a permutation of [1..n].
        2) Partition cells into contiguous groups of size in group_size_range.
        3) Assign operation & target for each group.
        4) Build puzzle spec.
        """
        # Step 1: Generate a random solution grid (Latin square).
        self._generate_random_solution()

        # Step 2: Partition into random contiguous groups
        self._create_random_groups()

        # Step 3: Assign an operation and compute target for each group
        self._assign_operations_and_targets()

        # Step 4: Build puzzle spec lines
        puzzle_spec = self._build_puzzle_spec()
        return puzzle_spec

    def _generate_random_solution(self):
        """
        Create an n×n grid that satisfies "all numbers 1..n in each row & column exactly once."
        This is effectively generating a random Latin square.
        """
        # A naive approach:
        #   For row i, randomly shuffle [1..n], but if we ever fail to place
        #   them uniquely in columns, re-try. 
        # For larger n, you might want a more robust approach.

        # Start with a random row permutation for row 0
        available_cols = [list(range(1, self.n+1)) for _ in range(self.n)]
        # for each row, we'll choose a permutation that doesn't clash with above rows

        for r in range(self.n):
            placed = False
            attempts = 0
            while not placed:
                attempts += 1
                if attempts > 10000:
                    # fallback: if this happens a lot, it's a sign we are stuck; re-start
                    raise RuntimeError("Stuck generating random solution. Try a different approach or seed.")
                row_perm = random.sample(range(1, self.n+1), self.n)
                # Check if it fits uniqueness constraints vs. columns
                valid = True
                for c in range(self.n):
                    val = row_perm[c]
                    # check column constraint
                    for rr in range(r):
                        if self.solution_grid[rr][c] == val:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Place in solution
                    for c in range(self.n):
                        self.solution_grid[r][c] = row_perm[c]
                    placed = True

    def _create_random_groups(self):
        """
        Randomly partition the grid into contiguous groups.
        We'll do a BFS/DFS-based approach:
        1) Start from any unvisited cell.
        2) Decide how big this group should be (random size within group_size_range, but not exceeding leftover cells).
        3) Grow that group by exploring neighbors until we reach desired size or can't expand further.
        4) Label those cells as belonging to the new group.  
        Continue until all cells are assigned to some group.

        For group labeling, we'll just use letters in sequence, then 'AA', etc. for bigger boards if needed.
        """
        label_list = self._generate_group_labels(self.n * self.n)

        all_cells = [(r, c) for r in range(self.n) for c in range(self.n)]
        random.shuffle(all_cells)  # random order to pick seeds
        visited = [[False]*self.n for _ in range(self.n)]
        label_index = 0

        for (start_r, start_c) in all_cells:
            if visited[start_r][start_c]:
                continue

            # Decide group size
            desired_size = random.randint(self.group_size_range[0], self.group_size_range[1])
            group_cells = []
            queue = deque()
            queue.append((start_r, start_c))
            group_cells.append((start_r, start_c))
            visited[start_r][start_c] = True

            while queue and len(group_cells) < desired_size:
                cur_r, cur_c = queue.popleft()
                # Explore neighbors in random order
                neighbors = []
                if cur_r > 0: 
                    neighbors.append((cur_r-1, cur_c))
                if cur_r < self.n-1: 
                    neighbors.append((cur_r+1, cur_c))
                if cur_c > 0: 
                    neighbors.append((cur_r, cur_c-1))
                if cur_c < self.n-1: 
                    neighbors.append((cur_r, cur_c+1))
                random.shuffle(neighbors)
                for nr, nc in neighbors:
                    if not visited[nr][nc]:
                        visited[nr][nc] = True
                        group_cells.append((nr, nc))
                        queue.append((nr, nc))
                    if len(group_cells) >= desired_size:
                        break

            # Assign these group cells to the label
            label = label_list[label_index]
            label_index += 1
            for (r, c) in group_cells:
                self.cell_group_label[r][c] = label

        # Collect group defs
        group_dict = {}
        for r in range(self.n):
            for c in range(self.n):
                lbl = self.cell_group_label[r][c]
                if lbl not in group_dict:
                    group_dict[lbl] = []
                group_dict[lbl].append((r, c))

        self.groups_info = [(lbl, group_dict[lbl]) for lbl in group_dict]

    def _assign_operations_and_targets(self):
        """
        For each group, pick an operation (+, -, *, /) at random, and compute the target.
        Requirements:
            - For '-' or '/', the result must be an integer > 0.
            - If a group has 1 cell, then the operation can be anything, but '+' or '*' makes sense 
            (or we treat single cells as special, effectively no operation needed).
            We'll just do '+' with that single cell's value.
        """
        possible_ops = ['+', '-', '*', '/']
        updated_groups = []

        for (label, cells) in self.groups_info:
            # Gather the solution values for these cells
            values = [self.solution_grid[r][c] for (r, c) in cells]
            # Decide operation
            op = None
            target_val = None

            # If group has only one cell, trivial: use '+' with target = that cell's value
            if len(cells) == 1:
                op = '+'
                target_val = values[0]
                updated_groups.append((label, op, target_val, cells))
                continue

            # Otherwise, keep trying random ops until we find one that works
            trial_ops = possible_ops[:]
            random.shuffle(trial_ops)
            success = False
            while trial_ops and not success:
                candidate_op = trial_ops.pop()
                candidate_target = self._calculate_target(candidate_op, values)
                if candidate_target is not None:
                    op = candidate_op
                    target_val = candidate_target
                    success = True

            if not success:
                # fallback: if none of the 4 ops worked out for integral positive, 
                # just default to '+' with sum
                op = '+'
                target_val = sum(values)

            updated_groups.append((label, op, target_val, cells))

        # Now we have operation & target for each group
        # Overwrite self.groups_info with operation included
        self.groups_info = updated_groups

    def _calculate_target(self, op, values):
        """
        Given an operation (op) and a list of integer values, return the target value if it is
        valid (positive integer, etc.) for the group, otherwise None.
        """
        if op == '+':
            return sum(values)
        elif op == '*':
            prod = 1
            for v in values:
                prod *= v
            return prod
        elif op == '-':
            # For subtraction, standard KenKen typically only has 2 cells. 
            # But let's handle any size: we check if there's a permutation p of values
            # so that p[0] - p[1] - ... p[n-1] is a positive integer.
            # We'll just pick the difference as max(val) - sum(others) for 2-cell groups, or do a check for multi-cell.
            if len(values) == 2:
                a, b = values[0], values[1]
                diff1 = a - b
                diff2 = b - a
                # Must be positive
                if diff1 > 0:
                    return abs(diff1)
                elif diff2 > 0:
                    return abs(diff2)
                else:
                    return None
            else:
                # For multi-cell difference, let's see if any permutation yields a positive result
                from itertools import permutations
                for p in permutations(values):
                    result = p[0]
                    for x in p[1:]:
                        result -= x
                    if result > 0:
                        return result
                return None

        elif op == '/':
            # For division, typically 2 cells. If bigger, we check permutations similarly.
            # For 2 cells, we just do int division check. 
            if len(values) == 2:
                a, b = values[0], values[1]
                # We want either a/b or b/a to be a positive integer
                if a >= b and a % b == 0:
                    return a // b
                elif b > a and b % a == 0:
                    return b // a
                else:
                    return None
            else:
                # multi-cell division check
                from itertools import permutations
                for p in permutations(values):
                    numerator = p[0]
                    ok = True
                    for x in p[1:]:
                        if numerator % x != 0: 
                            ok = False
                            break
                        numerator //= x
                    if ok and numerator > 0:
                        return numerator
                return None
        # If we reach here:
        return None

    def _build_puzzle_spec(self):
        """
        Return a list of puzzle_spec lines. Each line is an n-element list (strings) joined by spaces. 
        Each cell is either:
        - <group_label><op><target> if we want to embed the group clue right there, or
        - <group_label> if the operation/target is shown on a different cell (like how KenKen typically does).
        
        However, the usual puzzle_spec from the prompt includes exactly one cell in each group having
        e.g. "J+12", and the other cells in that group just "J".

        We'll do that: pick one cell in each group to display the clue, and the others only show the label.
        """
        # We’ll pick the first cell in each group to carry the operation & target
        # The rest only show the group label.
        puzzle_spec_grid = [['']*self.n for _ in range(self.n)]

        # Make a map group_label -> (op, target, cells)
        group_map = {}
        for (label, op, target, cells) in self.groups_info:
            group_map[label] = (op, target, cells)

        # Which cell in each group will carry the “op+target”?
        # We’ll just pick the first cell in cells for that group
        label_to_clue_cell = {}
        for lbl, (op, tgt, cells) in group_map.items():
            # pick first cell
            clue_cell = cells[0]
            label_to_clue_cell[lbl] = clue_cell

        # Now fill puzzle_spec_grid with either
        #   "<lbl><op><target>" if (r,c) == clue cell
        #   or simply "<lbl>" otherwise
        for r in range(self.n):
            for c in range(self.n):
                lbl = self.cell_group_label[r][c]
                op, tgt, cells = group_map[lbl]
                if (r, c) == label_to_clue_cell[lbl]:
                    # show the op & target
                    # e.g. "J+12" 
                    # If it's a single-cell group, it’s likely a +. We'll just do the same format.
                    puzzle_spec_grid[r][c] = f"{lbl}{op}{tgt}"
                else:
                    # just show the label
                    puzzle_spec_grid[r][c] = lbl

        # Optionally, we can also embed the solution number in one or two cells if we want to provide partial clues,
        # but that isn't strictly required for constructing a puzzle spec. 
        # The puzzle spec from your example sometimes contained cells like "5" or "3" with no label, indicating an initial clue.
        # If you want to add givens, you can do that here by randomly choosing a few cells to reveal.

        # Build lines as strings
        puzzle_spec_lines = []
        for r in range(self.n):
            row_str = " ".join(puzzle_spec_grid[r])
            puzzle_spec_lines.append(row_str)

        return puzzle_spec_lines

    def _generate_group_labels(self, count):
        """
        Returns a list of group labels (strings) of at least 'count' distinct labels.
        For n up to 26, you can just use single letters A..Z.
        If you might need more, generate double letters, triple letters, etc.
        """
        labels = []
        # build single letters first
        for ch in range(ord('A'), ord('Z')+1):
            labels.append(chr(ch))
        # then double letters if needed
        if len(labels) < count:
            for ch1 in range(ord('A'), ord('Z')+1):
                for ch2 in range(ord('A'), ord('Z')+1):
                    labels.append(chr(ch1)+chr(ch2))
                    if len(labels) >= count:
                        break
                if len(labels) >= count:
                    break
        return labels[:count]
    
def example_usage():
    """
    Generate a puzzle spec for a 6x6 Calcudoku puzzle.
    """
    generator = CalcudokuGenerator(n=6, group_size_range=(1,4), seed=1234)
    puzzle_spec = generator.generate_puzzle()
    print(f"puzzle_spec {puzzle_spec}")
    print("Generated puzzle spec:")
    for line in puzzle_spec:
        print(line)

if __name__ == "__main__":
    example_usage()
"""
Test a 2048 strategy against the game engine.

Usage:
    python examples/test_2048_strategy.py                      # uses built-in demo strategy
    python examples/test_2048_strategy.py --seed 123           # fixed seed
    python examples/test_2048_strategy.py --timeout 10         # longer timeout
    python examples/test_2048_strategy.py --strategy my_strat.py  # load from file
"""

import argparse
import importlib.util
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

spec = importlib.util.spec_from_file_location(
    "game2048",
    os.path.join(os.path.dirname(__file__), "../verl/utils/reward_score/game2048.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ---------------------------------------------------------------------------
# Default strategy to test — edit this directly or pass --strategy
# ---------------------------------------------------------------------------
DEFAULT_STRATEGY = '''
def strategy(board):
    def move_possible(board, direction):
        rows, cols = len(board), len(board[0])
        if direction == 'W':
            for j in range(cols):
                for i in range(1, rows):
                    if board[i][j] != 0:
                        for k in range(i-1, -1, -1):
                            if board[k][j] == 0 or board[k][j] == board[i][j]:
                                return True
                            if board[k][j] != 0:
                                break
        elif direction == 'S':
            for j in range(cols):
                for i in range(rows-2, -1, -1):
                    if board[i][j] != 0:
                        for k in range(i+1, rows):
                            if board[k][j] == 0 or board[k][j] == board[i][j]:
                                return True
                            if board[k][j] != 0:
                                break
        elif direction == 'A':
            for i in range(rows):
                for j in range(1, cols):
                    if board[i][j] != 0:
                        for k in range(j-1, -1, -1):
                            if board[i][k] == 0 or board[i][k] == board[i][j]:
                                return True
                            if board[i][k] != 0:
                                break
        elif direction == 'D':
            for i in range(rows):
                for j in range(cols-2, -1, -1):
                    if board[i][j] != 0:
                        for k in range(j+1, cols):
                            if board[i][k] == 0 or board[i][k] == board[i][j]:
                                return True
                            if board[i][k] != 0:
                                break
        return False

    # Prefer moves that allow a merge as they increase score
    for d in ('W', 'S', 'A', 'D'):
        if move_possible(board, d):
            return d
    # If no merges are possible, pick any direction that moves tiles
    for d in ('W', 'S', 'A', 'D'):
        if any(board[i][j] != 0 for i in range(len(board)) for j in range(len(board[0]))):
            return d
    return 'W'
'''

# ---------------------------------------------------------------------------
# A smarter reference strategy for comparison
# ---------------------------------------------------------------------------
REFERENCE_STRATEGY = '''
def strategy(board):
    """Always prefer W > A > D > S, skip if no change."""
    def compress(row):
        tiles = [x for x in row if x]
        i, merged = 0, []
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i+1]:
                merged.append(tiles[i] * 2)
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        return merged + [0] * (len(row) - len(merged))

    def move(board, direction):
        n = len(board)
        b = [row[:] for row in board]
        if direction in ("W", "S"):
            b = [[b[r][c] for r in range(n)] for c in range(n)]  # transpose
        if direction in ("S", "D"):
            b = [list(reversed(row)) for row in b]
        b = [compress(row) for row in b]
        if direction in ("S", "D"):
            b = [list(reversed(row)) for row in b]
        if direction in ("W", "S"):
            b = [[b[c][r] for c in range(n)] for r in range(n)]  # un-transpose
        return b

    def changed(a, b):
        return any(a[r][c] != b[r][c] for r in range(len(a)) for c in range(len(a)))

    for d in ["W", "A", "D", "S"]:
        if changed(board, move(board, d)):
            return d
    return "W"
'''


def run_strategy(code: str, seed: int, timeout: int, label: str):
    print(f"\n{'='*60}")
    print(f"Strategy: {label}")
    print(f"Seed: {seed}  |  Timeout: {timeout}s")
    print("=" * 60)

    ok, info = mod.check_python_modules(code)
    if "error" in info:
        print(f"SYNTAX ERROR: {info['error']}")
        return
    if not ok:
        print(f"CHEATING DETECTED: non-stdlib imports {info['non_stdlib']}")
        return

    try:
        fn = mod.create_locked_down_function(code)
    except Exception as e:
        print(f"EXEC FAILED: {e}")
        return

    game = mod.GameBoard(size=4, seed=seed, target=256, probability_fours=0.10)

    try:
        steps, state = mod.execute_strategy_with_timeout(fn, game, timeout_seconds=timeout)
    except mod._TimeoutError as e:
        steps = e.steps
        state = "timeout"
    except Exception as e:
        print(f"EXCEPTION during play: {type(e).__name__}: {e}")
        return

    board = game.board()
    max_tile = max(cell for row in board for cell in row)
    score = game.score()

    print(f"Result : {state}")
    print(f"Steps  : {steps:,}")
    print(f"Score  : {score:,}")
    print(f"Max tile: {max_tile}")
    print(f"\nFinal board:")
    for row in board:
        print("  " + "  ".join(f"{v:5d}" if v else "    ." for v in row))


def load_strategy_from_file(path: str) -> str:
    with open(path) as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None, help="Path to .py file containing strategy(board)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--reference", action="store_true", help="Also run the reference strategy")
    args = parser.parse_args()

    if args.strategy:
        code = load_strategy_from_file(args.strategy)
        label = args.strategy
    else:
        code = DEFAULT_STRATEGY
        label = "default (always W)"

    run_strategy(code, seed=args.seed, timeout=args.timeout, label=label)

    if args.reference:
        run_strategy(REFERENCE_STRATEGY, seed=args.seed, timeout=args.timeout, label="reference (W>A>D>S)")

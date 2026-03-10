# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward function for the 2048 game RL task.

The LLM is asked to generate a Python strategy(board) function that plays
the 2048 game. The reward is based on:
  1. Whether the generated code is syntactically valid (+1 / -2)
  2. Whether the code only uses stdlib Python, no third-party packages (+1 / -20)
  3. Whether the strategy actually wins (reaches target tile) when executed (+20 / +2 / -1)

Game logic ported from the Unsloth reinforcement-fine-tuning notebook.
"""

import ast
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 2048 game implementation (ported from the Unsloth notebook)
# ---------------------------------------------------------------------------

def _compress_and_merge_row_left(row: List[int]) -> Tuple[List[int], int, bool]:
    n = len(row)
    tiles = [x for x in row if x != 0]
    gained = 0
    i = 0
    merged = []
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] * 2
            gained += v
            merged.append(v)
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (n - len(merged))
    changed = merged != row
    return merged, gained, changed


def _move_left(board):
    changed_any = False
    total_gain = 0
    new_board = []
    for row in board:
        new_row, gained, changed = _compress_and_merge_row_left(row)
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any


def _move_right(board):
    changed_any = False
    total_gain = 0
    new_board = []
    for row in board:
        rev = list(reversed(row))
        new_rev, gained, changed = _compress_and_merge_row_left(rev)
        new_row = list(reversed(new_rev))
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any


def _transpose(board):
    return [list(row) for row in zip(*board)]


def _move_up(board):
    t = _transpose(board)
    moved, gain, changed = _move_left(t)
    return _transpose(moved), gain, changed


def _move_down(board):
    t = _transpose(board)
    moved, gain, changed = _move_right(t)
    return _transpose(moved), gain, changed


def _empty_cells(board):
    size = len(board)
    return [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]


def _can_move(board):
    if _empty_cells(board):
        return True
    size = len(board)
    for r in range(size):
        for c in range(size - 1):
            if board[r][c] == board[r][c + 1]:
                return True
    for r in range(size - 1):
        for c in range(size):
            if board[r][c] == board[r + 1][c]:
                return True
    return False


import random


@dataclass
class GameBoard:
    size: int
    seed: Optional[int] = None
    target: int = 2048
    probability_fours: float = 0.10
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _score: int = field(default=0, init=False, repr=False)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("Board size must be at least 2.")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self._add_random_tile()
        self._add_random_tile()
        self._update_state_after_change()

    def board(self):
        return self._board

    def state(self):
        return self._state

    def score(self):
        return self._score

    def do_action(self, key: str) -> None:
        if self._state != "ongoing":
            return
        if not isinstance(key, str) or len(key) == 0:
            self._state = "failed"
            return
        k = key.strip().lower()
        if k == "q":
            self._state = "failed"
            return
        move_map = {"a": _move_left, "d": _move_right, "w": _move_up, "s": _move_down}
        if k not in move_map:
            self._state = "failed"
            return
        mover = move_map[k]
        new_board, gain, changed = mover(self._board)
        if changed:
            self._board = new_board
            self._score += gain
            self._add_random_tile()
        self._update_state_after_change()

    def _add_random_tile(self) -> bool:
        empties = _empty_cells(self._board)
        if not empties:
            return False
        r, c = self._rng.choice(empties)
        self._board[r][c] = 4 if self._rng.random() < self.probability_fours else 2
        return True

    def _update_state_after_change(self) -> None:
        if any(self.target in row for row in self._board):
            self._state = "success"
            return
        if not _can_move(self._board):
            self._state = "failed"
            return
        self._state = "ongoing"


# ---------------------------------------------------------------------------
# Sandbox utilities
# ---------------------------------------------------------------------------

# Allowed stdlib module names (broad allowlist — covers almost all stdlib)
_STDLIB_MODULES = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else {
    "abc", "ast", "builtins", "collections", "copy", "dataclasses", "enum",
    "functools", "heapq", "inspect", "itertools", "math", "operator", "queue",
    "random", "re", "statistics", "string", "sys", "textwrap", "typing",
    "types", "weakref",
}


def check_python_modules(code: str) -> Tuple[bool, dict]:
    """Parse `code` with ast and check that all imports are stdlib-only."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {"error": f"SyntaxError: {e}", "stdlib": [], "non_stdlib": [], "relative_imports": 0}

    stdlib = []
    non_stdlib = []
    relative_imports = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _STDLIB_MODULES:
                    stdlib.append(top)
                else:
                    non_stdlib.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                relative_imports += 1
            else:
                top = (node.module or "").split(".")[0]
                if top in _STDLIB_MODULES:
                    stdlib.append(top)
                else:
                    non_stdlib.append(top)

    ok = len(non_stdlib) == 0 and relative_imports == 0
    return ok, {"stdlib": stdlib, "non_stdlib": non_stdlib, "relative_imports": relative_imports}


def create_locked_down_function(code: str) -> Callable:
    """
    exec `code` in a restricted namespace (no builtins globals leak) and
    return the `strategy` callable defined inside.
    """
    # Provide a minimal set of safe builtins
    safe_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in ("open", "exec", "eval", "compile", "__import__", "input")
    } if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k)
        for k in dir(__builtins__)
        if k not in ("open", "exec", "eval", "compile", "__import__", "input")
    }

    ns = {"__builtins__": safe_builtins}
    exec(compile(code, "<strategy>", "exec"), ns)  # noqa: S102

    if "strategy" not in ns:
        raise ValueError("No function named 'strategy' found in generated code.")
    fn = ns["strategy"]
    if not callable(fn):
        raise ValueError("'strategy' is not callable.")
    return fn


class _TimeoutError(Exception):
    pass


def execute_strategy_with_timeout(strategy: Callable, game: GameBoard, timeout_seconds: int = 5):
    """Run strategy(board) in a loop until game ends or timeout.
    Uses a time-based check so it works safely inside Ray worker threads
    (signal.SIGALRM only works in the main thread).
    """
    import time
    deadline = time.time() + timeout_seconds
    steps = 0
    while game.state() == "ongoing":
        if time.time() > deadline:
            raise _TimeoutError("Strategy timed out")
        action = strategy(list(game.board()))
        steps += 1
        if not isinstance(action, str):
            return steps, "failed"
        game.do_action(action)
    return steps, game.state()


# ---------------------------------------------------------------------------
# Extract strategy function from LLM response
# ---------------------------------------------------------------------------

def extract_function(text: str) -> Optional[str]:
    """Extract the Python function wrapped in triple backticks."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n").removeprefix("python3\n")
        # Find the def start
        def_pos = fx.find("def strategy(board)")
        if def_pos == -1:
            def_pos = fx.find("def strategy(board )")
        if def_pos != -1:
            return fx[def_pos:]
    return None


# ---------------------------------------------------------------------------
# Top-level compute_score (verl interface)
# ---------------------------------------------------------------------------

_PRINTER = 0  # module-level counter, increments per sample (per Ray worker)


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """
    Compute the reward for a 2048 strategy generation response.

    Args:
        solution_str: The LLM-generated response text.
        ground_truth:  Unused for 2048 (no symbolic ground truth). Pass "win".
        extra_info:    Optional dict; may contain 'seed' for reproducible game.

    Returns:
        A float reward composed of three sub-scores:
          function_works  (+1.0 valid / -2.0 unparseable)
          no_cheating     (+1.0 stdlib-only / -20.0 uses third-party libs)
          strategy_result (+20.0 wins / +2.0 runs but loses / -1.0 timeout / 0 crash)
    """
    global _PRINTER
    import numpy as np

    function = extract_function(solution_str)

    should_print = (_PRINTER % 5 == 0)
    _PRINTER += 1

    # ---- function_works ----
    if function is None:
        if should_print:
            print(f"[2048] sample {_PRINTER} | no function found | score=-2.0")
        return -2.0  # No function found at all

    ok, info = check_python_modules(function)
    if "error" in info:
        if should_print:
            print(f"[2048] sample {_PRINTER} | syntax error | score=-2.0\n{function}\n")
        return -2.0  # Syntax error

    try:
        strategy_fn = create_locked_down_function(function)
    except Exception:
        return -0.5  # exec failed

    score = 1.0  # function_works reward

    # ---- no_cheating ----
    if not ok:
        score += -20.0
        if should_print:
            print(f"[2048] sample {_PRINTER} | cheating detected | score={score}\n{function}\n")
        return score
    else:
        score += 1.0

    # ---- strategy_result ----
    seed = (extra_info or {}).get("seed", int(np.random.randint(10000)))
    game_state = "unknown"
    steps = 0
    exc_msg = ""
    try:
        game = GameBoard(size=6, seed=seed, target=2048, probability_fours=0.10)
        steps, game_state = execute_strategy_with_timeout(strategy_fn, game, timeout_seconds=5)
        if game_state == "success":
            score += 20.0
        else:
            score += 2.0
    except _TimeoutError:
        game_state = "timeout"
        score += -1.0
    except Exception as e:
        game_state = "exception"
        exc_msg = f" | error={type(e).__name__}: {e}"
        score += -3.0

    if should_print:
        print(f"[2048] sample {_PRINTER} | steps={steps} state={game_state} score={score}{exc_msg}\n{function}\n")

    return score

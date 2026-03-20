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


def _make_safe_import():
    """Return a restricted __import__ that only allows stdlib modules."""
    import importlib

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if top not in _STDLIB_MODULES:
            raise ImportError(f"Import of non-stdlib module '{name}' is not allowed")
        return importlib.__import__(name, globals, locals, fromlist, level)

    return _safe_import


def create_locked_down_function(code: str) -> Callable:
    """
    exec `code` in a restricted namespace (no builtins globals leak) and
    return the `strategy` callable defined inside.
    """
    builtins_src = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_builtins = {
        k: v for k, v in builtins_src.items()
        if k not in ("open", "exec", "eval", "compile", "__import__", "input")
    }
    # Allow stdlib imports but block third-party packages
    safe_builtins["__import__"] = _make_safe_import()

    ns = {"__builtins__": safe_builtins}
    exec(compile(code, "<strategy>", "exec"), ns)  # noqa: S102

    if "strategy" not in ns:
        raise ValueError("No function named 'strategy' found in generated code.")
    fn = ns["strategy"]
    if not callable(fn):
        raise ValueError("'strategy' is not callable.")
    return fn


class _TimeoutError(Exception):
    def __init__(self, msg, steps=0):
        super().__init__(msg)
        self.steps = steps


def execute_strategy_with_timeout(strategy: Callable, game: GameBoard,
                                   timeout_seconds: int = 5,
                                   strategy_code: Optional[str] = None):
    """Run strategy(board) until the game ends or the timeout fires.

    Uses SIGALRM in the main thread (tests/scripts) and
    PyThreadState_SetAsyncExc in worker threads (Ray run_in_executor) so that
    a blocking strategy call is hard-interrupted in both contexts.
    strategy_code is accepted for API compatibility but unused here.
    """
    import signal
    import threading

    steps = 0

    if threading.current_thread() is threading.main_thread():
        # Main thread: use SIGALRM — hard-interrupts even C-level waits.
        class _Alarm(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _Alarm()

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        remaining = signal.alarm(timeout_seconds)
        try:
            while game.state() == "ongoing":
                action = strategy([row[:] for row in game.board()])
                steps += 1
                if not isinstance(action, str):
                    return steps, "failed"
                game.do_action(action)
            return steps, game.state()
        except _Alarm:
            raise _TimeoutError("Strategy timed out", steps=steps)
        finally:
            signal.alarm(remaining)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Worker thread (Ray run_in_executor): use a threading.Event flag set
        # by a Timer.  No exception injection — _TimeoutError is only raised
        # from this thread at a safe check point, so it can never escape into
        # Ray's event loop or disrupt the training-step logger.
        timed_out = threading.Event()
        timer = threading.Timer(timeout_seconds, timed_out.set)
        timer.start()
        try:
            while game.state() == "ongoing":
                if timed_out.is_set():
                    raise _TimeoutError("Strategy timed out", steps=steps)
                action = strategy([row[:] for row in game.board()])
                steps += 1
                if not isinstance(action, str):
                    return steps, "failed"
                game.do_action(action)
            return steps, game.state()
        finally:
            timer.cancel()


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
    import numpy as np

    function = extract_function(solution_str)

    # ---- function_works ----
    if function is None:
        print("[2048] no function found | score=-2.0")
        return -2.0

    ok, info = check_python_modules(function)
    if "error" in info:
        print(f"[2048] syntax error | score=-2.0")
        # print(f"[2048] syntax error | score=-2.0\n{function}\n")
        return -2.0

    try:
        strategy_fn = create_locked_down_function(function)
    except Exception as e:
        print(f"[2048] exec failed | score=-0.5 | {type(e).__name__}: {e}")
        # print(f"[2048] exec failed | score=-0.5 | {type(e).__name__}: {e}\n{function}\n")
        return -0.5

    score = 1.0  # function_works reward

    # ---- no_cheating ----
    if not ok:
        score += -20.0
        print(f"[2048] cheating detected | score={score}")
        # print(f"[2048] cheating detected | score={score}\n{function}\n")
        return score
    else:
        score += 1.0

    # ---- strategy_result ----
    seed = (extra_info or {}).get("seed", int(np.random.randint(10000)))
    game_state = "unknown"
    steps = 0
    game_score = 0
    max_tile = 0
    exc_msg = ""
    game = GameBoard(size=4, seed=seed, target=256, probability_fours=0.10)
    try:
        steps, game_state = execute_strategy_with_timeout(
            strategy_fn, game, timeout_seconds=5, strategy_code=function
        )
        if game_state == "success":
            score += 20.0
        else:
            score += 2.0
    except _TimeoutError as e:
        game_state = "timeout"
        steps = e.steps
        score += -1.0
    except Exception as e:
        game_state = "exception"
        exc_msg = f" - error:{type(e).__name__}:{e}"
        score += -3.0

    # Capture game stats regardless of how the game ended.
    game_score = game.score()
    max_tile = max(c for row in game.board() for c in row)

    fields = [
        f"game/state:{game_state}",
        f"game/steps:{steps}",
        f"game/score:{game_score}",
        f"game/max_tile:{max_tile}",
        f"reward/total:{score}",
    ]
    if exc_msg:
        fields.append(exc_msg.lstrip(" - "))
    print(f"[2048] {' - '.join(fields)}")
    # print(f"[2048] {' - '.join(fields)}\n{function}\n")

    return score

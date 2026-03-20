"""
EquationFittingInteraction: RL environment for scientific equation discovery.

This interaction:
1. Parses LLM-generated Python code (equation skeleton with params[])
2. Executes in a sandboxed namespace
3. Fits parameters via scipy BFGS optimization
4. Computes reward = R_fit + R_valid + R_complexity
5. Returns diagnostic feedback for multi-turn self-correction
"""

import ast
import json
import logging
import os
import re
import signal
import warnings
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ──────────────────────────────────────────────
# Code parsing utilities
# ──────────────────────────────────────────────


def extract_python_code(text: str) -> Optional[str]:
    """Extract the first ```python ... ``` block from text."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_think_block(text: str) -> Optional[str]:
    """Extract content within <think>...</think> tags."""
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def count_params(code: str) -> int:
    """Detect the number of params used by finding max params[i] index."""
    pattern = r"params\[(\d+)\]"
    indices = [int(m) for m in re.findall(pattern, code)]
    if not indices:
        return 0
    return max(indices) + 1


def count_ast_nodes(code: str) -> int:
    """Count AST nodes in the equation function body as a complexity measure."""
    try:
        tree = ast.parse(code)
        # Find the equation function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "equation":
                return sum(1 for _ in ast.walk(node))
        # Fallback: count all nodes
        return sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        return 0


def check_uses_input(code: str, n_variables: int) -> bool:
    """Check if the equation actually uses input variables (X[:, i])."""
    for i in range(n_variables):
        if f"X[:, {i}]" in code or f"X[:,{i}]" in code:
            return True
    # Also check for individual variable names unpacked from X
    if "X[:," in code or "X[:" in code:
        return True
    return False


# ──────────────────────────────────────────────
# Sandboxed execution
# ──────────────────────────────────────────────

class ExecutionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ExecutionTimeout("Code execution timed out")


def execute_equation_sandboxed(
    code: str,
    X: np.ndarray,
    y: np.ndarray,
    max_params: int = 10,
    bfgs_maxiter: int = 100,
    n_restarts: int = 3,
    timeout_seconds: int = 30,
) -> dict:
    """
    Execute LLM-generated equation code in a sandbox and fit parameters via BFGS.

    Returns dict with keys:
        success (bool), mse (float), nmse (float), r_squared (float),
        best_params (list), y_pred (ndarray), error (str)
    """
    from scipy.optimize import minimize

    result = {
        "success": False,
        "mse": float("inf"),
        "nmse": float("inf"),
        "r_squared": float("-inf"),
        "best_params": [],
        "y_pred": None,
        "error": None,
    }

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        # Execute in isolated namespace with numpy pre-injected.
        # We allow __import__ for numpy only so that `import numpy as np` works,
        # but block other potentially dangerous imports.
        _ALLOWED_MODULES = {"numpy", "math"}

        def _safe_import(name, *args, **kwargs):
            if name in _ALLOWED_MODULES:
                return __import__(name, *args, **kwargs)
            raise ImportError(f"Import of '{name}' is not allowed in the sandbox")

        safe_builtins = {
            "__import__": _safe_import,
            "range": range, "len": len, "abs": abs, "min": min, "max": max,
            "sum": sum, "float": float, "int": int, "bool": bool,
            "list": list, "tuple": tuple, "dict": dict, "set": set,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "sorted": sorted, "reversed": reversed, "round": round, "pow": pow,
            "isinstance": isinstance, "type": type, "str": str,
            "print": lambda *a, **kw: None,  # no-op print
            "ValueError": ValueError, "TypeError": TypeError,
            "ZeroDivisionError": ZeroDivisionError, "RuntimeError": RuntimeError,
        }
        safe_namespace = {"np": np, "numpy": np, "__builtins__": safe_builtins}
        exec(code, safe_namespace)

        if "equation" not in safe_namespace:
            result["error"] = "No function named 'equation' found in the code."
            return result

        equation_fn = safe_namespace["equation"]

        # Determine number of params
        n_params = max(count_params(code), 1)
        n_params = min(n_params, max_params)

        y_var = np.var(y)
        if y_var == 0:
            y_var = 1.0  # Prevent division by zero

        best_mse = float("inf")
        best_params = None
        best_y_pred = None

        # Multiple random restarts for BFGS
        rng = np.random.RandomState(42)
        for restart in range(n_restarts):
            if restart == 0:
                init_params = np.ones(n_params)
            else:
                init_params = rng.randn(n_params)

            def loss(params):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        y_pred = equation_fn(X, params)
                        if y_pred is None:
                            return 1e10
                        y_pred = np.asarray(y_pred, dtype=np.float64)
                        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                            return 1e10
                        return float(np.mean((y_pred - y) ** 2))
                    except Exception:
                        return 1e10

            try:
                opt_result = minimize(
                    loss,
                    init_params,
                    method="BFGS",
                    options={"maxiter": bfgs_maxiter, "disp": False},
                )
                if opt_result.fun < best_mse:
                    best_mse = opt_result.fun
                    best_params = opt_result.x.tolist()
                    try:
                        best_y_pred = np.asarray(equation_fn(X, opt_result.x), dtype=np.float64)
                    except Exception:
                        best_y_pred = None
            except Exception:
                continue

        if best_params is not None and np.isfinite(best_mse):
            result["success"] = True
            result["mse"] = float(best_mse)
            result["nmse"] = float(best_mse / y_var)
            result["best_params"] = best_params
            result["y_pred"] = best_y_pred

            # R-squared
            if best_y_pred is not None:
                ss_res = np.sum((y - best_y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                result["r_squared"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            result["error"] = "BFGS optimization failed to converge on all restarts."

    except ExecutionTimeout:
        result["error"] = "Code execution timed out (exceeded 30 seconds)."
    except SyntaxError as e:
        result["error"] = f"Syntax error in generated code: {e}"
    except Exception as e:
        result["error"] = f"Runtime error: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return result


# ──────────────────────────────────────────────
# Reward computation
# ──────────────────────────────────────────────


def compute_reward(
    exec_result: dict,
    code: str,
    n_variables: int,
    alpha: float = 0.1,
    complexity_penalty_coef: float = 0.005,
) -> tuple[float, dict]:
    """
    Compute total reward from execution result.

    Returns:
        (R_total, info_dict)
    """
    info = {
        "r_fit": 0.0,
        "r_valid": 0.0,
        "r_complexity": 0.0,
        "r_total": 0.0,
    }

    # Case 1: Code parsing/execution failed
    if not exec_result["success"] and exec_result["error"]:
        if "Syntax error" in exec_result["error"]:
            info["r_valid"] = -5.0
        elif "timed out" in exec_result["error"]:
            info["r_valid"] = -3.0
        else:
            info["r_valid"] = -5.0
        info["r_total"] = info["r_valid"]
        return info["r_total"], info

    # Case 2: BFGS succeeded
    mse = exec_result["mse"]

    # Fit reward: 1/(1 + alpha * MSE), range [0, 1]
    info["r_fit"] = 1.0 / (1.0 + alpha * mse)

    # Validity check: does the equation use input variables?
    if not check_uses_input(code, n_variables):
        info["r_valid"] = -2.0
    else:
        info["r_valid"] = 0.0

    # Complexity penalty based on AST node count
    node_count = count_ast_nodes(code)
    info["r_complexity"] = -complexity_penalty_coef * max(node_count - 10, 0)  # free budget of 10 nodes

    info["r_total"] = info["r_fit"] + info["r_valid"] + info["r_complexity"]
    return info["r_total"], info


# ──────────────────────────────────────────────
# Diagnostic feedback generation
# ──────────────────────────────────────────────


def generate_feedback(
    exec_result: dict,
    reward_info: dict,
    code: str,
    X: np.ndarray,
    y: np.ndarray,
    variables: list[str],
    target: str,
) -> str:
    """Generate detailed diagnostic feedback for the LLM's next turn."""

    if not exec_result["success"]:
        return (
            f"**Evaluation Failed**\n"
            f"Error: {exec_result['error']}\n\n"
            f"Please fix the code and try again. Make sure:\n"
            f"1. The function is named `equation` and takes (X, params) as arguments\n"
            f"2. Use `params[0]`, `params[1]`, ... for unknown constants\n"
            f"3. Use numpy operations (np.sin, np.exp, etc.)\n"
            f"4. Return a numpy array of shape (N,)"
        )

    mse = exec_result["mse"]
    nmse = exec_result["nmse"]
    r2 = exec_result["r_squared"]
    best_params = exec_result["best_params"]
    y_pred = exec_result["y_pred"]

    lines = [
        f"**Current Equation Evaluation Results**:",
        f"- MSE: {mse:.6e}",
        f"- NMSE (normalized): {nmse:.6e}",
        f"- R² (goodness of fit): {r2:.6f}",
        f"- Optimized parameters: {[f'{p:.4f}' for p in best_params[:6]]}{'...' if len(best_params) > 6 else ''}",
        f"- Reward: {reward_info['r_total']:.4f} (fit={reward_info['r_fit']:.4f}, valid={reward_info['r_valid']:.1f}, complexity={reward_info['r_complexity']:.3f})",
    ]

    # Residual analysis
    if y_pred is not None and not np.any(np.isnan(y_pred)):
        residuals = y - y_pred
        abs_residuals = np.abs(residuals)

        # Find worst-fitting region
        worst_idx = np.argmax(abs_residuals)
        lines.append(f"\n**Residual Analysis**:")
        lines.append(f"- Mean absolute error: {np.mean(abs_residuals):.6e}")
        lines.append(f"- Max absolute error: {abs_residuals[worst_idx]:.6e}")

        # Describe where the worst fit occurs
        worst_point_desc = ", ".join(
            f"{variables[j]}={X[worst_idx, j]:.4f}" for j in range(len(variables))
        )
        lines.append(
            f"- Worst fit at: {worst_point_desc} "
            f"(predicted {target}={y_pred[worst_idx]:.4f}, actual={y[worst_idx]:.4f})"
        )

        # Check systematic patterns
        # Split data into low/high halves of target and check if residuals are biased
        median_y = np.median(y)
        low_mask = y < median_y
        high_mask = y >= median_y
        low_mean_res = np.mean(residuals[low_mask]) if low_mask.sum() > 0 else 0
        high_mean_res = np.mean(residuals[high_mask]) if high_mask.sum() > 0 else 0

        if abs(low_mean_res) > 0.1 * np.std(y) or abs(high_mean_res) > 0.1 * np.std(y):
            lines.append(f"- Systematic bias detected: low-{target} region mean residual={low_mean_res:.4e}, high-{target} region={high_mean_res:.4e}")

        # Variable importance (correlation of residuals with each variable)
        lines.append(f"\n**Residual-Variable Correlations** (hints for improvement):")
        for j, var in enumerate(variables):
            corr = np.corrcoef(X[:, j], abs_residuals)[0, 1]
            if abs(corr) > 0.1:
                lines.append(f"- |residual| correlates with {var}: r={corr:.3f} → consider adding nonlinear terms in {var}")

    lines.append(
        f"\nPlease rethink your approach and provide an improved equation. "
        f"Consider the residual patterns above to guide your modifications."
    )

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main Interaction class
# ──────────────────────────────────────────────


class EquationFittingInteraction(BaseInteraction):
    """
    Multi-turn RL environment for scientific equation discovery.

    Each turn:
    1. LLM generates <think>...</think> + ```python equation code ```
    2. Environment parses, executes, fits via BFGS, computes reward
    3. Returns diagnostic feedback for the next turn
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict: dict[str, dict] = {}

        # Configurable hyperparameters
        self.bfgs_maxiter = config.get("bfgs_maxiter", 100)
        self.n_restarts = config.get("n_restarts", 3)
        self.alpha = config.get("alpha", 0.1)
        self.complexity_penalty_coef = config.get("complexity_penalty_coef", 0.005)
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.early_stop_nmse = config.get("early_stop_nmse", 1e-5)

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        # Extract data from interaction_kwargs
        X_train = np.array(kwargs.get("X_train", []), dtype=np.float64)
        y_train = np.array(kwargs.get("y_train", []), dtype=np.float64)

        variables = kwargs.get("variables", [])
        if isinstance(variables, str):
            variables = json.loads(variables)

        variable_descs = kwargs.get("variable_descs", [])
        if isinstance(variable_descs, str):
            variable_descs = json.loads(variable_descs)

        self._instance_dict[instance_id] = {
            "X_train": X_train,
            "y_train": y_train,
            "variables": variables,
            "variable_descs": variable_descs,
            "target": kwargs.get("target", "y"),
            "target_desc": kwargs.get("target_desc", "target variable"),
            "task_id": kwargs.get("task_id", "unknown"),
            "best_reward": float("-inf"),
            "best_mse": float("inf"),
            "turn": 0,
        }

        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Process the LLM's response and return feedback.

        Returns:
            (should_terminate, feedback_text, reward, extra_data)
        """
        instance = self._instance_dict[instance_id]
        instance["turn"] += 1

        X = instance["X_train"]
        y = instance["y_train"]
        variables = instance["variables"]
        target = instance["target"]

        # Find the last assistant message
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "")
                break

        if not content:
            return True, "No assistant response found.", -5.0, {"error": "no_response"}

        # Extract python code
        code = extract_python_code(content)
        if code is None:
            feedback = (
                "**Error**: Could not find a Python code block in your response.\n"
                "Please wrap your equation function in ```python ... ``` markers.\n"
                "The function must be named `equation` and take (X, params) as arguments."
            )
            return False, feedback, -5.0, {"error": "no_code_block"}

        # Execute and fit
        exec_result = execute_equation_sandboxed(
            code=code,
            X=X,
            y=y,
            max_params=10,
            bfgs_maxiter=self.bfgs_maxiter,
            n_restarts=self.n_restarts,
            timeout_seconds=self.timeout_seconds,
        )

        # Compute reward
        reward, reward_info = compute_reward(
            exec_result=exec_result,
            code=code,
            n_variables=len(variables),
            alpha=self.alpha,
            complexity_penalty_coef=self.complexity_penalty_coef,
        )

        # Track best
        if reward > instance["best_reward"]:
            instance["best_reward"] = reward
        if exec_result["success"] and exec_result["mse"] < instance["best_mse"]:
            instance["best_mse"] = exec_result["mse"]

        # Generate feedback
        feedback = generate_feedback(
            exec_result=exec_result,
            reward_info=reward_info,
            code=code,
            X=X,
            y=y,
            variables=variables,
            target=target,
        )

        # Termination logic
        should_terminate = False
        if exec_result["success"] and exec_result["nmse"] < self.early_stop_nmse:
            should_terminate = True
            feedback = (
                f"Excellent! Your equation achieves NMSE={exec_result['nmse']:.2e}, "
                f"R²={exec_result['r_squared']:.6f}. This is a near-perfect fit!"
            )

        extra_data = {
            "mse": exec_result.get("mse", float("inf")),
            "nmse": exec_result.get("nmse", float("inf")),
            "r_squared": exec_result.get("r_squared", float("-inf")),
            "reward_breakdown": reward_info,
            "turn": instance["turn"],
            "task_id": instance["task_id"],
        }

        return should_terminate, feedback, reward, extra_data

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Return the best reward achieved across all turns."""
        if instance_id in self._instance_dict:
            return self._instance_dict[instance_id]["best_reward"]
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up instance state."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

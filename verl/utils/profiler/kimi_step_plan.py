import os

def profile_spec(step, rank, role):
    root = os.environ.get("KIMI_PROFILE_PLAN_ROOT")
    if not root:
        return None
    step, rank = int(step), int(rank)
    if step not in (12, 13) or not 0 <= rank < 32:
        raise ValueError(f"Invalid task profiling step/rank: {step}/{rank}")
    stack = step == 12
    return {"enabled": step == 13 or rank in (0, 8),
            "level": "level1" if stack else "level0", "with_stack": stack,
            "save_path": f"{root}/step{step}-{'L1-withstack' if stack else 'L0-all32'}/{role}/rank{rank}"}

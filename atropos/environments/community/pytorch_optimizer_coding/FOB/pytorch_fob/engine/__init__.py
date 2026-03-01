from pathlib import Path

# from pytorch_fob.engine.engine import Engine  # Unused import


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent

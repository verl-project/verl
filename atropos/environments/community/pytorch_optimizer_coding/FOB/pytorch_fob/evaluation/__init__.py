from pathlib import Path


def evaluation_path() -> Path:
    return Path(__file__).resolve().parent

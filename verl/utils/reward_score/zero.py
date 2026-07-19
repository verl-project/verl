def compute_score(*args, **kwargs):
    """Return a zero task reward so OPD can train from distillation loss only."""
    return {"score": 0.0}

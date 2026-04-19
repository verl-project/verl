"""Placeholder reward for XR-chest GRPO.

Computes ROUGE-L F-measure between the generated report and the
ground-truth report. This is a weak but non-trivial signal -- good enough
to exercise the GRPO pipeline end-to-end before the RATE-backed reward
is wired up.

Phase-2 plan: replace the body of ``compute_score`` with a call into the
RATE extraction + evaluation pipeline (see
``llama-factory-voio/experiment_docs/eval_stack.md``). The signature below
is intentionally identical to the verl ``custom_reward_function`` contract
so swapping implementations is a one-file change.

Hook via launcher:
    custom_reward_function.path=recipe/xr_chest/reward_xr_report.py
    custom_reward_function.name=compute_score
    data.reward_fn_key=data_source
    reward_model.reward_manager=naive
"""

from typing import Optional

from rouge_score import rouge_scorer

_SCORER: Optional[rouge_scorer.RougeScorer] = None


def _get_scorer() -> rouge_scorer.RougeScorer:
    """Lazy-init the ROUGE scorer once per worker process."""
    global _SCORER
    if _SCORER is None:
        _SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _SCORER


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """ROUGE-L F-measure in ``[0, 1]`` between generated and GT report.

    Args:
        data_source: string identifying the dataset (``"xr_chest"`` here).
            Ignored by this reward, but part of the verl contract.
        solution_str: the rollout text produced by the policy.
        ground_truth: full GT report text from the parquet.
        extra_info: ``{"study_id", "index", "split"}``. Unused by the
            placeholder; phase-2 RATE reward will key on ``study_id``.

    Returns:
        Float reward in ``[0, 1]``.
    """
    assert data_source == "xr_chest", f"unexpected data_source {data_source!r}"

    if not solution_str or not ground_truth:
        return 0.0

    scorer = _get_scorer()
    scores = scorer.score(ground_truth, solution_str)
    return float(scores["rougeL"].fmeasure)

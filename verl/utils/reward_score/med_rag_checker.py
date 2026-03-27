"""
MedRAGChecker: Medical RAG Verification Reward Function

Changes vs original:
  - Added FORMAT PENALTY to final reward computation:
      * -0.5 if <answer> tag is missing entirely
      * -0.3 if answer is present but < 50 chars
      * 0.0  if answer is >= 50 chars (no penalty)
  - format_score now uses graded length-aware scoring from correctness.py
  - All other logic unchanged

Why this matters:
  Without format penalty, the model learns to output 8-token direct answers
  (shortcut behavior) because F1 reward is just as easy to obtain without
  search+check. The penalty forces the model to produce substantive answers
  to avoid negative reward, breaking the shortcut equilibrium.
"""

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

class MedRAGConfig:
    """Configuration for MedRAGChecker, loaded from environment."""

    def __init__(self):
        # NLI Checker
        self.nli_enabled = os.getenv("MEDRAG_NLI_ENABLED", "false").lower() == "true"
        self.nli_model = os.getenv("MEDRAG_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        self.nli_device = os.getenv("MEDRAG_NLI_DEVICE", "cpu")

        # LLM Judge
        self.llm_judge_url = os.getenv("MEDRAG_LLM_JUDGE_URL", "")
        self.llm_judge_model = os.getenv("MEDRAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

        # BioPortal
        self.bioportal_key = os.getenv("MEDRAG_BIOPORTAL_KEY", "")

        # Format penalty settings (NEW)
        self.format_penalty_no_tag   = float(os.getenv("MEDRAG_PENALTY_NO_TAG",   "-0.5"))
        self.format_penalty_short    = float(os.getenv("MEDRAG_PENALTY_SHORT",    "-0.3"))
        self.format_min_answer_chars = int(os.getenv("MEDRAG_MIN_ANSWER_CHARS",   "50"))

        # Weights for score combination
        default_weights = {
            "em": 0.35,
            "f1": 0.15,
            "format": 0.10,
            "nli": 0.20,
            "llm_judge": 0.10,
            "entity": 0.10,
        }
        weights_str = os.getenv("MEDRAG_WEIGHTS", "")
        if weights_str:
            try:
                user_weights = json.loads(weights_str)
                default_weights.update(user_weights)
            except json.JSONDecodeError:
                logger.warning(f"Invalid MEDRAG_WEIGHTS: {weights_str}, using defaults")
        self.weights = default_weights

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def __repr__(self):
        return (
            f"MedRAGConfig(\n"
            f"  nli_enabled={self.nli_enabled}, nli_model={self.nli_model},\n"
            f"  llm_judge_url={self.llm_judge_url or 'disabled'},\n"
            f"  bioportal={'enabled' if self.bioportal_key else 'disabled'},\n"
            f"  format_penalty_no_tag={self.format_penalty_no_tag},\n"
            f"  format_penalty_short={self.format_penalty_short},\n"
            f"  format_min_answer_chars={self.format_min_answer_chars},\n"
            f"  weights={self.weights}\n"
            f")"
        )


# Global config (initialized lazily)
_config: Optional[MedRAGConfig] = None
_bioportal_checker = None


def _get_config() -> MedRAGConfig:
    global _config
    if _config is None:
        _config = MedRAGConfig()
        logger.info(f"MedRAGChecker config: {_config}")
    return _config


def _get_bioportal_checker():
    global _bioportal_checker
    config = _get_config()
    if _bioportal_checker is None and config.bioportal_key:
        from .checkers.bioportal_checker import BioPortalChecker
        _bioportal_checker = BioPortalChecker(api_key=config.bioportal_key)
    return _bioportal_checker


def normalize_reward_output(res):
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        score, details = res
        return float(score), details
    if isinstance(res, dict):
        return float(res.get("score", 0.0)), res
    return float(res), {}


# ============================================================
# Format Penalty (NEW)
# ============================================================

def _compute_format_penalty(solution_str: str, config: MedRAGConfig) -> float:
    """
    Compute format penalty for the final reward.

    Returns a NEGATIVE value (or 0) to be ADDED to the base reward:
      -0.5  if <answer> tag is completely missing
      -0.3  if answer exists but is shorter than min_answer_chars
       0.0  if answer is long enough (no penalty)

    These defaults can be overridden via environment variables:
      MEDRAG_PENALTY_NO_TAG    (default: -0.5)
      MEDRAG_PENALTY_SHORT     (default: -0.3)
      MEDRAG_MIN_ANSWER_CHARS  (default: 50)
    """
    from .checkers.correctness import extract_answer

    has_tag = "<answer>" in solution_str and "</answer>" in solution_str
    if not has_tag:
        return config.format_penalty_no_tag   # e.g. -0.5

    answer = extract_answer(solution_str) or ""
    if len(answer.strip()) < config.format_min_answer_chars:
        return config.format_penalty_short    # e.g. -0.3

    return 0.0   # no penalty


# ============================================================
# Main Scoring Function
# ============================================================

def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    return_details: bool = False,
    **kwargs,
) -> dict:
    """
    Compute multi-dimensional RAG quality score.

    Final reward = weighted_score + format_penalty
    where format_penalty is 0 or negative (see _compute_format_penalty).
    """
    config = _get_config()
    start_time = time.time()

    # Normalize ground truth
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        tgt = ground_truth["target"]
        if isinstance(tgt, str):
            ground_truths = [tgt]
        elif isinstance(tgt, list):
            ground_truths = [str(x) for x in tgt if x is not None]
        else:
            ground_truths = [str(tgt)]
    elif isinstance(ground_truth, str):
        ground_truths = [ground_truth]
    elif isinstance(ground_truth, list):
        ground_truths = [str(g) for g in ground_truth]
    else:
        ground_truths = [str(ground_truth)]

    # Extract context from extra_info
    retrieved_docs = ""
    question = ""
    if extra_info and isinstance(extra_info, dict):
        retrieved_docs = extra_info.get("retrieved_docs", "")
        question = extra_info.get("question", "")

    # ── 1. Correctness (always on) ──
    from .checkers.correctness import (
        compute_em, compute_f1, compute_format_score, extract_answer
    )

    prediction   = extract_answer(solution_str)
    em_score     = compute_em(prediction, ground_truths)
    f1_score     = compute_f1(prediction, ground_truths)
    format_score = compute_format_score(solution_str)   # now length-aware

    scores = {
        "em":        em_score,
        "f1":        f1_score,
        "format":    format_score,
        "nli":       0.0,
        "llm_judge": 0.0,
        "entity":    0.0,
    }
    details = {
        "prediction":    prediction,
        "ground_truths": ground_truths,
    }

    # ── 2. NLI Faithfulness (if enabled) ──
    if config.nli_enabled and retrieved_docs and prediction:
        try:
            from .checkers.nli_checker import compute_nli_score
            nli_result = compute_nli_score(
                premise=retrieved_docs,
                hypothesis=prediction,
                model_name=config.nli_model,
                device=config.nli_device,
            )
            scores["nli"] = nli_result["faithfulness_score"]
            details["nli"] = nli_result
        except Exception as e:
            logger.warning(f"NLI check failed: {e}")
            details["nli_error"] = str(e)

    # ── 3. LLM Judge (if server available) ──
    if config.llm_judge_url and retrieved_docs and prediction:
        try:
            from .checkers.llm_judge import judge_faithfulness_remote
            judge_result = judge_faithfulness_remote(
                documents=retrieved_docs,
                question=question,
                answer=prediction,
                api_url=config.llm_judge_url,
                model_name=config.llm_judge_model,
            )
            scores["llm_judge"] = judge_result["score"]
            details["llm_judge"] = judge_result
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            details["llm_judge_error"] = str(e)

    # ── 4. BioPortal Entity Verification (if API key set) ──
    bioportal = _get_bioportal_checker()
    if bioportal and prediction:
        try:
            entities = []
            if config.llm_judge_url:
                from .checkers.llm_judge import extract_entities_remote
                entities = extract_entities_remote(
                    text=prediction,
                    api_url=config.llm_judge_url,
                    model_name=config.llm_judge_model,
                )
            if not entities:
                annotation = bioportal.annotate_text(prediction)
                entities = [e["label"] for e in annotation.get("entities", [])]
            if entities:
                entity_result = bioportal.verify_entities(entities)
                scores["entity"] = entity_result["verification_score"]
                details["entity"] = entity_result
        except Exception as e:
            logger.warning(f"Entity verification failed: {e}")
            details["entity_error"] = str(e)

    # ── 5. Weighted Base Score ──
    active_weights = {}
    for component, weight in config.weights.items():
        if component in ("em", "f1", "format"):
            active_weights[component] = weight
        elif component == "nli" and config.nli_enabled:
            active_weights[component] = weight
        elif component == "llm_judge" and config.llm_judge_url:
            active_weights[component] = weight
        elif component == "entity" and config.bioportal_key:
            active_weights[component] = weight

    total_weight = sum(active_weights.values())
    if total_weight > 0:
        active_weights = {k: v / total_weight for k, v in active_weights.items()}

    base_score = sum(scores[k] * active_weights.get(k, 0.0) for k in scores)

    # ── 6. Format Penalty (NEW) ──
    # This is the key change: adds a negative penalty for missing/short answers.
    # Without this, the model games the reward by outputting 8-token shortcuts.
    format_penalty = _compute_format_penalty(solution_str, config)
    final_score = base_score + format_penalty

    # Log format penalty for monitoring
    if format_penalty < 0:
        ans_preview = (prediction or "")[:60]
        logger.debug(
            f"[format_penalty] penalty={format_penalty:.1f} "
            f"ans_len={len(prediction or '')} "
            f"ans='{ans_preview}'"
        )

    elapsed = time.time() - start_time
    details["elapsed_ms"]     = round(elapsed * 1000, 2)
    details["active_weights"] = active_weights
    details["format_penalty"] = format_penalty   # visible in logs

    result = {"score": float(final_score)}
    if return_details:
        tool_calls = 0
        if isinstance(extra_info, dict):
            tr = extra_info.get("tool_rewards")
            if isinstance(tr, list):
                tool_calls = len(tr)
        result["details"] = {
            "component_scores": scores,
            "base_score":       base_score,
            "format_penalty":   format_penalty,
            "final_score":      final_score,
            "weights":          config.weights,
            "num_evidence":     len(retrieved_docs) if isinstance(retrieved_docs, list) else int(bool(retrieved_docs)),
            "num_tool_calls":   int(tool_calls),
            "internal_details": details,
        }
    return result


# ============================================================
# Simplified interface for verl compatibility
# ============================================================

def compute_score_simple(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Simple interface that returns just a float score."""
    result = compute_score(solution_str, ground_truth, extra_info, **kwargs)
    return result["score"]

# """
# MedRAGChecker: Medical RAG Verification Reward Function

# This is the main orchestrator that combines all checking modules into a single
# reward function compatible with verl's reward system.

# Architecture:
#     ┌─────────────────────────────────────────────────────┐
#     │                  MedRAGChecker                       │
#     │                                                     │
#     │  1. Correctness  (EM + F1)         — always on      │
#     │  2. Format       (<answer> tags)   — always on      │
#     │  3. NLI          (entailment)      — if model avail │
#     │  4. LLM Judge    (faithfulness)    — if server avail│
#     │  5. BioPortal    (entity check)    — if API key set │
#     │                                                     │
#     │  Final Score = weighted combination                  │
#     └─────────────────────────────────────────────────────┘

# Integration with verl:
#     Drop this into verl/utils/reward_score/med_rag_checker.py
#     Then register in verl/utils/reward_score/__init__.py

# Configuration via environment variables:
#     MEDRAG_NLI_MODEL       — NLI model name (default: DeBERTa-v3-base-mnli)
#     MEDRAG_NLI_DEVICE      — NLI device (default: cpu)
#     MEDRAG_LLM_JUDGE_URL   — LLM judge API URL (default: disabled)
#     MEDRAG_LLM_MODEL       — LLM judge model name
#     MEDRAG_BIOPORTAL_KEY   — BioPortal API key (default: disabled)
#     MEDRAG_WEIGHTS         — JSON string of weight overrides
# """

# import json
# import logging
# import os
# import time
# from typing import Any, Optional

# logger = logging.getLogger(__name__)


# # ============================================================
# # Configuration
# # ============================================================

# class MedRAGConfig:
#     """Configuration for MedRAGChecker, loaded from environment."""

#     def __init__(self):
#         # NLI Checker
#         self.nli_enabled = os.getenv("MEDRAG_NLI_ENABLED", "false").lower() == "true"
#         self.nli_model = os.getenv("MEDRAG_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
#         self.nli_device = os.getenv("MEDRAG_NLI_DEVICE", "cpu")

#         # LLM Judge
#         self.llm_judge_url = os.getenv("MEDRAG_LLM_JUDGE_URL", "")  # Empty = disabled
#         self.llm_judge_model = os.getenv("MEDRAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

#         # BioPortal
#         self.bioportal_key = os.getenv("MEDRAG_BIOPORTAL_KEY", "")  # Empty = disabled

#         # Weights for score combination
#         default_weights = {
#             "em": 0.35,
#             "f1": 0.15,
#             "format": 0.10,
#             "nli": 0.20,
#             "llm_judge": 0.10,
#             "entity": 0.10,
#         }
#         weights_str = os.getenv("MEDRAG_WEIGHTS", "")
#         if weights_str:
#             try:
#                 user_weights = json.loads(weights_str)
#                 default_weights.update(user_weights)
#             except json.JSONDecodeError:
#                 logger.warning(f"Invalid MEDRAG_WEIGHTS: {weights_str}, using defaults")
#         self.weights = default_weights

#         # Normalize weights to sum to 1.0
#         total = sum(self.weights.values())
#         if total > 0:
#             self.weights = {k: v / total for k, v in self.weights.items()}

#     def __repr__(self):
#         return (
#             f"MedRAGConfig(\n"
#             f"  nli_enabled={self.nli_enabled}, nli_model={self.nli_model},\n"
#             f"  llm_judge_url={self.llm_judge_url or 'disabled'},\n"
#             f"  bioportal={'enabled' if self.bioportal_key else 'disabled'},\n"
#             f"  weights={self.weights}\n"
#             f")"
#         )


# # Global config & checkers (initialized lazily)
# _config: Optional[MedRAGConfig] = None
# _bioportal_checker = None
# _initialized = False


# def _get_config() -> MedRAGConfig:
#     global _config
#     if _config is None:
#         _config = MedRAGConfig()
#         logger.info(f"MedRAGChecker config: {_config}")
#     return _config


# def _get_bioportal_checker():
#     global _bioportal_checker
#     config = _get_config()
#     if _bioportal_checker is None and config.bioportal_key:
#         from .checkers.bioportal_checker import BioPortalChecker
#         _bioportal_checker = BioPortalChecker(api_key=config.bioportal_key)
#     return _bioportal_checker

# def normalize_reward_output(res):
#     # Res can be: float | dict(score=...) | (float, dict)
#     if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
#         score, details = res
#         return float(score), details
#     if isinstance(res, dict):
#         return float(res.get("score", 0.0)), res
#     return float(res), {}

# # ============================================================
# # Main Scoring Function
# # ============================================================

# def compute_score(
#     solution_str: str,
#     ground_truth: Any,
#     extra_info: Optional[dict] = None,
#     return_details: bool =False,
#     **kwargs,
# ) -> dict:
#     """
#     Compute multi-dimensional RAG quality score.

#     This is the main entry point, compatible with verl's reward system.

#     Args:
#         solution_str: Model's full output (including tool calls and final answer)
#         ground_truth: Ground truth answer (str or list[str])
#         extra_info: Optional dict with additional context:
#             - retrieved_docs: str — documents retrieved by search tool
#             - question: str — original question
#             - queries: list[str] — search queries the model generated

#     Returns:
#         dict with:
#             - score: float — weighted final score [0, 1]
#             - em, f1, format, nli, llm_judge, entity: individual scores
#             - details: dict with detailed information
#     """
#     config = _get_config()
#     start_time = time.time()

#     # # Normalize ground truth
#     # if isinstance(ground_truth, str):
#     #     ground_truths = [ground_truth]
#     # elif isinstance(ground_truth, list):
#     #     ground_truths = [str(g) for g in ground_truth]
#     # else:
#     #     ground_truths = [str(ground_truth)]


#     if isinstance(ground_truth, dict) and "target" in ground_truth:
#         tgt = ground_truth["target"]
#         if isinstance(tgt, str):
#             ground_truths = [tgt]
#         elif isinstance(tgt, list):
#             ground_truths = [str(x) for x in tgt if x is not None]
#         else:
#             ground_truths = [str(tgt)]
#     elif isinstance(ground_truth, str):
#         ground_truths = [ground_truth]
#     elif isinstance(ground_truth, list):
#         ground_truths = [str(g) for g in ground_truth]
#     else:
#         ground_truths = [str(ground_truth)]
    
#     # Extract context from extra_info
#     retrieved_docs = ""
#     question = ""
#     if extra_info and isinstance(extra_info, dict):
#         retrieved_docs = extra_info.get("retrieved_docs", "")
#         question = extra_info.get("question", "")

#     # ── 1. Correctness (always on) ──
#     from .checkers.correctness import compute_em, compute_f1, compute_format_score, extract_answer

#     prediction = extract_answer(solution_str)
#     em_score = compute_em(prediction, ground_truths)
#     f1_score = compute_f1(prediction, ground_truths)
#     format_score = compute_format_score(solution_str)

#     scores = {
#         "em": em_score,
#         "f1": f1_score,
#         "format": format_score,
#         "nli": 0.0,
#         "llm_judge": 0.0,
#         "entity": 0.0,
#     }
#     details = {
#         "prediction": prediction,
#         "ground_truths": ground_truths,
#     }

#     # ── 2. NLI Faithfulness (if enabled and docs available) ──
#     if config.nli_enabled and retrieved_docs and prediction:
#         try:
#             from .checkers.nli_checker import compute_nli_score
#             nli_result = compute_nli_score(
#                 premise=retrieved_docs,
#                 hypothesis=prediction,
#                 model_name=config.nli_model,
#                 device=config.nli_device,
#             )
#             scores["nli"] = nli_result["faithfulness_score"]
#             details["nli"] = nli_result
#         except Exception as e:
#             logger.warning(f"NLI check failed: {e}")
#             details["nli_error"] = str(e)

#     # ── 3. LLM Judge (if server available and docs available) ──
#     if config.llm_judge_url and retrieved_docs and prediction:
#         try:
#             from .checkers.llm_judge import judge_faithfulness_remote
#             judge_result = judge_faithfulness_remote(
#                 documents=retrieved_docs,
#                 question=question,
#                 answer=prediction,
#                 api_url=config.llm_judge_url,
#                 model_name=config.llm_judge_model,
#             )
#             scores["llm_judge"] = judge_result["score"]
#             details["llm_judge"] = judge_result
#         except Exception as e:
#             logger.warning(f"LLM judge failed: {e}")
#             details["llm_judge_error"] = str(e)

#     # ── 4. BioPortal Entity Verification (if API key set) ──
#     bioportal = _get_bioportal_checker()
#     if bioportal and prediction:
#         try:
#             # Extract entities using LLM if judge server is available,
#             # otherwise fall back to BioPortal's own annotator
#             entities = []
#             if config.llm_judge_url:
#                 from .checkers.llm_judge import extract_entities_remote
#                 entities = extract_entities_remote(
#                     text=prediction,
#                     api_url=config.llm_judge_url,
#                     model_name=config.llm_judge_model,
#                 )
            
#             if not entities:
#                 # Fallback: use BioPortal annotator
#                 annotation = bioportal.annotate_text(prediction)
#                 entities = [e["label"] for e in annotation.get("entities", [])]

#             if entities:
#                 entity_result = bioportal.verify_entities(entities)
#                 scores["entity"] = entity_result["verification_score"]
#                 details["entity"] = entity_result
#         except Exception as e:
#             logger.warning(f"Entity verification failed: {e}")
#             details["entity_error"] = str(e)

#     # ── 5. Compute Weighted Final Score ──
#     # Only use weights for components that are actually enabled
#     active_weights = {}
#     for component, weight in config.weights.items():
#         if component in ("em", "f1", "format"):
#             # Always active
#             active_weights[component] = weight
#         elif component == "nli" and config.nli_enabled:
#             active_weights[component] = weight
#         elif component == "llm_judge" and config.llm_judge_url:
#             active_weights[component] = weight
#         elif component == "entity" and config.bioportal_key:
#             active_weights[component] = weight

#     # Re-normalize active weights
#     total_weight = sum(active_weights.values())
#     if total_weight > 0:
#         active_weights = {k: v / total_weight for k, v in active_weights.items()}

#     final_score = sum(scores[k] * active_weights.get(k, 0.0) for k in scores)

#     elapsed = time.time() - start_time
#     details["elapsed_ms"] = round(elapsed * 1000, 2)
#     details["active_weights"] = active_weights

#     # return {
#     #     "score": final_score,
#     #     **scores,
#     #     "details": details,
#     # }
#     # result = {
#     #     "score": float(total_score),
#     #     "em": float(em),
#     #     "f1": float(f1),
#     #     "format_score": float(format_score),
#     #     # 下面这些也都保持 float
#     #     "correctness": float(correctness_score),
#     #     "nli": float(nli_score),
#     #     "llm": float(llm_score),
#     #     "bioportal": float(bio_score),
#     # }
#     # if return_details:
#     #     result["details"] = details   # 仅调试时需要
#     result = {"score": float(final_score)}
#     if return_details:
#         tool_calls = 0
#         if isinstance(extra_info, dict):
#             tr = extra_info.get("tool_rewards")
#             if isinstance(tr, list):
#                 tool_calls = len(tr)
#         result["details"] = {
#             "component_scores": scores,
#             "weights": config.weights,
#             "num_evidence": len(retrieved_docs) if isinstance(retrieved_docs, list) else int(bool(retrieved_docs)),
#             "num_tool_calls": int(tool_calls),
#             "internal_details": details,
#         }
#     return result


# # ============================================================
# # Simplified interface for verl compatibility
# # ============================================================

# def compute_score_simple(
#     solution_str: str,
#     ground_truth: Any,
#     extra_info: Optional[dict] = None,
#     **kwargs,
# ) -> float:
#     """
#     Simple interface that returns just a float score.
#     Use this if verl expects a float rather than a dict.
#     """
#     result = compute_score(solution_str, ground_truth, extra_info, **kwargs)
#     return result["score"]

"""
NLI-based faithfulness checker.

Uses a Natural Language Inference model to check whether the retrieved documents
entail the generated answer. This is a key component for verifying that the model's
answer is grounded in the evidence.

Supports models like:
- microsoft/deberta-v2-xlarge-mnli
- MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- Any HuggingFace NLI model with {entailment, neutral, contradiction} labels
"""

import logging
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_nli_model = None
_nli_tokenizer = None
_nli_device = None


def load_nli_model(
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device: str = "cpu",
):
    """
    Load NLI model (cached globally so it's only loaded once per process).

    Args:
        model_name: HuggingFace model name/path for NLI
        device: Device to load model on ("cpu", "cuda", "cuda:0", etc.)
    """
    global _nli_model, _nli_tokenizer, _nli_device

    if _nli_model is not None:
        return _nli_model, _nli_tokenizer, _nli_device

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info(f"Loading NLI model: {model_name} on {device}")
    _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _nli_model.eval()
    _nli_model.to(device)
    _nli_device = device

    logger.info(f"NLI model loaded. Labels: {_nli_model.config.id2label}")
    return _nli_model, _nli_tokenizer, _nli_device


def get_entailment_label_id(model) -> int:
    """Find the label index for 'entailment' in the model's config."""
    id2label = model.config.id2label
    for idx, label in id2label.items():
        if label.lower() in ("entailment", "entail"):
            return int(idx)
    # Fallback: assume index 0 or 2 depending on common conventions
    # MNLI convention: 0=contradiction, 1=neutral, 2=entailment
    return 2


def get_contradiction_label_id(model) -> int:
    """Find the label index for 'contradiction' in the model's config."""
    id2label = model.config.id2label
    for idx, label in id2label.items():
        if label.lower() in ("contradiction", "contradict"):
            return int(idx)
    return 0


def _chunk_text(text: str, max_chunk_len: int = 500) -> list[str]:
    """
    Split text into chunks for NLI (NLI models have limited input length).
    Splits by sentences/paragraphs rather than hard truncation.
    """
    # Split by double newline (paragraphs), then by single newline, then by periods
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_len:
            current_chunk += " " + para
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If no chunks were created, just truncate
    if not chunks:
        chunks = [text[:max_chunk_len]]

    return chunks


def compute_nli_score(
    premise: str,
    hypothesis: str,
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device: str = "cpu",
    max_premise_chunks: int = 5,
) -> dict:
    """
    Compute NLI-based faithfulness score.

    The premise is the retrieved document(s), the hypothesis is the model's answer.
    If the documents entail the answer, the answer is considered faithful.

    Args:
        premise: Retrieved documents text (evidence)
        hypothesis: Model's generated answer (claim)
        model_name: NLI model to use
        device: Device for inference
        max_premise_chunks: Maximum number of premise chunks to evaluate

    Returns:
        dict with:
            - entailment_score: float [0, 1] — max entailment prob across chunks
            - contradiction_score: float [0, 1] — max contradiction prob across chunks
            - nli_label: str — predicted label for best chunk
            - faithfulness_score: float [0, 1] — final score (entail - contradict)
    """
    if not premise or not hypothesis:
        return {
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "nli_label": "unknown",
            "faithfulness_score": 0.0,
        }

    model, tokenizer, device = load_nli_model(model_name, device)
    entail_id = get_entailment_label_id(model)
    contradict_id = get_contradiction_label_id(model)

    # Chunk the premise (retrieved docs can be long)
    premise_chunks = _chunk_text(premise)[:max_premise_chunks]

    best_entail = 0.0
    best_contradict = 0.0
    best_label = "neutral"

    with torch.no_grad():
        for chunk in premise_chunks:
            inputs = tokenizer(
                chunk,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

            entail_prob = probs[entail_id].item()
            contradict_prob = probs[contradict_id].item()

            if entail_prob > best_entail:
                best_entail = entail_prob
                pred_id = probs.argmax().item()
                best_label = model.config.id2label[pred_id]

            best_contradict = max(best_contradict, contradict_prob)

    # Faithfulness score: high entailment and low contradiction is good
    faithfulness = max(0.0, best_entail - best_contradict)

    return {
        "entailment_score": best_entail,
        "contradiction_score": best_contradict,
        "nli_label": best_label,
        "faithfulness_score": faithfulness,
    }


def batch_compute_nli_score(
    premises: list[str],
    hypotheses: list[str],
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device: str = "cpu",
) -> list[dict]:
    """Batch version for efficiency during training."""
    results = []
    for premise, hypothesis in zip(premises, hypotheses):
        result = compute_nli_score(premise, hypothesis, model_name, device)
        results.append(result)
    return results

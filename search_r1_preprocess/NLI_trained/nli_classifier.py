"""
PubMedBERT-MedNLI checker — drop-in wrapper matching the interface used by
checker_medrag_gpt4omini.py (extract_claims / verify_claim / set_question).

Labels returned are the literal strings "entail" / "neutral" / "contradict".
They MUST match the strings the /check endpoint counts when computing
support_rate / contradiction_rate.  Do not change the spelling.

Default index mapping (produced by finetune_pubmedbert_mednli.py which sets
LABEL2ID = {entailment:0, neutral:1, contradiction:2}):
    0 -> entail
    1 -> neutral
    2 -> contradict
Override by passing label_map={...} if you fine-tune with a different order.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


_VALID_LABELS = {"entail", "neutral", "contradict"}
DEFAULT_LABEL_MAP: Dict[int, str] = {0: "entail", 1: "neutral", 2: "contradict"}

# Sentence-level splitter; tolerates Chinese full-stop because some rollouts
# emit Chinese during language-drift shortcuts.
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!。！？])\s*")


def _split_claims(answer: str, max_claims: int = 8) -> List[str]:
    if not answer:
        return []
    answer = re.sub(r"</?answer>", "", answer).strip()
    parts = [p.strip() for p in _SENT_SPLIT.split(answer) if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts[:max_claims]


class PubMedBERTMedNLIChecker:
    """
    Interface (matches GPTCheckerPipeline / StudentCheckerLM / LocalNLIChecker
    in checker_medrag_gpt4omini.py):
        .set_question(question)                  # optional, no-op here
        .extract_claims(answer)   -> list[str]
        .verify_claim(claim, evidence)
                                 -> {"label": "entail"|"neutral"|"contradict",
                                     "confidence": float in [0,1]}
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 256,
        label_map: Optional[Dict[int, str]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.label_map = dict(label_map) if label_map else dict(DEFAULT_LABEL_MAP)

        # Sanity-check: the strings must match what the checker server counts.
        bad = set(self.label_map.values()) - _VALID_LABELS
        if bad:
            raise ValueError(
                f"label_map values must be a subset of {_VALID_LABELS}; "
                f"got unexpected labels {bad}"
            )

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_path)
            .to(self.device)
            .eval()
        )
        self._question: str = ""

    # ---- server-expected interface -----------------------------------------

    def set_question(self, question: str) -> None:
        self._question = question or ""

    def extract_claims(self, answer: str) -> List[str]:
        return _split_claims(answer)

    @torch.inference_mode()
    def verify_claim(self, claim: str, evidence: str) -> Dict[str, Any]:
        if not claim.strip():
            return {"label": "neutral", "confidence": 0.0}

        # premise = retrieved evidence, hypothesis = claim from the answer.
        enc = self.tok(
            evidence or "",
            claim,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1).tolist()
        top_idx = max(range(len(probs)), key=lambda i: probs[i])
        label = self.label_map.get(top_idx, "neutral")
        return {
            "label": label,
            "confidence": round(float(probs[top_idx]), 4),
            "scores": {self.label_map[i]: round(float(p), 4)
                       for i, p in enumerate(probs) if i in self.label_map},
        }


if __name__ == "__main__":
    # Quick CLI smoke test.
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args()
    c = PubMedBERTMedNLIChecker(args.model_path)
    claims = c.extract_claims(args.answer)
    print("claims:", claims)
    for cl in claims:
        print(cl, "->", json.dumps(c.verify_claim(cl, args.evidence)))

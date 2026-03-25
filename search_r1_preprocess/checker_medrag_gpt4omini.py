#!/usr/bin/env python3
"""
Checker server for Search + Checker tool training.

Contract (unchanged from original):
  POST /check  {"answer": "...", "evidence": "...", "question": "..."}
  Returns NLI labels: entail / contradict / neutral

Modes:
  --mode openai          GPT-4o-mini, full pipeline (extract + verify)   [recommended]
  --mode hybrid          GPT-4o-mini extraction + local Meditron verify  [cheaper]
  --mode student_pipeline  local extractor + local checker              [no API cost]
  --mode local_nli       DeBERTa MNLI                                    [baseline]
  --mode mock            always returns entail (for smoke tests)

Usage examples:

  # Recommended: GPT-4o-mini full pipeline
  python checker_medrag.py \
      --mode openai \
      --model_name gpt-4o-mini \
      --max_claims 2 \
      --max_concurrent 4 \
      --max_queue 20 \
      --port 8004

  # Hybrid: GPT extraction + local Meditron checker
  CUDA_VISIBLE_DEVICES=2 python checker_medrag.py \
      --mode hybrid \
      --checker_model_path /path/to/checker_sft_balanced_v1 \
      --checker_base_model_path /path/to/Meditron3-8B \
      --max_claims 2 \
      --port 8004

  # Original: local student pipeline
  CUDA_VISIBLE_DEVICES=2 python checker_medrag.py \
      --mode student_pipeline \
      --checker_model_path /path/to/checker_sft_balanced_v1 \
      --checker_base_model_path /path/to/Meditron3-8B \
      --extractor_model_path /path/to/extractor_sft_meditron3-8b \
      --extractor_base_model_path /path/to/Meditron3-8B \
      --max_claims 2 \
      --checker_fast_mode \
      --max_concurrent 1 \
      --max_queue 20 \
      --port 8004

  # Health check
  curl -s http://127.0.0.1:8004/health

  # Manual test
  curl -s http://127.0.0.1:8004/check \
    -H 'Content-Type: application/json' \
    -d '{"question":"What is first-line treatment for type 2 diabetes?",
         "answer":"Metformin is first-line unless contraindicated.",
         "evidence":"ADA guidelines recommend metformin as initial pharmacologic therapy."}'
"""

import argparse
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MedRAGChecker API")

_inference_semaphore: threading.Semaphore = threading.Semaphore(1)
_active_count: int = 0
_active_lock: threading.Lock = threading.Lock()
_MAX_QUEUE_DEPTH: int = 20


# =============================================================================
# Request / Response models (unchanged)
# =============================================================================

class CheckRequest(BaseModel):
    answer:   str
    evidence: Optional[str] = ""
    question: Optional[str] = ""


class ClaimResult(BaseModel):
    claim:      str
    label:      str       # entail | contradict | neutral
    confidence: float


class CheckResponse(BaseModel):
    claims:               list[ClaimResult]
    verification_results: list[ClaimResult]
    num_claims:           int
    num_supported:        int
    num_contradicted:     int
    num_neutral:          int
    support_rate:         float
    contradiction_rate:   float


# =============================================================================
# Label normalisation helpers
# =============================================================================

def _normalize_label(text: str) -> str:
    t = text.lower()
    if "contradict" in t or "refut" in t or "oppos" in t:
        return "contradict"
    if "support" in t or "entail" in t or "confirm" in t:
        return "entail"
    return "neutral"


def _extract_confidence(text: str, default: float = 0.7) -> float:
    m = re.search(r"(?:confidence|score)\s*[:=]?\s*([01](?:\.\d+)?)", text.lower())
    if not m:
        return default
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        return default


# =============================================================================
# GPT-4o-mini checker (full pipeline)  ← IMPROVED prompts, forces non-neutral
# =============================================================================

class GPTCheckerPipeline:
    """
    GPT-4o-mini as both claim extractor and NLI verifier.
    Prompts are designed to minimise neutral output — checker prefers
    SUPPORTED or CONTRADICTED when evidence is topically related.
    Output labels: entail / contradict / neutral  (same as original contract)
    """

    def __init__(self, model_name: str = "gpt-4o-mini", max_claims: int = 2):
        from openai import OpenAI
        self.client     = OpenAI()
        self.model_name = model_name
        self.max_claims = max(1, max_claims)
        self._last_question = ""

    def set_question(self, question: str) -> None:
        self._last_question = question or ""

    def extract_claims(self, answer: str) -> list[str]:
        if not answer.strip():
            return []
        prompt = (
            f"Extract {self.max_claims} key atomic factual claims from this medical answer.\n\n"
            "Rules:\n"
            "- Each claim must be a concrete, verifiable medical fact\n"
            "- One claim per line, no numbering, no preamble\n"
            "- Focus on: treatments, drugs, doses, diagnoses, contraindications\n"
            "- Skip vague phrases like 'consult a doctor' or 'it depends'\n"
            f"- Output exactly {self.max_claims} line(s)\n\n"
            f"Question: {self._last_question}\n"
            f"Answer: {answer}\n\n"
            "Claims:"
        )
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            text = (res.choices[0].message.content or "").strip()
            claims = []
            for line in text.splitlines():
                line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                if len(line) > 10:
                    claims.append(line)
            return claims[:self.max_claims] or self._fallback_claims(answer)
        except Exception:
            return self._fallback_claims(answer)

    def _fallback_claims(self, answer: str) -> list[str]:
        parts = re.split(r"[.!?]+", answer)
        return [p.strip() for p in parts if len(p.strip()) > 20][:self.max_claims]

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        if not evidence.strip():
            return {"label": "neutral", "confidence": 0.3}

        prompt = (
            "You are a medical fact-checker. Decide if EVIDENCE supports or contradicts CLAIM.\n\n"
            "Decision rules:\n"
            "1. If evidence directly or indirectly supports the claim → SUPPORTED\n"
            "2. If evidence clearly contradicts or is inconsistent with the claim → CONTRADICTED\n"
            "3. NEUTRAL only if the evidence is completely unrelated to the claim topic\n"
            "   (do NOT use NEUTRAL just because the evidence is incomplete)\n\n"
            "Return exactly:\n"
            "Label: <SUPPORTED|CONTRADICTED|NEUTRAL>\n"
            "Confidence: <0.0-1.0>\n\n"
            f"EVIDENCE:\n{evidence}\n\n"
            f"CLAIM:\n{claim}\n"
        )
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            text = (res.choices[0].message.content or "").strip()
            return {
                "label":      _normalize_label(text),
                "confidence": _extract_confidence(text, default=0.75),
            }
        except Exception:
            return {"label": "neutral", "confidence": 0.3}


# =============================================================================
# Hybrid: GPT-4o-mini extraction + local Meditron checker
# =============================================================================

class HybridCheckerPipeline:
    """
    GPT-4o-mini for claim extraction (quality) +
    local Meditron-based StudentCheckerLM for NLI (no verification API cost).
    Best cost/quality balance when local checker has ≥ 80% NLI accuracy.
    """

    def __init__(
        self,
        checker_model_path:      str,
        checker_base_model_path: Optional[str] = None,
        model_name:              str = "gpt-4o-mini",
        max_claims:              int = 2,
        checker_fast_mode:       bool = True,
    ):
        from openai import OpenAI
        self.client     = OpenAI()
        self.model_name = model_name
        self.max_claims = max(1, max_claims)
        self._last_question = ""
        self.checker = StudentCheckerLM(
            model_path      = checker_model_path,
            base_model_path = checker_base_model_path,
            fast_mode       = checker_fast_mode,
        )

    def set_question(self, question: str) -> None:
        self._last_question = question or ""

    def extract_claims(self, answer: str) -> list[str]:
        """GPT-4o-mini extraction — same prompt as GPTCheckerPipeline."""
        if not answer.strip():
            return []
        prompt = (
            f"Extract {self.max_claims} key atomic factual claims from this medical answer.\n\n"
            "Rules:\n"
            "- One claim per line, no numbering\n"
            "- Concrete verifiable medical facts only\n"
            "- Focus on: treatments, drugs, dosing, contraindications\n"
            f"- Output exactly {self.max_claims} line(s)\n\n"
            f"Question: {self._last_question}\n"
            f"Answer: {answer}\n\nClaims:"
        )
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            text = (res.choices[0].message.content or "").strip()
            claims = []
            for line in text.splitlines():
                line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                if len(line) > 10:
                    claims.append(line)
            return claims[:self.max_claims] or self._fallback_claims(answer)
        except Exception:
            return self._fallback_claims(answer)

    def _fallback_claims(self, answer: str) -> list[str]:
        parts = re.split(r"[.!?]+", answer)
        return [p.strip() for p in parts if len(p.strip()) > 20][:self.max_claims]

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        """Local Meditron checker for NLI."""
        return self.checker.verify_claim(claim, evidence)


# =============================================================================
# Local StudentCheckerLM (unchanged from original, thread-safe)
# =============================================================================

class StudentCheckerLM:
    def __init__(
        self,
        model_path:      str,
        base_model_path: Optional[str] = None,
        fast_mode:       bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        mp            = Path(model_path)
        is_lora       = (mp / "adapter_config.json").exists()

        if is_lora:
            if base_model_path is None:
                with open(mp / "adapter_config.json") as f:
                    base_model_path = json.load(f).get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError("LoRA adapter detected but base_model_path missing.")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(model, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.fast_mode = fast_mode
        self._lock = threading.Lock()

    @staticmethod
    def extract_claims(answer: str) -> list[str]:
        if not answer.strip():
            return []
        parts = re.split(r"[.!?。！？]+", answer)
        return [p.strip() for p in parts if len(p.strip()) > 15][:8]

    @staticmethod
    def _prompt(claim: str, evidence: str) -> str:
        return (
            "Decide whether the EVIDENCE entails, contradicts, or is neutral to the CLAIM.\n"
            "Respond with one of: entailed | contradicted | neutral\n\n"
            f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}\n\nLabel:"
        )

    def _score_label(self, prompt: str, label_text: str) -> float:
        import torch
        import torch.nn.functional as F
        p = self.tokenizer(prompt,               return_tensors="pt", add_special_tokens=False)
        x = self.tokenizer(prompt + label_text,  return_tensors="pt", add_special_tokens=False)
        input_ids  = x["input_ids"].to(self.model.device)
        prompt_len = p["input_ids"].shape[1]
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[:, :-1, :]
            target = input_ids[:, 1:]
            tok_lp = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        comp = tok_lp[:, max(prompt_len - 1, 0):]
        return float(comp.mean().item()) if comp.numel() > 0 else -1e9

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        import torch
        if not evidence.strip():
            return {"label": "neutral", "confidence": 0.5}

        prompt = self._prompt(claim, evidence)
        candidates = {
            "entail":     [" entailed"] if self.fast_mode else [" entailed", " supported", " entail"],
            "contradict": [" contradicted"] if self.fast_mode else [" contradicted", " refuted", " contradict"],
            "neutral":    [" neutral"]  if self.fast_mode else [" neutral", " unknown", " insufficient"],
        }
        scores: dict[str, float] = {}
        with self._lock:
            for out_label, variants in candidates.items():
                scores[out_label] = max(self._score_label(prompt, v) for v in variants)

        pred  = max(scores, key=scores.get)
        probs = torch.softmax(
            torch.tensor([scores["entail"], scores["contradict"], scores["neutral"]], dtype=torch.float32), dim=0
        )
        return {"label": pred, "confidence": round(float(probs.max().item()), 4), "scores": scores}


# =============================================================================
# Student extractor + checker pipeline (unchanged)
# =============================================================================

def _parse_triples_json(text: str) -> list[list[str]]:
    if not text: return []
    s = text.replace("```json", "").replace("```", "").strip()
    seen: set = set()
    for cand in sorted({s} | {
        s[m.start(): next((i for i in range(len(s)-1, m.start()-1, -1) if s[i] == "]"), len(s)) + 1]
        for m in re.finditer(r"\[", s)
    }, key=len, reverse=True):
        if cand in seen: continue
        seen.add(cand)
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if not isinstance(data, list): continue
        triples = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                triples.append([str(x) for x in item])
            elif isinstance(item, dict):
                s2 = item.get("subject") or item.get("subj") or item.get("s")
                r  = item.get("relation") or item.get("predicate") or item.get("rel") or item.get("p")
                o  = item.get("object") or item.get("obj") or item.get("o")
                if s2 is not None and r is not None and o is not None:
                    triples.append([str(s2), str(r), str(o)])
        if triples: return triples
    return []


class StudentExtractorLM:
    def __init__(self, model_path: str, base_model_path: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        mp = Path(model_path)
        is_lora = (mp / "adapter_config.json").exists()
        if is_lora:
            if base_model_path is None:
                with open(mp / "adapter_config.json") as f:
                    base_model_path = json.load(f).get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError("Extractor LoRA adapter detected but base_model_path missing.")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(model, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self._lock = threading.Lock()

    @staticmethod
    def _build_prompt(question: str, answer: str) -> str:
        return (
            "You are an information extraction assistant. "
            "Given a medical question and its answer, extract ALL factual triples "
            "as [subject, relation, object]. "
            "Copy entity names EXACTLY. Return pure JSON array, no explanations.\n\n"
            f"Question: {question.strip()}\nAnswer: {answer.strip()}\n\n"
            'Triples (JSON only, e.g. [["subj", "rel", "obj"], ...]):\n'
        )

    def extract_triples(self, question: str, answer: str) -> list[list[str]]:
        import torch
        prompt = self._build_prompt(question, answer)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            vocab_size = int(self.model.get_input_embeddings().weight.shape[0])
            bad_mask = (input_ids < 0) | (input_ids >= vocab_size)
            if bad_mask.any():
                unk_id = self.tokenizer.unk_token_id or self.tokenizer.eos_token_id or 0
                input_ids = input_ids.clone(); input_ids[bad_mask] = int(unk_id)
                inputs["input_ids"] = input_ids
        try:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self._lock:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=256, do_sample=False, temperature=0.0,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            generated = self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            return _parse_triples_json(generated)
        except RuntimeError as e:
            print(f"[StudentExtractorLM] failed: {e}")
            return []


class StudentExtractorCheckerPipeline:
    def __init__(
        self,
        checker_model_path:      str,
        extractor_model_path:    str,
        checker_base_model_path:  Optional[str] = None,
        extractor_base_model_path:Optional[str] = None,
        max_claims:              int = 2,
        checker_fast_mode:       bool = True,
    ):
        self.extractor = StudentExtractorLM(extractor_model_path, base_model_path=extractor_base_model_path)
        self.checker   = StudentCheckerLM(checker_model_path,
                                          base_model_path=checker_base_model_path,
                                          fast_mode=checker_fast_mode)
        self._last_question = ""
        self.max_claims     = max(1, int(max_claims))

    def set_question(self, question: str) -> None:
        self._last_question = question or ""

    def extract_claims(self, answer: str) -> list[str]:
        triples = self.extractor.extract_triples(self._last_question, answer)
        claims = []
        for t in triples:
            if len(t) != 3: continue
            s, r, o = t
            # Filter trivial triples
            if len(s) < 3 or len(o) < 3: continue
            if r.lower().strip() in {"is", "are", "has", "have", "be", "was", "were"}: continue
            claim = f"{s} {r} {o}".strip()
            if len(claim) > 10: claims.append(claim)
        if claims:
            return claims[:self.max_claims]
        # Fallback: sentence split
        parts = re.split(r"[.!?]+", answer)
        return [p.strip() for p in parts if len(p.strip()) > 20][:self.max_claims]

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        return self.checker.verify_claim(claim, evidence)


# =============================================================================
# Mock checker (smoke tests)
# =============================================================================

class MockChecker:
    @staticmethod
    def extract_claims(answer: str) -> list[str]:
        return [answer.strip()[:200]] if answer.strip() else []

    @staticmethod
    def verify_claim(claim: str, evidence: str) -> dict[str, Any]:
        if not evidence.strip():
            return {"label": "neutral", "confidence": 0.5}
        return {"label": "entail", "confidence": 0.6}


# =============================================================================
# Local DeBERTa MNLI checker
# =============================================================================

class LocalNLIChecker:
    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model.eval()
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.label_map = {0: "contradict", 1: "neutral", 2: "entail"}

    @staticmethod
    def extract_claims(answer: str) -> list[str]:
        if not answer.strip(): return []
        parts = re.split(r"[.!?。！？]+", answer)
        return [p.strip() for p in parts if len(p.strip()) > 15 and len(p.split()) >= 4][:8]

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        import torch
        if not evidence.strip():
            return {"label": "neutral", "confidence": 0.5}
        inputs = self.tokenizer(evidence, claim, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)
            pred  = int(torch.argmax(probs, dim=-1).item())
            conf  = float(probs[0, pred].item())
        return {"label": self.label_map[pred], "confidence": round(conf, 4)}


# =============================================================================
# FastAPI endpoint (unchanged contract)
# =============================================================================

checker: Optional[Any] = None
import time as _time


@app.post("/check", response_model=CheckResponse)
def check_answer(request: CheckRequest):
    global _active_count
    assert checker is not None, "Checker not initialized"

    with _active_lock:
        if _active_count >= _MAX_QUEUE_DEPTH:
            raise HTTPException(503, f"Checker overloaded (queue={_active_count}/{_MAX_QUEUE_DEPTH}).")
        _active_count += 1

    t_arrive = _time.time()
    try:
        acquired = _inference_semaphore.acquire(timeout=30)
        if not acquired:
            raise HTTPException(503, "Checker semaphore timeout (30s).")
        t_gpu = _time.time()
        try:
            if hasattr(checker, "set_question"):
                try: checker.set_question(request.question or "")
                except Exception: pass
            try:    claims_text = checker.extract_claims(request.answer)
            except Exception as e:
                print(f"[checker] extract_claims failed: {e}")
                claims_text = []

            results: list[ClaimResult] = []
            for claim in claims_text:
                try:    v = checker.verify_claim(claim, request.evidence or "")
                except Exception as e:
                    print(f"[checker] verify_claim failed: {e}")
                    v = {"label": "neutral", "confidence": 0.0}
                results.append(ClaimResult(
                    claim=claim,
                    label=str(v.get("label", "neutral")),
                    confidence=float(v.get("confidence", 0.0)),
                ))

            t_done = _time.time()
            print(f"[checker] wait={t_gpu-t_arrive:.1f}s gpu={t_done-t_gpu:.1f}s "
                  f"claims={len(results)} active={_active_count} "
                  f"labels={[r.label for r in results]}")
        finally:
            _inference_semaphore.release()
    finally:
        with _active_lock:
            _active_count -= 1

    n    = len(results)
    nsup = sum(1 for r in results if r.label == "entail")
    ncon = sum(1 for r in results if r.label == "contradict")
    nneu = sum(1 for r in results if r.label == "neutral")
    return CheckResponse(
        claims=results, verification_results=results,
        num_claims=n, num_supported=nsup, num_contradicted=ncon, num_neutral=nneu,
        support_rate=round(nsup/n, 4) if n else 0.0,
        contradiction_rate=round(ncon/n, 4) if n else 0.0,
    )


@app.get("/health")
def health():
    return {"status": "ok", "checker_type": type(checker).__name__ if checker else "uninitialized"}


# =============================================================================
# Main
# =============================================================================

def main():
    global checker, _inference_semaphore, _MAX_QUEUE_DEPTH

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["openai","hybrid","student_pipeline","student_checker","local_nli","mock"],
        default="openai")
    parser.add_argument("--model_name",               type=str, default="gpt-4o-mini")
    parser.add_argument("--base_model_path",           type=str, default=None)
    parser.add_argument("--checker_model_path",        type=str, default=None)
    parser.add_argument("--checker_base_model_path",   type=str, default=None)
    parser.add_argument("--extractor_model_path",      type=str, default=None)
    parser.add_argument("--extractor_base_model_path", type=str, default=None)
    parser.add_argument("--max_claims",       type=int,  default=2)
    parser.add_argument("--checker_fast_mode",action="store_true", default=True)
    parser.add_argument("--port",             type=int,  default=8004)
    parser.add_argument("--host",             type=str,  default="0.0.0.0")
    parser.add_argument("--max_concurrent",   type=int,  default=1)
    parser.add_argument("--max_queue",        type=int,  default=20)
    args = parser.parse_args()

    _inference_semaphore = threading.Semaphore(args.max_concurrent)
    _MAX_QUEUE_DEPTH     = args.max_queue

    if args.mode == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY required for --mode openai")
        checker = GPTCheckerPipeline(model_name=args.model_name, max_claims=args.max_claims)

    elif args.mode == "hybrid":
        if not args.checker_model_path:
            raise ValueError("--mode hybrid requires --checker_model_path")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY required for --mode hybrid (GPT extraction)")
        checker = HybridCheckerPipeline(
            checker_model_path      = args.checker_model_path,
            checker_base_model_path = args.checker_base_model_path,
            model_name              = args.model_name,
            max_claims              = args.max_claims,
            checker_fast_mode       = args.checker_fast_mode,
        )

    elif args.mode == "student_pipeline":
        if not args.checker_model_path or not args.extractor_model_path:
            raise ValueError("--mode student_pipeline requires --checker_model_path and --extractor_model_path")
        checker = StudentExtractorCheckerPipeline(
            checker_model_path        = args.checker_model_path,
            extractor_model_path      = args.extractor_model_path,
            checker_base_model_path   = args.checker_base_model_path,
            extractor_base_model_path = args.extractor_base_model_path,
            max_claims                = args.max_claims,
            checker_fast_mode         = args.checker_fast_mode,
        )

    elif args.mode == "student_checker":
        if not args.model_name and not args.checker_model_path:
            raise ValueError("--mode student_checker requires --model_name or --checker_model_path")
        checker = StudentCheckerLM(
            model_path      = args.checker_model_path or args.model_name,
            base_model_path = args.checker_base_model_path or args.base_model_path,
            fast_mode       = args.checker_fast_mode,
        )

    elif args.mode == "local_nli":
        checker = LocalNLIChecker(nli_model_name=args.model_name)

    else:
        checker = MockChecker()

    print(f"Starting checker server http://{args.host}:{args.port} "
          f"mode={args.mode} max_concurrent={args.max_concurrent} "
          f"max_queue={args.max_queue} max_claims={args.max_claims}")

    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=5)


if __name__ == "__main__":
    main()

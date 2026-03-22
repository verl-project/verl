#!/usr/bin/env python3
"""
Checker server for Search + Checker tool training.

Compatible with checker tool contract:
- POST /check with {"answer": "...", "evidence": "...", "question": "..."}
- Returns claim-level labels in `claims` (and mirrored in `verification_results`)
  where label in {"entail", "contradict", "neutral"}.
测试脚本 
  curl -s http://127.0.0.1:8004/health

  3. 手动测一次接口

  curl -s http://127.0.0.1:8004/check \
    -H 'Content-Type: application/json' \
    -d '{
      "question":"What is first-line treatment for type 2 diabetes?",
      "answer":"Metformin is first-line unless contraindicated.",
      "evidence":"ADA guidelines recommend metformin as initial pharmacologic therapy."
    }'
只用 checker

  python search_r1_preprocess/checker_medrag.py \
    --mode student_checker \
    --model_name /ocean/projects/med230010p/yji3/MedicalRagChecker/runs/checker_sft_balanced_v1 \
    --base_model_path /ocean/projects/med230010p/yji3/models/Meditron3-8B \
    --host 0.0.0.0 \
    --port 8004

用上 checker 和 extractor
CUDA_VISIBLE_DEVICES=1,2
  python search_r1_preprocess/checker_medrag.py \
    --mode student_pipeline \
    --checker_model_path /ocean/projects/med230010p/yji3/MedicalRagChecker/runs/checker_sft_balanced_v1  \
    --checker_base_model_path /ocean/projects/med230010p/yji3/models/Meditron3-8B \
    --extractor_model_path  /ocean/projects/med230010p/yji3/MedicalRagChecker/runs/extractor_sft_meditron3-8b \
    --extractor_base_model_path /ocean/projects/med230010p/yji3/models/Meditron3-8B \
    --max_claims 3 \
    --checker_fast_mode \
    --host 0.0.0.0 \
    --port 8004

mar 6 号代码 代码中只能用 一个 GPU 跑
# 方案B：固定到 GPU 2（和 Search 分开）
CUDA_VISIBLE_DEVICES=2 python search_r1_preprocess/checker_medrag.py \
    --mode student_pipeline \
    --checker_model_path /ocean/projects/med230010p/yji3/MedicalRagChecker/runs/checker_sft_balanced_v1 \
    --checker_base_model_path /ocean/projects/med230010p/yji3/models/Meditron3-8B \
    --extractor_model_path /ocean/projects/med230010p/yji3/MedicalRagChecker/runs/extractor_sft_meditron3-8b \
    --extractor_base_model_path /ocean/projects/med230010p/yji3/models/Meditron3-8B \
    --max_claims 1 \
    --checker_fast_mode \
    --max_concurrent 1 \
    --max_queue 4 \
    --host 0.0.0.0 \
    --port 8004
"""

"""
Checker server for Search + Checker tool training.

Compatible with checker tool contract:
- POST /check with {"answer": "...", "evidence": "...", "question": "..."}
- Returns claim-level labels in `claims` (and mirrored in `verification_results`)
  where label in {"entail", "contradict", "neutral"}.
"""


import argparse
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MedRAGChecker API")

# ── 全局并发控制 ──────────────────────────────────────────────
# GPU 推理串行，semaphore 防止多请求同时占用 GPU 造成 OOM 或竞争
# 这两个值在 main() 里根据 --max_concurrent / --max_queue 初始化
_inference_semaphore: threading.Semaphore = threading.Semaphore(1)
_active_count: int = 0          # 当前排队+处理中的请求数
_active_lock: threading.Lock = threading.Lock()
_MAX_QUEUE_DEPTH: int = 4       # 超过此数立即返回 503，快速失败


class CheckRequest(BaseModel):
    answer: str
    evidence: Optional[str] = ""
    question: Optional[str] = ""


class ClaimResult(BaseModel):
    claim: str
    label: str
    confidence: float


class CheckResponse(BaseModel):
    claims: list[ClaimResult]
    verification_results: list[ClaimResult]
    num_claims: int
    num_supported: int
    num_contradicted: int
    num_neutral: int
    support_rate: float
    contradiction_rate: float


def _normalize_label(text: str) -> str:
    t = text.lower()
    if "contradict" in t:
        return "contradict"
    if "support" in t or "entail" in t:
        return "entail"
    return "neutral"


def _extract_confidence(text: str, default: float = 0.5) -> float:
    m = re.search(r"(?:confidence|score)\s*[:=]?\s*([01](?:\.\d+)?)", text.lower())
    if not m:
        return default
    try:
        v = float(m.group(1))
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def _normalize_student_label(text: str) -> str:
    t = (text or "").lower().strip().replace(".", "").replace("label:", "").strip()
    if any(k in t for k in ["contradicted", "contradict", "refuted"]):
        return "contradict"
    if any(k in t for k in ["entailed", "entail", "supported"]):
        return "entail"
    return "neutral"


class RAGCheckerBasedChecker:
    """LLM-based claim extraction + claim verification."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI()

    def extract_claims(self, answer: str) -> list[str]:
        if not answer.strip():
            return []
        prompt = (
            "Extract atomic factual claims from the text below.\n"
            "Rules:\n"
            "- one claim per line\n"
            "- no numbering\n"
            "- only factual, verifiable claims\n\n"
            f"Text:\n{answer}\n\nClaims:"
        )
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            text = (res.choices[0].message.content or "").strip()
            claims = []
            for line in text.splitlines():
                line = line.strip()
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                if len(line) > 10:
                    claims.append(line)
            return claims[:10]
        except Exception:
            return self._fallback_extract_claims(answer)

    @staticmethod
    def _fallback_extract_claims(answer: str) -> list[str]:
        parts = re.split(r"[.!?。！？]+", answer)
        return [p.strip() for p in parts if len(p.strip()) > 15][:5]

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        if not evidence.strip():
            return {"label": "neutral", "confidence": 0.5}

        prompt = (
            "Given EVIDENCE and CLAIM, classify relation as SUPPORTED, CONTRADICTED, or NEUTRAL.\n"
            "Return exactly:\n"
            "Label: <SUPPORTED|CONTRADICTED|NEUTRAL>\n"
            "Confidence: <0.0-1.0>\n\n"
            f"EVIDENCE:\n{evidence}\n\nCLAIM:\n{claim}\n"
        )
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            text = (res.choices[0].message.content or "").strip()
            return {"label": _normalize_label(text), "confidence": _extract_confidence(text)}
        except Exception:
            return {"label": "neutral", "confidence": 0.5}


class LocalNLIChecker:
    """Local DeBERTa MNLI checker."""

    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # MNLI: 0 contradiction, 1 neutral, 2 entailment
        self.label_map = {0: "contradict", 1: "neutral", 2: "entail"}

    @staticmethod
    def extract_claims(answer: str) -> list[str]:
        if not answer.strip():
            return []
        parts = re.split(r"[.!?。！？]+", answer)
        return [p.strip() for p in parts if len(p.strip()) > 15 and len(p.split()) >= 4][:8]

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        import torch

        if not evidence.strip():
            return {"label": "neutral", "confidence": 0.5}

        inputs = self.tokenizer(
            evidence,
            claim,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = int(torch.argmax(probs, dim=-1).item())
            conf = float(probs[0, pred].item())
        return {"label": self.label_map[pred], "confidence": round(conf, 4)}


class MockChecker:
    @staticmethod
    def extract_claims(answer: str) -> list[str]:
        if not answer.strip():
            return []
        return [answer.strip()[:200]]

    @staticmethod
    def verify_claim(claim: str, evidence: str) -> dict[str, Any]:
        if not evidence.strip():
            return {"label": "neutral", "confidence": 0.5}
        return {"label": "entail", "confidence": 0.6}


class MedRAGCheckerLocal:
    """Adapter for local `medragchecker` package."""

    def __init__(self, model_path: str):
        if not model_path:
            raise ValueError("`--model_name` must be set to your medragchecker model path in medragchecker mode.")
        try:
            from medragchecker import MedRAGChecker
        except Exception as e:
            raise RuntimeError(
                "Failed to import medragchecker. Please install it in this runtime environment."
            ) from e
        self.model = MedRAGChecker(model_path)

    def extract_claims(self, answer: str) -> list[str]:
        return self.model.extract_claims(answer) or []

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        out = self.model.verify_claim(claim, evidence or "")
        # Normalize keys/labels to server contract.
        label = str(out.get("label", "neutral")).lower()
        if label not in ("entail", "contradict", "neutral"):
            if "support" in label:
                label = "entail"
            elif "contra" in label:
                label = "contradict"
            else:
                label = "neutral"
        return {"label": label, "confidence": float(out.get("confidence", 0.0))}


class StudentCheckerLM:
    """
    Adapter to reuse your `eval_student_models.py` style checker model as online checker service.
    Input: (claim, evidence) -> one of {entail, contradict, neutral}.
    """

    def __init__(self, model_path: str, base_model_path: Optional[str] = None, fast_mode: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path_obj = Path(model_path)
        is_lora_adapter = (model_path_obj / "adapter_config.json").exists()

        if is_lora_adapter:
            if base_model_path is None:
                import json

                with open(model_path_obj / "adapter_config.json", "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                base_model_path = adapter_cfg.get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError("LoRA adapter detected, but base model path is missing.")

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
        # StudentCheckerLM 原来没有锁，多线程并发会 GPU 竞争崩溃
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
            f"CLAIM:\n{claim}\n\n"
            f"EVIDENCE:\n{evidence}\n\n"
            "Label:"
        )

    def _score_label(self, prompt: str, label_text: str) -> float:
        """
        Score one candidate label by average token log-probability of label_text
        conditioned on prompt.
        """
        import torch
        import torch.nn.functional as F

        p = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        x = self.tokenizer(prompt + label_text, return_tensors="pt", add_special_tokens=False)

        input_ids = x["input_ids"].to(self.model.device)
        prompt_len = p["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]   # next-token logits
            target = input_ids[:, 1:]            # target token ids
            logp = F.log_softmax(logits, dim=-1)
            tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

        # label tokens start at position prompt_len
        start = max(prompt_len - 1, 0)
        comp_lp = tok_lp[:, start:]
        if comp_lp.numel() == 0:
            return -1e9
        return float(comp_lp.mean().item())

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        import torch

        if not evidence.strip():
            return {"label": "neutral", "confidence": 0.5}

        prompt = self._prompt(claim, evidence)

        candidates = {
            "entail": [" entailed"] if self.fast_mode else [" entailed", " supported", " entail"],
            "contradict": [" contradicted"] if self.fast_mode else [" contradicted", " refuted", " contradict"],
            "neutral": [" neutral"] if self.fast_mode else [" neutral", " unknown", " insufficient"],
        }

        scores: dict[str, float] = {}
        # 加锁保证同一时刻只有一个线程使用 GPU（防止竞争崩溃）
        with self._lock:
            for out_label, variants in candidates.items():
                best = -1e9
                for v in variants:
                    s = self._score_label(prompt, v)
                    if s > best:
                        best = s
                scores[out_label] = best

        pred = max(scores, key=scores.get)

        vals = torch.tensor([scores["entail"], scores["contradict"], scores["neutral"]], dtype=torch.float32)
        probs = torch.softmax(vals, dim=0)
        confidence = float(probs.max().item())

        return {"label": pred, "confidence": round(confidence, 4), "scores": scores}


def _parse_triples_json(text: str) -> list[list[str]]:
    if not text:
        return []
    s = text.replace("```json", "").replace("```", "").strip()
    candidates = [s]
    for m in re.finditer(r"\[", s):
        start = m.start()
        for end in range(len(s) - 1, start, -1):
            if s[end] == "]":
                candidates.append(s[start : end + 1].strip())
                break
    seen = set()
    for cand in sorted(candidates, key=len, reverse=True):
        if cand in seen:
            continue
        seen.add(cand)
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        triples = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                triples.append([str(item[0]), str(item[1]), str(item[2])])
            elif isinstance(item, dict):
                subj = item.get("subject") or item.get("subj") or item.get("s")
                rel = item.get("relation") or item.get("predicate") or item.get("rel") or item.get("p")
                obj = item.get("object") or item.get("obj") or item.get("o")
                if subj is not None and rel is not None and obj is not None:
                    triples.append([str(subj), str(rel), str(obj)])
        if triples:
            return triples
    return []


class StudentExtractorLM:
    """Adapter for student extractor model that outputs factual triples JSON."""

    def __init__(self, model_path: str, base_model_path: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path_obj = Path(model_path)
        is_lora_adapter = (model_path_obj / "adapter_config.json").exists()

        if is_lora_adapter:
            if base_model_path is None:
                with open(model_path_obj / "adapter_config.json", "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                base_model_path = adapter_cfg.get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError("Extractor LoRA adapter detected, but base model path is missing.")
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
        system_part = (
            "You are an information extraction assistant. "
            "Given a medical question and its answer, extract ALL factual triples "
            "as [subject, relation, object]. "
            "Always copy entity names and key phrases EXACTLY from the question or answer; "
            "do NOT paraphrase biomedical terms, abbreviations, or disease names. "
            "Return a pure JSON array of triples, with no explanations, no extra text, "
            "no comments. If there are no clear factual triples, return an empty JSON array []."
        )
        qa_part = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
        return system_part + "\n\n" + qa_part + "\n\nTriples (JSON only, e.g. [[\"subj\", \"rel\", \"obj\"], ...]):\n"

    def extract_triples(self, question: str, answer: str) -> list[list[str]]:
        import torch

        prompt = self._build_prompt(question, answer)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)

        # Guard against rare tokenizer/model vocab mismatch that can crash CUDA kernels.
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            vocab_size = int(self.model.get_input_embeddings().weight.shape[0])
            bad_mask = (input_ids < 0) | (input_ids >= vocab_size)
            if bad_mask.any():
                unk_id = self.tokenizer.unk_token_id
                if unk_id is None:
                    unk_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
                input_ids = input_ids.clone()
                input_ids[bad_mask] = int(unk_id)
                inputs["input_ids"] = input_ids

        try:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self._lock:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        temperature=0.0,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            generated = self.tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            return _parse_triples_json(generated)
        except RuntimeError as e:
            # Keep service alive on CUDA assert; caller will fallback to sentence splitting.
            print(f"[StudentExtractorLM] extract_triples failed: {e}")
            return []


class StudentExtractorCheckerPipeline:
    """Use student extractor -> claims, then student checker for verification."""

    def __init__(
        self,
        checker_model_path: str,
        extractor_model_path: str,
        checker_base_model_path: Optional[str] = None,
        extractor_base_model_path: Optional[str] = None,
        max_claims: int = 3,
        checker_fast_mode: bool = True,
    ):
        self.extractor = StudentExtractorLM(extractor_model_path, base_model_path=extractor_base_model_path)
        self.checker = StudentCheckerLM(
            checker_model_path,
            base_model_path=checker_base_model_path,
            fast_mode=checker_fast_mode,
        )
        self._last_question = ""
        self.max_claims = max(1, int(max_claims))

    def set_question(self, question: str) -> None:
        self._last_question = question or ""

    def extract_claims(self, answer: str) -> list[str]:
        triples = self.extractor.extract_triples(self._last_question, answer)
        claims = []
        for t in triples:
            if len(t) != 3:
                continue
            s, r, o = t
            claim = f"{s} {r} {o}".strip()
            if len(claim) > 8:
                claims.append(claim)
        if claims:
            return claims[: self.max_claims]
        # fallback
        parts = re.split(r"[.!?。！？]+", answer)
        return [p.strip() for p in parts if len(p.strip()) > 15][: self.max_claims]

    def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
        return self.checker.verify_claim(claim, evidence)


checker: Optional[Any] = None
import time as _time


@app.post("/check", response_model=CheckResponse)
def check_answer(request: CheckRequest):
    """
    GPU 推理并发保护：
      1. 先检查队列深度，超过 _MAX_QUEUE_DEPTH 立即 503（快速失败，不堆积）
      2. 用 semaphore 串行化 GPU 访问（防止多线程竞争 GPU OOM 或崩溃）
      3. 打印 wait/gpu 耗时，便于调参
    """
    global _active_count
    from fastapi import HTTPException

    assert checker is not None, "Checker not initialized"

    # ① 快速失败：队列已满时拒绝，不让请求堆积
    with _active_lock:
        if _active_count >= _MAX_QUEUE_DEPTH:
            raise HTTPException(
                status_code=503,
                detail=f"Checker overloaded (queue={_active_count}/{_MAX_QUEUE_DEPTH}). Try later.",
            )
        _active_count += 1

    t_arrive = _time.time()
    try:
        # ② 等待 GPU 槽位，最多等 30s，超时则 503
        acquired = _inference_semaphore.acquire(timeout=30)
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="Checker semaphore wait timeout (30s). Server too busy.",
            )
        t_gpu_start = _time.time()
        try:
            if hasattr(checker, "set_question"):
                try:
                    checker.set_question(request.question or "")
                except Exception:
                    pass

            try:
                claims = checker.extract_claims(request.answer)
            except Exception as e:
                print(f"[checker] extract_claims failed: {e}")
                claims = []

            results: list[ClaimResult] = []
            for claim in claims:
                try:
                    v = checker.verify_claim(claim, request.evidence or "")
                except Exception as e:
                    print(f"[checker] verify_claim failed: {e}")
                    v = {"label": "neutral", "confidence": 0.0}
                results.append(
                    ClaimResult(
                        claim=claim,
                        label=str(v.get("label", "neutral")),
                        confidence=float(v.get("confidence", 0.0)),
                    )
                )

            t_done = _time.time()
            print(
                f"[checker] queue_wait={t_gpu_start - t_arrive:.1f}s "
                f"gpu_time={t_done - t_gpu_start:.1f}s "
                f"claims={len(results)} active={_active_count}"
            )
        finally:
            _inference_semaphore.release()

    finally:
        with _active_lock:
            _active_count -= 1

    num_claims = len(results)
    num_supported = sum(1 for r in results if r.label == "entail")
    num_contradicted = sum(1 for r in results if r.label == "contradict")
    num_neutral = sum(1 for r in results if r.label == "neutral")
    support_rate = round(num_supported / num_claims, 4) if num_claims else 0.0
    contradiction_rate = round(num_contradicted / num_claims, 4) if num_claims else 0.0

    return CheckResponse(
        claims=results,
        verification_results=results,
        num_claims=num_claims,
        num_supported=num_supported,
        num_contradicted=num_contradicted,
        num_neutral=num_neutral,
        support_rate=support_rate,
        contradiction_rate=contradiction_rate,
    )


@app.get("/health")
def health():
    return {"status": "ok", "checker_type": type(checker).__name__ if checker is not None else "uninitialized"}


def main():
    global checker, _inference_semaphore, _MAX_QUEUE_DEPTH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["openai", "local_nli", "medragchecker", "student_checker", "student_pipeline", "mock"],
        default="local_nli",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-large-mnli",
        help=(
            "For local_nli: HF model name; "
            "for openai: OpenAI model name; "
            "for medragchecker/student_checker: local model or adapter path"
        ),
    )
    parser.add_argument("--base_model_path", type=str, default=None, help="Required when --mode student_checker uses LoRA adapter.")
    parser.add_argument("--checker_model_path", type=str, default=None, help="Student checker model/adapter path for student_pipeline.")
    parser.add_argument("--checker_base_model_path", type=str, default=None, help="Base model path if checker is LoRA.")
    parser.add_argument("--extractor_model_path", type=str, default=None, help="Student extractor model/adapter path for student_pipeline.")
    parser.add_argument("--extractor_base_model_path", type=str, default=None, help="Base model path if extractor is LoRA.")
    parser.add_argument("--max_claims", type=int, default=1, help="Maximum claims checked per request. Default=1 (faster, less GPU pressure).")
    parser.add_argument("--checker_fast_mode", action="store_true", default=True, help="Use 1 verbalizer per label (faster). Default=True.")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    # ── 新增并发控制参数 ──────────────────────────────────────
    parser.add_argument(
        "--max_concurrent", type=int, default=1,
        help=(
            "Max simultaneous GPU inferences. "
            "For LLM-based checker with device_map=auto (model parallelism), use 1. "
            "Only increase if you have truly independent model replicas per GPU."
        ),
    )
    parser.add_argument(
        "--max_queue", type=int, default=20,
        help="Max requests allowed to queue (including the one being processed). "
             "Requests beyond this are immediately rejected with 503. "
             "Set high enough to absorb bursts: num_AgentLoopWorkers * max_check_calls_per_step.",
    )
    args = parser.parse_args()

    # 初始化全局并发控制
    _inference_semaphore = threading.Semaphore(args.max_concurrent)
    _MAX_QUEUE_DEPTH = args.max_queue

    if args.mode == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --mode openai")
        checker = RAGCheckerBasedChecker(model_name=args.model_name)
    elif args.mode == "local_nli":
        checker = LocalNLIChecker(nli_model_name=args.model_name)
    elif args.mode == "medragchecker":
        checker = MedRAGCheckerLocal(model_path=args.model_name)
    elif args.mode == "student_checker":
        checker = StudentCheckerLM(
            model_path=args.model_name,
            base_model_path=args.base_model_path,
            fast_mode=args.checker_fast_mode,
        )
    elif args.mode == "student_pipeline":
        if not args.checker_model_path or not args.extractor_model_path:
            raise ValueError("--mode student_pipeline requires --checker_model_path and --extractor_model_path")
        checker = StudentExtractorCheckerPipeline(
            checker_model_path=args.checker_model_path,
            extractor_model_path=args.extractor_model_path,
            checker_base_model_path=args.checker_base_model_path,
            extractor_base_model_path=args.extractor_base_model_path,
            max_claims=args.max_claims,
            checker_fast_mode=args.checker_fast_mode,
        )
    else:
        checker = MockChecker()

    print(
        f"Starting checker server on http://{args.host}:{args.port} "
        f"(mode={args.mode}, max_concurrent={args.max_concurrent}, "
        f"max_queue={args.max_queue}, max_claims={args.max_claims})"
    )
    # 不用 uvicorn limit_concurrency，完全依赖应用层 semaphore + _MAX_QUEUE_DEPTH 控制
    # 两层限制叠加会导致请求在 HTTP 层就被拒绝，semaphore 根本没机会运行
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
# import argparse
# import json
# import os
# import re
# import threading
# from pathlib import Path
# from typing import Any, Optional

# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI(title="MedRAGChecker API")


# class CheckRequest(BaseModel):
#     answer: str
#     evidence: Optional[str] = ""
#     question: Optional[str] = ""


# class ClaimResult(BaseModel):
#     claim: str
#     label: str
#     confidence: float


# class CheckResponse(BaseModel):
#     claims: list[ClaimResult]
#     verification_results: list[ClaimResult]
#     num_claims: int
#     num_supported: int
#     num_contradicted: int
#     num_neutral: int
#     support_rate: float
#     contradiction_rate: float


# def _normalize_label(text: str) -> str:
#     t = text.lower()
#     if "contradict" in t:
#         return "contradict"
#     if "support" in t or "entail" in t:
#         return "entail"
#     return "neutral"


# def _extract_confidence(text: str, default: float = 0.5) -> float:
#     m = re.search(r"(?:confidence|score)\s*[:=]?\s*([01](?:\.\d+)?)", text.lower())
#     if not m:
#         return default
#     try:
#         v = float(m.group(1))
#         return max(0.0, min(1.0, v))
#     except Exception:
#         return default


# def _normalize_student_label(text: str) -> str:
#     t = (text or "").lower().strip().replace(".", "").replace("label:", "").strip()
#     if any(k in t for k in ["contradicted", "contradict", "refuted"]):
#         return "contradict"
#     if any(k in t for k in ["entailed", "entail", "supported"]):
#         return "entail"
#     return "neutral"


# class RAGCheckerBasedChecker:
#     """LLM-based claim extraction + claim verification."""

#     def __init__(self, model_name: str = "gpt-4o-mini"):
#         from openai import OpenAI

#         self.model_name = model_name
#         self.client = OpenAI()

#     def extract_claims(self, answer: str) -> list[str]:
#         if not answer.strip():
#             return []
#         prompt = (
#             "Extract atomic factual claims from the text below.\n"
#             "Rules:\n"
#             "- one claim per line\n"
#             "- no numbering\n"
#             "- only factual, verifiable claims\n\n"
#             f"Text:\n{answer}\n\nClaims:"
#         )
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0,
#                 max_tokens=1000,
#             )
#             text = (res.choices[0].message.content or "").strip()
#             claims = []
#             for line in text.splitlines():
#                 line = line.strip()
#                 line = re.sub(r"^\d+[\.\)]\s*", "", line)
#                 if len(line) > 10:
#                     claims.append(line)
#             return claims[:10]
#         except Exception:
#             return self._fallback_extract_claims(answer)

#     @staticmethod
#     def _fallback_extract_claims(answer: str) -> list[str]:
#         parts = re.split(r"[.!?。！？]+", answer)
#         return [p.strip() for p in parts if len(p.strip()) > 15][:5]

#     def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
#         if not evidence.strip():
#             return {"label": "neutral", "confidence": 0.5}

#         prompt = (
#             "Given EVIDENCE and CLAIM, classify relation as SUPPORTED, CONTRADICTED, or NEUTRAL.\n"
#             "Return exactly:\n"
#             "Label: <SUPPORTED|CONTRADICTED|NEUTRAL>\n"
#             "Confidence: <0.0-1.0>\n\n"
#             f"EVIDENCE:\n{evidence}\n\nCLAIM:\n{claim}\n"
#         )
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0,
#                 max_tokens=100,
#             )
#             text = (res.choices[0].message.content or "").strip()
#             return {"label": _normalize_label(text), "confidence": _extract_confidence(text)}
#         except Exception:
#             return {"label": "neutral", "confidence": 0.5}


# class LocalNLIChecker:
#     """Local DeBERTa MNLI checker."""

#     def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli"):
#         import torch
#         from transformers import AutoModelForSequenceClassification, AutoTokenizer

#         self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
#         self.model.eval()
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(self.device)
#         # MNLI: 0 contradiction, 1 neutral, 2 entailment
#         self.label_map = {0: "contradict", 1: "neutral", 2: "entail"}

#     @staticmethod
#     def extract_claims(answer: str) -> list[str]:
#         if not answer.strip():
#             return []
#         parts = re.split(r"[.!?。！？]+", answer)
#         return [p.strip() for p in parts if len(p.strip()) > 15 and len(p.split()) >= 4][:8]

#     def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
#         import torch

#         if not evidence.strip():
#             return {"label": "neutral", "confidence": 0.5}

#         inputs = self.tokenizer(
#             evidence,
#             claim,
#             truncation=True,
#             max_length=512,
#             return_tensors="pt",
#         )
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             probs = torch.softmax(outputs.logits, dim=-1)
#             pred = int(torch.argmax(probs, dim=-1).item())
#             conf = float(probs[0, pred].item())
#         return {"label": self.label_map[pred], "confidence": round(conf, 4)}


# class MockChecker:
#     @staticmethod
#     def extract_claims(answer: str) -> list[str]:
#         if not answer.strip():
#             return []
#         return [answer.strip()[:200]]

#     @staticmethod
#     def verify_claim(claim: str, evidence: str) -> dict[str, Any]:
#         if not evidence.strip():
#             return {"label": "neutral", "confidence": 0.5}
#         return {"label": "entail", "confidence": 0.6}


# class MedRAGCheckerLocal:
#     """Adapter for local `medragchecker` package."""

#     def __init__(self, model_path: str):
#         if not model_path:
#             raise ValueError("`--model_name` must be set to your medragchecker model path in medragchecker mode.")
#         try:
#             from medragchecker import MedRAGChecker
#         except Exception as e:
#             raise RuntimeError(
#                 "Failed to import medragchecker. Please install it in this runtime environment."
#             ) from e
#         self.model = MedRAGChecker(model_path)

#     def extract_claims(self, answer: str) -> list[str]:
#         return self.model.extract_claims(answer) or []

#     def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
#         out = self.model.verify_claim(claim, evidence or "")
#         # Normalize keys/labels to server contract.
#         label = str(out.get("label", "neutral")).lower()
#         if label not in ("entail", "contradict", "neutral"):
#             if "support" in label:
#                 label = "entail"
#             elif "contra" in label:
#                 label = "contradict"
#             else:
#                 label = "neutral"
#         return {"label": label, "confidence": float(out.get("confidence", 0.0))}


# class StudentCheckerLM:
#     """
#     Adapter to reuse your `eval_student_models.py` style checker model as online checker service.
#     Input: (claim, evidence) -> one of {entail, contradict, neutral}.
#     """

#     def __init__(self, model_path: str, base_model_path: Optional[str] = None, fast_mode: bool = True):
#         import torch
#         from transformers import AutoModelForCausalLM, AutoTokenizer

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         model_path_obj = Path(model_path)
#         is_lora_adapter = (model_path_obj / "adapter_config.json").exists()

#         if is_lora_adapter:
#             if base_model_path is None:
#                 import json

#                 with open(model_path_obj / "adapter_config.json", "r", encoding="utf-8") as f:
#                     adapter_cfg = json.load(f)
#                 base_model_path = adapter_cfg.get("base_model_name_or_path")
#             if not base_model_path:
#                 raise ValueError("LoRA adapter detected, but base model path is missing.")

#             self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
#             model = AutoModelForCausalLM.from_pretrained(
#                 base_model_path,
#                 torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
#                 device_map="auto",
#             )
#             from peft import PeftModel

#             self.model = PeftModel.from_pretrained(model, model_path)
#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
#                 device_map="auto",
#             )

#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model.eval()
#         self.fast_mode = fast_mode

#     @staticmethod
#     def extract_claims(answer: str) -> list[str]:
#         if not answer.strip():
#             return []
#         parts = re.split(r"[.!?。！？]+", answer)
#         return [p.strip() for p in parts if len(p.strip()) > 15][:8]

#     @staticmethod
#     def _prompt(claim: str, evidence: str) -> str:
#         return (
#             "Decide whether the EVIDENCE entails, contradicts, or is neutral to the CLAIM.\n"
#             "Respond with one of: entailed | contradicted | neutral\n\n"
#             f"CLAIM:\n{claim}\n\n"
#             f"EVIDENCE:\n{evidence}\n\n"
#             "Label:"
#         )

#     def _score_label(self, prompt: str, label_text: str) -> float:
#         """
#         Score one candidate label by average token log-probability of label_text
#         conditioned on prompt.
#         """
#         import torch
#         import torch.nn.functional as F

#         p = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
#         x = self.tokenizer(prompt + label_text, return_tensors="pt", add_special_tokens=False)

#         input_ids = x["input_ids"].to(self.model.device)
#         prompt_len = p["input_ids"].shape[1]

#         with torch.no_grad():
#             outputs = self.model(input_ids=input_ids)
#             logits = outputs.logits[:, :-1, :]   # next-token logits
#             target = input_ids[:, 1:]            # target token ids
#             logp = F.log_softmax(logits, dim=-1)
#             tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

#         # label tokens start at position prompt_len
#         start = max(prompt_len - 1, 0)
#         comp_lp = tok_lp[:, start:]
#         if comp_lp.numel() == 0:
#             return -1e9
#         return float(comp_lp.mean().item())

#     def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
#         import torch

#         if not evidence.strip():
#             return {"label": "neutral", "confidence": 0.5}

#         prompt = self._prompt(claim, evidence)

#         candidates = {
#             "entail": [" entailed"] if self.fast_mode else [" entailed", " supported", " entail"],
#             "contradict": [" contradicted"] if self.fast_mode else [" contradicted", " refuted", " contradict"],
#             "neutral": [" neutral"] if self.fast_mode else [" neutral", " unknown", " insufficient"],
#         }

#         scores: dict[str, float] = {}
#         for out_label, variants in candidates.items():
#             best = -1e9
#             for v in variants:
#                 s = self._score_label(prompt, v)
#                 if s > best:
#                     best = s
#             scores[out_label] = best

#         pred = max(scores, key=scores.get)

#         vals = torch.tensor([scores["entail"], scores["contradict"], scores["neutral"]], dtype=torch.float32)
#         probs = torch.softmax(vals, dim=0)
#         confidence = float(probs.max().item())

#         return {"label": pred, "confidence": round(confidence, 4), "scores": scores}


# def _parse_triples_json(text: str) -> list[list[str]]:
#     if not text:
#         return []
#     s = text.replace("```json", "").replace("```", "").strip()
#     candidates = [s]
#     for m in re.finditer(r"\[", s):
#         start = m.start()
#         for end in range(len(s) - 1, start, -1):
#             if s[end] == "]":
#                 candidates.append(s[start : end + 1].strip())
#                 break
#     seen = set()
#     for cand in sorted(candidates, key=len, reverse=True):
#         if cand in seen:
#             continue
#         seen.add(cand)
#         try:
#             data = json.loads(cand)
#         except Exception:
#             continue
#         if not isinstance(data, list):
#             continue
#         triples = []
#         for item in data:
#             if isinstance(item, (list, tuple)) and len(item) == 3:
#                 triples.append([str(item[0]), str(item[1]), str(item[2])])
#             elif isinstance(item, dict):
#                 subj = item.get("subject") or item.get("subj") or item.get("s")
#                 rel = item.get("relation") or item.get("predicate") or item.get("rel") or item.get("p")
#                 obj = item.get("object") or item.get("obj") or item.get("o")
#                 if subj is not None and rel is not None and obj is not None:
#                     triples.append([str(subj), str(rel), str(obj)])
#         if triples:
#             return triples
#     return []


# class StudentExtractorLM:
#     """Adapter for student extractor model that outputs factual triples JSON."""

#     def __init__(self, model_path: str, base_model_path: Optional[str] = None):
#         import torch
#         from transformers import AutoModelForCausalLM, AutoTokenizer

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         model_path_obj = Path(model_path)
#         is_lora_adapter = (model_path_obj / "adapter_config.json").exists()

#         if is_lora_adapter:
#             if base_model_path is None:
#                 with open(model_path_obj / "adapter_config.json", "r", encoding="utf-8") as f:
#                     adapter_cfg = json.load(f)
#                 base_model_path = adapter_cfg.get("base_model_name_or_path")
#             if not base_model_path:
#                 raise ValueError("Extractor LoRA adapter detected, but base model path is missing.")
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
#             model = AutoModelForCausalLM.from_pretrained(
#                 base_model_path,
#                 torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
#                 device_map="auto",
#             )
#             from peft import PeftModel

#             self.model = PeftModel.from_pretrained(model, model_path)
#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
#                 device_map="auto",
#             )

#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model.eval()
#         self._lock = threading.Lock()

#     @staticmethod
#     def _build_prompt(question: str, answer: str) -> str:
#         system_part = (
#             "You are an information extraction assistant. "
#             "Given a medical question and its answer, extract ALL factual triples "
#             "as [subject, relation, object]. "
#             "Always copy entity names and key phrases EXACTLY from the question or answer; "
#             "do NOT paraphrase biomedical terms, abbreviations, or disease names. "
#             "Return a pure JSON array of triples, with no explanations, no extra text, "
#             "no comments. If there are no clear factual triples, return an empty JSON array []."
#         )
#         qa_part = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
#         return system_part + "\n\n" + qa_part + "\n\nTriples (JSON only, e.g. [[\"subj\", \"rel\", \"obj\"], ...]):\n"

#     def extract_triples(self, question: str, answer: str) -> list[list[str]]:
#         import torch

#         prompt = self._build_prompt(question, answer)
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)

#         # Guard against rare tokenizer/model vocab mismatch that can crash CUDA kernels.
#         input_ids = inputs.get("input_ids")
#         if input_ids is not None:
#             vocab_size = int(self.model.get_input_embeddings().weight.shape[0])
#             bad_mask = (input_ids < 0) | (input_ids >= vocab_size)
#             if bad_mask.any():
#                 unk_id = self.tokenizer.unk_token_id
#                 if unk_id is None:
#                     unk_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
#                 input_ids = input_ids.clone()
#                 input_ids[bad_mask] = int(unk_id)
#                 inputs["input_ids"] = input_ids

#         try:
#             inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
#             with self._lock:
#                 with torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=256,
#                         do_sample=False,
#                         temperature=0.0,
#                         eos_token_id=self.tokenizer.eos_token_id,
#                         pad_token_id=self.tokenizer.eos_token_id,
#                     )
#             generated = self.tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
#             return _parse_triples_json(generated)
#         except RuntimeError as e:
#             # Keep service alive on CUDA assert; caller will fallback to sentence splitting.
#             print(f"[StudentExtractorLM] extract_triples failed: {e}")
#             return []


# class StudentExtractorCheckerPipeline:
#     """Use student extractor -> claims, then student checker for verification."""

#     def __init__(
#         self,
#         checker_model_path: str,
#         extractor_model_path: str,
#         checker_base_model_path: Optional[str] = None,
#         extractor_base_model_path: Optional[str] = None,
#         max_claims: int = 3,
#         checker_fast_mode: bool = True,
#     ):
#         self.extractor = StudentExtractorLM(extractor_model_path, base_model_path=extractor_base_model_path)
#         self.checker = StudentCheckerLM(
#             checker_model_path,
#             base_model_path=checker_base_model_path,
#             fast_mode=checker_fast_mode,
#         )
#         self._last_question = ""
#         self.max_claims = max(1, int(max_claims))

#     def set_question(self, question: str) -> None:
#         self._last_question = question or ""

#     def extract_claims(self, answer: str) -> list[str]:
#         triples = self.extractor.extract_triples(self._last_question, answer)
#         claims = []
#         for t in triples:
#             if len(t) != 3:
#                 continue
#             s, r, o = t
#             claim = f"{s} {r} {o}".strip()
#             if len(claim) > 8:
#                 claims.append(claim)
#         if claims:
#             return claims[: self.max_claims]
#         # fallback
#         parts = re.split(r"[.!?。！？]+", answer)
#         return [p.strip() for p in parts if len(p.strip()) > 15][: self.max_claims]

#     def verify_claim(self, claim: str, evidence: str) -> dict[str, Any]:
#         return self.checker.verify_claim(claim, evidence)


# checker: Optional[Any] = None


# @app.post("/check", response_model=CheckResponse)
# def check_answer(request: CheckRequest):
#     assert checker is not None, "Checker not initialized"
#     if hasattr(checker, "set_question"):
#         try:
#             checker.set_question(request.question or "")
#         except Exception:
#             pass

#     try:
#         claims = checker.extract_claims(request.answer)
#     except Exception as e:
#         print(f"[checker] extract_claims failed: {e}")
#         claims = []
#     results: list[ClaimResult] = []
#     for claim in claims:
#         try:
#             v = checker.verify_claim(claim, request.evidence or "")
#         except Exception as e:
#             print(f"[checker] verify_claim failed: {e}")
#             v = {"label": "neutral", "confidence": 0.0}
#         results.append(
#             ClaimResult(
#                 claim=claim,
#                 label=str(v.get("label", "neutral")),
#                 confidence=float(v.get("confidence", 0.0)),
#             )
#         )

#     num_claims = len(results)
#     num_supported = sum(1 for r in results if r.label == "entail")
#     num_contradicted = sum(1 for r in results if r.label == "contradict")
#     num_neutral = sum(1 for r in results if r.label == "neutral")
#     support_rate = round(num_supported / num_claims, 4) if num_claims else 0.0
#     contradiction_rate = round(num_contradicted / num_claims, 4) if num_claims else 0.0

#     return CheckResponse(
#         claims=results,
#         verification_results=results,
#         num_claims=num_claims,
#         num_supported=num_supported,
#         num_contradicted=num_contradicted,
#         num_neutral=num_neutral,
#         support_rate=support_rate,
#         contradiction_rate=contradiction_rate,
#     )


# @app.get("/health")
# def health():
#     return {"status": "ok", "checker_type": type(checker).__name__ if checker is not None else "uninitialized"}


# def main():
#     global checker

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--mode",
#         choices=["openai", "local_nli", "medragchecker", "student_checker", "student_pipeline", "mock"],
#         default="local_nli",
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="microsoft/deberta-large-mnli",
#         help=(
#             "For local_nli: HF model name; "
#             "for openai: OpenAI model name; "
#             "for medragchecker/student_checker: local model or adapter path"
#         ),
#     )
#     parser.add_argument("--base_model_path", type=str, default=None, help="Required when --mode student_checker uses LoRA adapter.")
#     parser.add_argument("--checker_model_path", type=str, default=None, help="Student checker model/adpater path for student_pipeline.")
#     parser.add_argument("--checker_base_model_path", type=str, default=None, help="Base model path if checker is LoRA.")
#     parser.add_argument("--extractor_model_path", type=str, default=None, help="Student extractor model/adapter path for student_pipeline.")
#     parser.add_argument("--extractor_base_model_path", type=str, default=None, help="Base model path if extractor is LoRA.")
#     parser.add_argument("--max_claims", type=int, default=3, help="Maximum claims checked per request in student_pipeline.")
#     parser.add_argument("--checker_fast_mode", action="store_true", help="Use 1 verbalizer per label for faster checker inference.")
#     parser.add_argument("--port", type=int, default=8004)
#     parser.add_argument("--host", type=str, default="0.0.0.0")
#     args = parser.parse_args()

#     if args.mode == "openai":
#         if not os.getenv("OPENAI_API_KEY"):
#             raise RuntimeError("OPENAI_API_KEY is required for --mode openai")
#         checker = RAGCheckerBasedChecker(model_name=args.model_name)
#     elif args.mode == "local_nli":
#         checker = LocalNLIChecker(nli_model_name=args.model_name)
#     elif args.mode == "medragchecker":
#         checker = MedRAGCheckerLocal(model_path=args.model_name)
#     elif args.mode == "student_checker":
#         checker = StudentCheckerLM(
#             model_path=args.model_name,
#             base_model_path=args.base_model_path,
#             fast_mode=args.checker_fast_mode,
#         )
#     elif args.mode == "student_pipeline":
#         if not args.checker_model_path or not args.extractor_model_path:
#             raise ValueError("--mode student_pipeline requires --checker_model_path and --extractor_model_path")
#         checker = StudentExtractorCheckerPipeline(
#             checker_model_path=args.checker_model_path,
#             extractor_model_path=args.extractor_model_path,
#             checker_base_model_path=args.checker_base_model_path,
#             extractor_base_model_path=args.extractor_base_model_path,
#             max_claims=args.max_claims,
#             checker_fast_mode=args.checker_fast_mode,
#         )
#     else:
#         checker = MockChecker()

#     print(f"Starting checker server on http://{args.host}:{args.port} (mode={args.mode})")
#     uvicorn.run(app, host=args.host, port=args.port)


# if __name__ == "__main__":
#     main()

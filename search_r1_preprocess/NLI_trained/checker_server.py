#!/usr/bin/env python3
"""
cd /ocean/projects/med230010p/yji3/BrowseCamp/verl/search_r1_preprocess

# 确认文件都在
ls NLI_trained/checker_server.py NLI_trained/nli_classifier.py pubmedbert-mednli/config.json

# 再起


CUDA_VISIBLE_DEVICES=1  python search_r1_preprocess/NLI_trained/checker_server.py \
    --model_path     search_r1_preprocess/pubmedbert-mednli \
    --max_claims     2 \
    --max_concurrent 2 \
    --max_queue      40 \
    --port           8010 \
    > search_r1_preprocess/NLI_trained/checker_server.log 2>&1 &


# 记下 pid
echo "server PID = $!"

# 等模型加载(PubMedBERT 冷启动 ~10-20s)
sleep 15

# 确认进程还在
ps -p $! && echo "server is alive"

# 确认端口起来
curl -s http://127.0.0.1:8010/health
# 期望: {"status":"ok","checker_type":"PubMedBERTMedNLIChecker"}


Standalone PubMedBERT-MedNLI checker server.

Drop this file at: search_r1_preprocess/NLI_trained/checker_server.py

Same /check contract as checker_medrag_gpt4omini.py so the RL trainer does
not need to change.  Run this INSTEAD of that server when you want to use
the fine-tuned PubMedBERT-MedNLI classifier as the reward verifier.

Usage:

  # from .../search_r1_preprocess/
  CUDA_VISIBLE_DEVICES=1 python NLI_trained/checker_server.py \
      --model_path   NLI_trained/pubmedbert-mednli \
      --max_claims   2 \
      --max_concurrent 2 \
      --max_queue    40 \
      --port         8004

  # health check
  curl -s http://127.0.0.1:8004/health

  # manual test
  curl -s http://127.0.0.1:8004/check \
    -H 'Content-Type: application/json' \
    -d '{"question":"How does aspirin work?",
         "answer":"Aspirin irreversibly inhibits COX-1.",
         "evidence":"Aspirin acetylates serine 530 of COX-1 and permanently blocks it."}'
"""

from __future__ import annotations

import argparse
import threading
import time as _time
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local class.  Because this file lives in NLI_trained/ and nli_classifier.py
# lives next to it, either layout works depending on how you launch:
#   * `python NLI_trained/checker_server.py`     -> CWD on sys.path, relative import needs help
#   * `python -m NLI_trained.checker_server`     -> package import
# We handle both.
try:
    from NLI_trained.nli_classifier import PubMedBERTMedNLIChecker
except ImportError:  # fallback: same-dir import
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nli_classifier import PubMedBERTMedNLIChecker


app = FastAPI(title="PubMedBERT-MedNLI checker API")

# Concurrency guards (mirrors checker_medrag_gpt4omini.py behaviour)
_inference_semaphore: threading.Semaphore = threading.Semaphore(1)
_active_count: int = 0
_active_lock: threading.Lock = threading.Lock()
_MAX_QUEUE_DEPTH: int = 40


# -----------------------------------------------------------------------------
# Request / response models — IDENTICAL to checker_medrag_gpt4omini.py so the
# trainer client does not need to change.
# -----------------------------------------------------------------------------

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


checker: Optional[PubMedBERTMedNLIChecker] = None
_max_claims: int = 2


@app.post("/check", response_model=CheckResponse)
def check_answer(request: CheckRequest):
    global _active_count
    assert checker is not None, "Checker not initialised"

    # queue-depth fast-fail
    with _active_lock:
        if _active_count >= _MAX_QUEUE_DEPTH:
            raise HTTPException(
                503,
                f"Checker overloaded (queue={_active_count}/{_MAX_QUEUE_DEPTH}).",
            )
        _active_count += 1

    t_arrive = _time.time()
    try:
        got = _inference_semaphore.acquire(timeout=30)
        if not got:
            raise HTTPException(503, "Checker semaphore timeout (30s).")
        t_gpu = _time.time()
        try:
            checker.set_question(request.question or "")
            claims = checker.extract_claims(request.answer)[:_max_claims]

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
                f"[checker] wait={t_gpu - t_arrive:.2f}s "
                f"gpu={t_done - t_gpu:.2f}s "
                f"claims={len(results)} active={_active_count} "
                f"labels={[r.label for r in results]}"
            )
        finally:
            _inference_semaphore.release()
    finally:
        with _active_lock:
            _active_count -= 1

    n = len(results)
    nsup = sum(1 for r in results if r.label == "entail")
    ncon = sum(1 for r in results if r.label == "contradict")
    nneu = sum(1 for r in results if r.label == "neutral")
    return CheckResponse(
        claims=results,
        verification_results=results,
        num_claims=n,
        num_supported=nsup,
        num_contradicted=ncon,
        num_neutral=nneu,
        support_rate=round(nsup / n, 4) if n else 0.0,
        contradiction_rate=round(ncon / n, 4) if n else 0.0,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checker_type": type(checker).__name__ if checker else "uninitialized",
    }


def main():
    global checker, _inference_semaphore, _MAX_QUEUE_DEPTH, _max_claims

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the fine-tuned PubMedBERT-MedNLI checkpoint "
                        "(the ./pubmedbert-mednli/ dir produced by "
                        "finetune_pubmedbert_mednli.py)")
    p.add_argument("--max_length",     type=int, default=256)
    p.add_argument("--max_claims",     type=int, default=2)
    p.add_argument("--max_concurrent", type=int, default=2)
    p.add_argument("--max_queue",      type=int, default=40)
    p.add_argument("--host",           type=str, default="0.0.0.0")
    p.add_argument("--port",           type=int, default=8004)
    # Optional: override MedNLI label order if you ever fine-tune with a
    # different mapping (format: "entail,neutral,contradict" -> ids 0,1,2)
    p.add_argument("--label_order", type=str, default="entail,neutral,contradict",
                   help="Comma-separated labels in index order 0,1,2")
    args = p.parse_args()

    _inference_semaphore = threading.Semaphore(args.max_concurrent)
    _MAX_QUEUE_DEPTH = args.max_queue
    _max_claims = max(1, args.max_claims)

    labels = [s.strip() for s in args.label_order.split(",")]
    assert set(labels) == {"entail", "neutral", "contradict"}, (
        f"label_order must be a permutation of entail,neutral,contradict; got {labels}"
    )
    label_map = {i: lbl for i, lbl in enumerate(labels)}

    checker = PubMedBERTMedNLIChecker(
        model_path=args.model_path,
        max_length=args.max_length,
        label_map=label_map,
    )

    print(
        f"Starting PubMedBERT-MedNLI checker http://{args.host}:{args.port} "
        f"model_path={args.model_path} "
        f"max_concurrent={args.max_concurrent} max_queue={args.max_queue} "
        f"max_claims={_max_claims} label_map={label_map}"
    )
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=5)


if __name__ == "__main__":
    main()

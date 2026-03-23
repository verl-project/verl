"""
LLM-based faithfulness judge.

Uses a small LLM to evaluate whether the generated answer is faithful to the
retrieved documents. This provides a more nuanced evaluation than NLI alone,
as it can assess complex medical reasoning chains.

Designed for two deployment modes:
1. Local model (e.g., Qwen2.5-1.5B-Instruct) — runs on CPU/GPU alongside training
2. Remote API (e.g., vLLM/SGLang server) — separate inference server

For RL training, Mode 2 (remote) is recommended to avoid GPU memory conflicts.
"""

import json
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ============================================================
# Prompt Templates
# ============================================================

FAITHFULNESS_JUDGE_PROMPT = """You are a medical fact-checker. Your job is to evaluate whether an answer is faithfully supported by the provided reference documents.

## Reference Documents
{documents}

## Question
{question}

## Answer to Evaluate
{answer}

## Evaluation Criteria
1. Is the answer directly supported by information in the reference documents?
2. Does the answer contain any claims NOT found in the documents?
3. Does the answer contradict any information in the documents?

## Instructions
Rate the faithfulness on a scale of 0 to 1:
- 1.0: Fully supported — every claim in the answer is found in the documents
- 0.7: Mostly supported — main claims are supported, minor details may not be
- 0.5: Partially supported — some claims supported, some not
- 0.3: Weakly supported — only tangentially related to documents
- 0.0: Not supported or contradicts the documents

Respond in this exact JSON format:
{{"score": <float>, "reason": "<brief explanation>"}}
"""

ENTITY_EXTRACTION_PROMPT = """Extract all medical entities from the following text. Include:
- Drug names (generic and brand)
- Disease/condition names
- Symptoms
- Medical procedures
- Anatomical terms
- Lab tests / biomarkers

Text: {text}

Respond with a JSON list of entities:
{{"entities": ["entity1", "entity2", ...]}}
"""


# ============================================================
# Remote LLM Judge (via API - recommended for RL training)
# ============================================================

def judge_faithfulness_remote(
    documents: str,
    question: str,
    answer: str,
    api_url: str = "http://127.0.0.1:8001/v1/chat/completions",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    temperature: float = 0.1,
    timeout: int = 30,
) -> dict:
    """
    Use a remote LLM to judge faithfulness.

    Args:
        documents: Retrieved documents text
        question: Original question
        answer: Model's generated answer
        api_url: vLLM/SGLang compatible API endpoint
        model_name: Model name for the API
        temperature: Generation temperature
        timeout: Request timeout

    Returns:
        dict with score and reason
    """
    if not documents or not answer:
        return {"score": 0.0, "reason": "Missing documents or answer"}

    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        documents=documents[:3000],  # Truncate to avoid context overflow
        question=question,
        answer=answer,
    )

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 200,
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result_text = response.json()["choices"][0]["message"]["content"]
        return _parse_judge_response(result_text)

    except requests.exceptions.ConnectionError:
        logger.warning(f"LLM Judge server not available at {api_url}, returning default score")
        return {"score": 0.0, "reason": "Judge server unavailable"}
    except Exception as e:
        logger.warning(f"LLM Judge error: {e}")
        return {"score": 0.0, "reason": f"Judge error: {str(e)}"}


def extract_entities_remote(
    text: str,
    api_url: str = "http://127.0.0.1:8001/v1/chat/completions",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    timeout: int = 15,
) -> list[str]:
    """
    Use LLM to extract medical entities from text.

    Args:
        text: Text to extract entities from
        api_url: vLLM/SGLang compatible API endpoint
        model_name: Model name for the API
        timeout: Request timeout

    Returns:
        List of extracted entity strings
    """
    if not text:
        return []

    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:2000])

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500,
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result_text = response.json()["choices"][0]["message"]["content"]
        return _parse_entity_response(result_text)

    except Exception as e:
        logger.warning(f"Entity extraction error: {e}")
        return []


# ============================================================
# Local LLM Judge (loads model into memory)
# ============================================================

_local_model = None
_local_tokenizer = None


def load_local_judge(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cpu",
):
    """Load a local judge model (cached globally)."""
    global _local_model, _local_tokenizer

    if _local_model is not None:
        return _local_model, _local_tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading local judge model: {model_name}")
    _local_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _local_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
    )
    _local_model.eval()
    return _local_model, _local_tokenizer


def judge_faithfulness_local(
    documents: str,
    question: str,
    answer: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cpu",
) -> dict:
    """
    Use a local LLM to judge faithfulness.
    NOTE: Not recommended during RL training due to GPU memory conflicts.
    """
    import torch

    if not documents or not answer:
        return {"score": 0.0, "reason": "Missing documents or answer"}

    model, tokenizer = load_local_judge(model_name, device)

    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        documents=documents[:2000],
        question=question,
        answer=answer,
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=False)

    result_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_judge_response(result_text)


# ============================================================
# Response Parsers
# ============================================================

def _parse_judge_response(text: str) -> dict:
    """Parse the LLM judge's JSON response."""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            return {
                "score": score,
                "reason": result.get("reason", ""),
            }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse judge response: {text[:200]}, error: {e}")

    # Fallback: try to find a number in the text
    numbers = re.findall(r"(\d+\.?\d*)", text)
    if numbers:
        score = float(numbers[0])
        if score > 1.0:
            score = score / 10.0  # Handle case where model outputs 7/10 style
        score = max(0.0, min(1.0, score))
        return {"score": score, "reason": "Parsed from text"}

    return {"score": 0.0, "reason": "Failed to parse response"}


def _parse_entity_response(text: str) -> list[str]:
    """Parse the entity extraction response."""
    try:
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            entities = result.get("entities", [])
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if e]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try to extract quoted strings
    entities = re.findall(r'"([^"]+)"', text)
    return entities

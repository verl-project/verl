import logging
import os
import re
import threading
import time
from typing import Any, Optional

import ray
import requests
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _do_checker_http(
    checker_url: str,
    answer: str,
    evidence: str,
    question: str,
    timeout: int,
    max_retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    answer = answer[:600]
    evidence = evidence[:1200]
    question = question[:512]

    payload = {"answer": answer, "evidence": evidence, "question": question}
    last_err: Exception | None = None
    data: dict[str, Any] = {}

    for attempt in range(max(1, max_retries + 1)):
        try:
            resp = requests.post(checker_url, json=payload, timeout=(5, timeout))
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))

    if not data:
        raise last_err or RuntimeError("Checker HTTP call returned empty response.")

    results: list | None = data.get("verification_results")
    if not isinstance(results, list):
        results = data.get("claims")
    if not isinstance(results, list):
        results = data.get("results")
    if not isinstance(results, list):
        results = []

    if not results and "faithfulness" in data:
        faith = float(data.get("faithfulness", 0.0))
        hall = float(data.get("hallucination", 0.0))
        if faith >= 0.8 and hall <= 0.2:
            label = "entail"
        elif hall >= 0.5:
            label = "contradict"
        else:
            label = "neutral"
        results = [{"claim": "overall_answer", "label": label, "confidence": max(faith, 1.0 - hall)}]

    return results, data


@ray.remote
class CheckerExecutionWorker:
    def execute(
        self,
        checker_url: str,
        answer: str,
        evidence: str,
        question: str,
        timeout: int,
        max_retries: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _do_checker_http(checker_url, answer, evidence, question, timeout, max_retries)


class _CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0):
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._open_until == 0.0:
                return False
            if time.time() > self._open_until:
                self._open_until = 0.0
                self._failures = 0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self, is_timeout: bool = False) -> None:
        if not is_timeout:
            return
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._open_until = time.time() + self._cooldown


class MedRAGCheckerTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        self.checker_url = config.get("checker_url", "").strip()
        self.timeout = int(config.get("timeout", 60))
        self.max_retries = int(config.get("max_retries", 0))
        self.checker_node_ip = str(config.get("checker_node_ip", "")).strip()
        self.local_fallback = bool(config.get("local_fallback", False))
        self.checker_model_path = config.get("checker_model_path")
        self.num_workers = int(config.get("num_workers", 2))

        self.call_bonus = float(config.get("call_bonus", 0.0))
        self.contradiction_weight = float(config.get("contradiction_weight", 1.5))
        self.entailment_weight = float(config.get("entailment_weight", 1.0))
        self.neutral_weight = float(config.get("neutral_weight", 0.0))
        self.min_reward_confidence = float(config.get("min_reward_confidence", 0.65))
        self.invalid_penalty = float(config.get("invalid_penalty", -0.10))
        self.reward_min = float(config.get("reward_min", -1.0))
        self.reward_max = float(config.get("reward_max", 1.0))

        self._circuit = _CircuitBreaker(
            threshold=int(config.get("circuit_timeout_threshold", 3)),
            cooldown_s=float(config.get("circuit_cooldown_s", 60.0)),
        )
        self.execution_pool = self._init_execution_pool() if self.checker_url else None

    def _resolve_node_id_by_ip(self, node_ip: str) -> Optional[str]:
        if not node_ip:
            return None
        try:
            for n in ray.nodes():
                if n.get("Alive") and n.get("NodeManagerAddress") == node_ip:
                    return n.get("NodeID")
        except Exception:
            return None
        return None

    def _init_execution_pool(self):
        options: dict[str, Any] = {"max_concurrency": self.num_workers}
        if self.checker_node_ip:
            node_id = self._resolve_node_id_by_ip(self.checker_node_ip)
            if node_id:
                options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        return CheckerExecutionWorker.options(**options).remote()

    @staticmethod
    def _should_skip_check_answer(answer: str) -> tuple[bool, str]:
        x = (answer or "").strip()
        x_low = x.lower()
        if not x:
            return True, "empty_answer"
        if x_low in {"your answer", "my answer", "answer", "placeholder_answer", "n/a", "none", "unknown"}:
            return True, "placeholder_answer"
        if x_low.startswith("verification results"):
            return True, "checker_output_recycled"
        if re.match(r'^\[[\"\']', x) or re.match(r'^\{"query', x):
            return True, "search_query_as_answer"
        if re.match(r'^\{[\\]?[\"\']answer[\\]?[\"\']', x):
            return True, "json_object_as_answer"
        if len(x.split()) <= 2 and not any(p in x for p in ".!?"):
            return True, "too_short"
        return False, ""

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(time.time_ns())
        self._instance_dict[instance_id] = {"reward": [], "last_result": None}
        return instance_id, ToolResponse()

    def _extract_latest_evidence(self, kwargs: dict[str, Any]) -> str:
        tool_results = kwargs.get("tool_results") or []
        for item in reversed(tool_results):
            if getattr(item, "tool_name", None) == "search":
                text = getattr(item, "tool_result", None)
                if text:
                    return str(text)
        return ""

    @staticmethod
    def _extract_text_from_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks)
        return ""

    def _extract_latest_evidence_from_agent_data(self, agent_data: Any) -> str:
        messages = getattr(agent_data, "messages", None) or []
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            text = self._extract_text_from_message_content(msg.get("content", "")).strip()
            if not text:
                continue
            lower = text.lower()
            # Skip prior checker outputs; we only want retrieval evidence here.
            if (
                "'supports':" in lower
                or '"supports":' in lower
                or "'contradictions':" in lower
                or '"contradictions":' in lower
                or "checker skipped" in lower
                or "checker failed" in lower
                or "checker temporarily unavailable" in lower
            ):
                continue
            return text
        return ""

    def _extract_question_from_agent_data(self, agent_data: Any) -> str:
        messages = getattr(agent_data, "messages", None) or []
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            text = self._extract_text_from_message_content(msg.get("content", "")).strip()
            if text:
                return text
        return ""

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        answer = str(parameters.get("answer", "")).strip()
        evidence = str(parameters.get("evidence", "")).strip()
        question = str(kwargs.get("question", "")).strip()
        agent_data = kwargs.get("agent_data")

        if not question and agent_data is not None:
            question = self._extract_question_from_agent_data(agent_data)

        if not evidence:
            evidence = self._extract_latest_evidence(kwargs)
        if not evidence and agent_data is not None:
            evidence = self._extract_latest_evidence_from_agent_data(agent_data)

        should_skip, skip_reason = self._should_skip_check_answer(answer)
        if should_skip:
            msg = f"Checker skipped: {skip_reason}"
            return ToolResponse(text=msg), self.invalid_penalty, {"skip_reason": skip_reason}
        if not evidence.strip():
            msg = "Checker skipped: missing_evidence"
            return ToolResponse(text=msg), self.invalid_penalty, {"skip_reason": "missing_evidence"}

        if self._circuit.is_open():
            return ToolResponse(text="Checker temporarily unavailable."), 0.0, {"circuit_open": True}

        try:
            results, raw = await self.execution_pool.execute.remote(
                self.checker_url, answer, evidence, question, self.timeout, self.max_retries
            )
            self._circuit.record_success()
        except Exception as e:  # noqa: BLE001
            is_timeout = isinstance(e, TimeoutError) or "timeout" in str(e).lower()
            self._circuit.record_failure(is_timeout=is_timeout)
            return ToolResponse(text=f"Checker failed: {e}"), 0.0, {"error": str(e)}

        contradictions = 0
        supports = 0
        neutrals = 0
        weighted_sum = 0.0
        total_confidence = 0.0
        for item in results:
            label = str(item.get("label", "")).lower()
            confidence = float(item.get("confidence", 0.5) or 0.5)
            confidence = max(0.0, min(1.0, confidence))
            total_confidence += confidence
            # Low-confidence checker outputs should be treated as uncertain rather than
            # producing dense reward that can suppress search.
            if confidence < self.min_reward_confidence:
                neutrals += 1
                continue
            if "contrad" in label:
                contradictions += 1
                weighted_sum -= self.contradiction_weight * confidence
            elif "entail" in label or "support" in label:
                supports += 1
                weighted_sum += self.entailment_weight * confidence
            else:
                neutrals += 1
                weighted_sum += self.neutral_weight * confidence

        num_claims = max(1, len(results))
        confidence_score = weighted_sum / num_claims
        reward = self.call_bonus + confidence_score
        reward = max(self.reward_min, min(self.reward_max, reward))

        summary = {
            "supports": supports,
            "contradictions": contradictions,
            "neutrals": neutrals,
            "num_claims": len(results),
            "num_supported": supports,
            "num_contradicted": contradictions,
            "num_neutral": neutrals,
            "support_rate": (supports / num_claims) if results else 0.0,
            "contradiction_rate": (contradictions / num_claims) if results else 0.0,
            "verification_results": results,
            "avg_confidence": (total_confidence / num_claims) if results else 0.0,
            "confidence_score": confidence_score,
            "raw": raw,
        }
        self._instance_dict[instance_id]["reward"].append(summary)
        self._instance_dict[instance_id]["last_result"] = summary
        logger.warning(
            "[CHECKER] answer=%s supports=%s contradictions=%s neutrals=%s num_claims=%s avg_confidence=%.3f reward=%.3f",
            answer[:160],
            supports,
            contradictions,
            neutrals,
            len(results),
            summary["avg_confidence"],
            reward,
        )
        return ToolResponse(text=str(summary)), reward, summary

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return str(self._instance_dict[instance_id]["reward"])

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

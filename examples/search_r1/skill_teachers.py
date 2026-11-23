"""Static skill contexts and conservative token masks for Qwen ChatML Search-R1.

No response is retokenized: generated token IDs are also the teacher targets.
Byte offsets allow masks to handle noncanonical BPE and Unicode generations.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import torch

TEACHERS = ("entity", "planning", "query", "answer")
GUIDANCE = (
    "Follow these search guidelines while continuing the assistant response in the original format. "
    "These are general procedures, not factual evidence. Do not write a critique or evaluate the response.\n"
)
TAG = re.compile(rb"</?(think|search|answer)>")


def byte_level_decoder():
    """Inverse GPT-2/Qwen byte alphabet, independent of Transformers internals.

    Visible Latin-1 bytes keep their code points; the remaining bytes map to
    consecutive code points starting at 256. Decode bytes before UTF-8 so even
    tokens splitting a multibyte character retain their exact offsets.
    """
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    decoder = {chr(byte): byte for byte in visible}
    next_codepoint = 256
    for byte in range(256):
        if byte not in visible:
            decoder[chr(next_codepoint)] = byte
            next_codepoint += 1
    return decoder


def load_bank(path):
    raw = Path(path).read_bytes()
    bank = json.loads(raw)
    if set(bank.get("teachers", {})) != set(TEACHERS):
        raise ValueError("Skill bank must contain exactly entity, planning, query and answer teachers")
    contexts = {}
    for name in TEACHERS:
        rules = bank["teachers"][name].get("rules")
        if not isinstance(rules, list) or not rules or any(not isinstance(r, str) or not r.strip() for r in rules):
            raise ValueError(f"Teacher {name} must contain nonempty text rules")
        # Chat role delimiters must never be injected through rule content.
        if any("<|" in r or "|>" in r for r in rules):
            raise ValueError("Skill rules cannot contain chat control tokens")
        contexts[name] = GUIDANCE + "\n".join(f"- {r}" for r in rules)
    contexts["combined"] = GUIDANCE + "\n".join(
        f"- {rule}" for name in TEACHERS for rule in bank["teachers"][name]["rules"]
    )
    return bank, contexts, hashlib.sha256(raw).hexdigest()


def validate_settings(config):
    """Fail before allocating workers for configurations outside this first recipe."""
    skill = config.skill_distillation
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("Skill teachers require the synchronous V1 trainer")
    if config.trainer.v1.sync.get("parameter_sync_step", 1) != 1:
        raise ValueError("Skill teachers require parameter_sync_step=1 for an on-policy snapshot")
    if actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("Skill teachers currently support FSDP/FSDP2 only")
    if config.algorithm.adv_estimator != "grpo" or config.algorithm.use_kl_in_reward:
        raise ValueError("This recipe requires GRPO with final task rewards, without reward-side KL")
    if config.get("distillation", {}).get("enabled", False):
        raise ValueError("Static self-teachers cannot be combined with external-teacher distillation")
    for extension in ("sd_search", "turn_hidden_diversity"):
        if config.get(extension, {}).get("enabled", False):
            raise ValueError(f"Skill forward KL cannot share the actor loss hook with {extension}")
    if rollout.agent.default_agent_loop != "search_r1":
        raise ValueError("Skill masks require the Search-R1 agent loop")
    if rollout.temperature != 1.0 or rollout.top_p != 1.0 or rollout.top_k != -1:
        raise ValueError("The first skill recipe requires temperature=1, top_p=1, top_k=-1")
    if config.algorithm.get("rollout_correction", {}).get("bypass_mode", False):
        raise ValueError("Skill teachers require recomputed old-policy log probabilities")
    if skill.mode not in {"split", "combined", "all"}:
        raise ValueError("skill_distillation.mode must be split, combined or all")
    if skill.get("objective") != "forward_kl_topk_tail":
        raise ValueError("Skill teachers require forward_kl_topk_tail")
    if isinstance(skill.top_k, bool) or not isinstance(skill.top_k, int) or skill.top_k < 1:
        raise ValueError("skill_distillation.top_k must be a positive integer")
    if not math.isfinite(skill.min_topk_overlap) or not 0 <= skill.min_topk_overlap <= 1:
        raise ValueError("min_topk_overlap must be in [0, 1]")
    if not math.isfinite(skill.prefix_fraction) or not 0 < skill.prefix_fraction <= 1:
        raise ValueError("prefix_fraction must be in (0, 1]")
    if set(skill.answer_component_weights) != {"think", "decision", "content"}:
        raise ValueError("answer_component_weights must contain think, decision and content")
    if any(not math.isfinite(w) or w < 0 for w in skill.answer_component_weights.values()):
        raise ValueError("Answer component weights must be finite and nonnegative")
    if config.actor_rollout_ref.model.get("use_fused_kernels", False):
        raise ValueError("Skill forward KL requires use_fused_kernels=False")
    if actor.get("fsdp_config", {}).get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("Skill forward KL currently requires Ulysses SP=1")
    for name, value in {"beta": skill.beta, **dict(skill.weights)}.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Skill coefficient {name} must be finite and nonnegative")
    for name in ("warmup_steps", "teacher_micro_batch_size_per_gpu", "max_teacher_context_length"):
        value = skill[name]
        minimum = 0 if name == "warmup_steps" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid skill setting: {name}")
    active = ("combined",) if skill.mode == "combined" else TEACHERS
    if any(name not in skill.weights for name in active):
        raise ValueError("Missing teacher loss weight")


class SkillTokenCodec:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.byte_decoder = byte_level_decoder()
        self.special_ids = set(tokenizer.all_special_ids)
        self.start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.start not in self.special_ids or self.end not in self.special_ids or self.start == self.end:
            raise ValueError("Skill teachers require a Qwen ChatML byte-level tokenizer")
        self.system_header = tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
        self._bytes = {}

    def token_bytes(self, token):
        if token not in self._bytes:
            piece = self.tokenizer.convert_ids_to_tokens(token)
            if token in self.special_ids:
                value = piece.encode("utf-8")
            else:
                try:
                    value = bytes(self.byte_decoder[c] for c in piece)
                except KeyError as error:
                    raise ValueError("Skill masks require byte-level BPE token pieces") from error
            self._bytes[token] = value
        return self._bytes[token]

    def teacher_prompt(self, prompt, context):
        """Insert rules into the system message without changing any original IDs."""
        if not prompt or prompt[0] != self.start:
            raise ValueError("Expected an unpadded ChatML prompt")
        if prompt[: len(self.system_header)] == self.system_header:
            insertion = self.tokenizer.encode(context + "\n\n", add_special_tokens=False)
            return prompt[: len(self.system_header)] + insertion + prompt[len(self.system_header) :]
        prefix = self.tokenizer.encode("<|im_start|>system\n" + context + "<|im_end|>\n", add_special_tokens=False)
        return prefix + prompt

    def span_masks(self, responses, response_mask):
        scopes = self.assistant_masks(responses, response_mask)
        return scopes["think"], scopes["query"]

    def assistant_masks(self, responses, response_mask):
        """Only unambiguous content tokens inside assistant think/search spans.

        Tokens crossing XML boundaries are left to GRPO. Unclosed spans are
        accepted up to the end of an assistant run (generation truncation).
        Malformed/nested XML invalidates that run's auxiliary masks.
        """
        ids = responses.tolist() if isinstance(responses, torch.Tensor) else list(responses)
        generated = response_mask.tolist() if isinstance(response_mask, torch.Tensor) else list(response_mask)
        if len(ids) != len(generated):
            raise ValueError("Response mask/token length mismatch")
        scopes = {name: torch.zeros(len(ids), dtype=torch.bool) for name in ("think", "query", "answer", "decision")}
        begin = 0
        while begin < len(ids):
            if not generated[begin] or ids[begin] in self.special_ids:
                begin += 1
                continue
            end = begin
            while end < len(ids) and generated[end] and ids[end] not in self.special_ids:
                end += 1
            pieces = [self.token_bytes(token) for token in ids[begin:end]]
            text = b"".join(pieces)
            state, content_start, valid, spans, decisions = None, 0, True, [], []
            for match in TAG.finditer(text):
                closing, kind = match[0].startswith(b"</"), match[1]
                if closing:
                    if state != kind:
                        valid = False
                        break
                    spans.append((state, content_start, match.start()))
                    state = None
                else:
                    if state is not None:
                        valid = False
                        break
                    state, content_start = kind, match.end()
                    if kind in {b"search", b"answer"}:
                        decisions.append((match.start() + 1, match.end() - 1))
            if state is not None and valid:
                spans.append((state, content_start, len(text)))
            if valid:
                offset = 0
                for index, piece in enumerate(pieces, begin):
                    finish = offset + len(piece)
                    for kind, left, right in spans:
                        if left <= offset < finish <= right:
                            if kind == b"think":
                                scopes["think"][index] = True
                            elif kind == b"search":
                                scopes["query"][index] = True
                            elif kind == b"answer":
                                scopes["answer"][index] = True
                    if any(left <= offset < finish <= right for left, right in decisions):
                        scopes["decision"][index] = True
                    offset = finish
            begin = end
        return scopes


def teacher_masks(think, query, response_mask, mode, answer=None, decision=None):
    answer = torch.zeros_like(think) if answer is None else answer
    decision = torch.zeros_like(think) if decision is None else decision
    if mode == "combined":
        return {"combined": think | query | answer | decision}
    if mode == "all":
        return {name: response_mask.bool().clone() for name in TEACHERS}
    return {"entity": think | query, "planning": think, "query": query, "answer": think | decision | answer}


def beta_at_step(beta, warmup_steps, step):
    return beta * min(1.0, max(0, step) / warmup_steps) if warmup_steps else beta


def clipped_skill_loss(log_prob, old_log_prob, advantage, mask, count, dp_size, clip_low, clip_high):
    """One teacher's PPO surrogate, normalized over its global mini-batch tokens."""
    if count <= 0:
        return log_prob.sum() * 0.0
    # Select before exponentiation: masked tool positions must not introduce NaNs.
    selected = mask.bool()
    delta = log_prob[selected] - old_log_prob.detach()[selected]
    ratio = torch.exp(torch.clamp(delta, -20.0, 20.0))
    advantage = advantage.detach()[selected]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_low, 1 + clip_high) * advantage)
    return -surrogate.sum() * dp_size / count

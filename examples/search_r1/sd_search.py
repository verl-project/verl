"""SD-Search for synchronous verl V1. Extra forwards favor memory over throughput."""

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .protocol import normalize_answer, parse_action
from .sd_hindsight import build_steps, queries
from .sd_losses import sd_coefficient, topk_union_jsd
from .skill_teachers import SkillTokenCodec

MASK = "sd_trajectory_mask"


def answer_f1(final, ground_truth, termination):
    kind, answer = parse_action(final)
    if kind != "answer" or termination != "answer":
        return 0.0
    targets = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    if isinstance(targets, str):
        targets = [targets]
    a = normalize_answer(answer).split()
    best = 0.0
    for target in targets:
        b = normalize_answer(target).split()
        n = sum((Counter(a) & Counter(b)).values())
        if n:
            best = max(best, 2 * n / (len(a) + len(b)))
    return best


def validate_settings(config):
    c = config.sd_search
    a = config.actor_rollout_ref.actor
    m = config.actor_rollout_ref.model
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("SD requires synchronous V1")
    if config.algorithm.adv_estimator != "grpo" or a.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("SD requires GRPO and FSDP")
    if m.use_remove_padding or m.get("use_fused_kernels", False) or a.fsdp_config.ulysses_sequence_parallel_size != 1:
        raise ValueError("SD currently requires padded, unfused, SP=1")
    for name in ("skill_distillation", "turn_hidden_diversity", "distillation"):
        if config.get(name, {}).get("enabled", False):
            raise ValueError("SD is isolated from other auxiliary algorithms")
    if config.trainer.v1.sync.get("parameter_sync_step", 1) != 1:
        raise ValueError("SD requires on-policy sync=1")
    if config.actor_rollout_ref.rollout.agent.default_agent_loop != "search_r1":
        raise ValueError("SD requires Search-R1 protocol")
    if c.outcome_threshold != 0:
        raise ValueError("This reproduction uses paper rho=0")
    if not math.isfinite(c.alpha) or c.alpha < 0:
        raise ValueError("Invalid alpha")
    for key in (
        "warmup_steps",
        "top_k",
        "slots",
        "max_hindsight_tokens",
        "max_context_tokens",
        "teacher_micro_batch_size_per_gpu",
    ):
        if type(c[key]) is not int or c[key] < (0 if key == "warmup_steps" else 1):
            raise ValueError("Invalid " + key)
    if config.actor_rollout_ref.rollout.temperature != 1:
        raise ValueError("SD requires temperature=1")


class HindsightSupervisor:
    def __init__(self, config):
        validate_settings(config)
        self.c = config.sd_search
        self.group_size = config.actor_rollout_ref.rollout.n

    def initialize(self, trainer):
        self.codec = SkillTokenCodec(trainer.tokenizer)
        path = Path(trainer.config.trainer.default_local_dir) / "sd-search.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        from omegaconf import OmegaConf

        manifest = {
            "paper": "2605.18299v1",
            "settings": OmegaConf.to_container(self.c),
            "task_reward": "EM",
            "outcome_labels": "normalized answer F1 > 0",
            "deviations": [
                "multi-message ChatML retained",
                "separate per-search student forwards after PPO backward",
                "1024-token hindsight tail truncation; paper does not specify direction",
            ],
        }
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Run has different SD settings")
        path.write_text(json.dumps(manifest, indent=2))

    def prepare(self, trainer, batch, metrics):
        import transfer_queue as tq
        from tensordict import TensorDict

        started = time.monotonic()
        alpha = sd_coefficient(trainer.global_steps - 1, self.c.warmup_steps, self.c.alpha)
        batch.extra_info.update(sd_alpha=alpha)
        metrics["sd/alpha"] = alpha
        if not alpha:
            return batch
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["prompts", "responses", "response_mask", "reward_model", "extra_fields"],
        )
        groups = defaultdict(list)
        for i, key in enumerate(batch.keys):
            if not batch.tags[i].get("is_padding", False):
                groups[key.split("_")[0]].append(i)
        if any(len(g) != self.group_size for g in groups.values()):
            raise ValueError("Incomplete SD rollout groups")
        spans = []
        outcomes = []
        for response, mask, rm, extra in zip(
            data["responses"].unbind(),
            data["response_mask"].unbind(),
            data["reward_model"].tolist(),
            data["extra_fields"].tolist(),
            strict=True,
        ):
            spans.append(queries(self.codec, response.tolist(), mask.tolist()))
            outcomes.append(
                answer_f1(extra["search_r1_final_assistant"], rm["ground_truth"], extra["search_r1_termination"])
            )
        prompts = data["prompts"].unbind()
        responses = data["responses"].unbind()
        rows = [[] for _ in batch.keys]
        for indices in groups.values():
            for focal, i in enumerate(indices):
                rows[i] = build_steps(
                    self.codec,
                    prompts[i].tolist(),
                    responses[i].tolist(),
                    [spans[j] for j in indices],
                    [outcomes[j] for j in indices],
                    focal,
                    self.c.max_hindsight_tokens,
                    self.c.max_context_tokens,
                )
        if any(len(row) > self.c.slots for row in rows):
            raise ValueError("SD query slots exceeded")
        fields = {
            MASK: torch.nested.as_nested_tensor(
                [torch.tensor([not batch.tags[i].get("is_padding", False)]) for i in range(len(rows))],
                layout=torch.jagged,
            )
        }
        pad = self.codec.tokenizer.pad_token_id
        for slot in range(self.c.slots):
            for key in ("student_ids", "teacher_ids", "student_positions", "teacher_positions"):
                dummy = [pad, pad] if key.endswith("ids") else [1]
                values = [torch.tensor(row[slot][key] if slot < len(row) else dummy, dtype=torch.long) for row in rows]
                fields[f"sd_{slot}_{key}"] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
            fields[f"sd_{slot}_weights"] = torch.nested.as_nested_tensor(
                [
                    torch.full(
                        (len(row[slot]["student_positions"]),),
                        1 / max(1, sum(len(x["student_positions"]) for x in row)),
                    )
                    if slot < len(row)
                    else torch.zeros(1)
                    for row in rows
                ],
                layout=torch.jagged,
            )
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=TensorDict(fields, batch_size=[len(batch)])
        )
        batch.fields = None
        batch.extra_info.update(
            sd_slots=self.c.slots,
            sd_top_k=self.c.top_k,
            sd_micro_batch=self.c.teacher_micro_batch_size_per_gpu,
            auxiliary_loss_mask_names=[MASK],
        )
        metrics.update(
            {
                "sd/query_tokens": sum(len(q.positions) for x in spans for q in x),
                "sd/hindsight_truncated_tokens": sum(x["hindsight_truncated_tokens"] for row in rows for x in row),
                "sd/prepare_seconds": time.monotonic() - started,
                "sd/groups": len(groups),
            }
        )
        return batch


def query_logits(model, data, slot, role, indices):
    ids = [data[f"sd_{slot}_{role}_ids"].unbind()[i] for i in indices]
    positions = [data[f"sd_{slot}_{role}_positions"].unbind()[i] for i in indices]
    lengths = torch.tensor([len(x) for x in ids], device=ids[0].device)
    padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
    output = model(
        input_ids=padded, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp_min(0), use_cache=False
    )
    return torch.cat([output.logits[j, p - 1] for j, p in enumerate(positions)])


def backward_sd(engine, data):
    """Exact current-parameter teacher, no gradient; student gets JSD gradients.

    Fixed slot count and fixed auxiliary microbatch splits keep FSDP collective
    ordering equal even on ranks with no queries. Empty trajectories stay in the
    optimizer minibatch denominator (paper Eq 6), and padding is excluded.
    """
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric

    alpha = tu.get(data, "sd_alpha")
    slots = tu.get(data, "sd_slots")
    size = tu.get(data, "sd_micro_batch")
    count = tu.get(data, "auxiliary_loss_mask_counts")[MASK]
    scale = data["dp_size"] / max(1, count)
    started = time.monotonic()
    total = 0.0
    training = engine.module.training
    for start in range(0, data.batch_size[0], size):
        indices = list(range(start, min(start + size, data.batch_size[0])))
        for slot in range(slots):
            try:
                with torch.autocast(
                    device_type=next(engine.module.parameters()).device.type,
                    dtype=torch.bfloat16,
                    enabled=next(engine.module.parameters()).is_cuda,
                ):
                    engine.module.eval()
                    with torch.no_grad():
                        teacher = query_logits(engine.module, data, slot, "teacher", indices)
                    engine.module.train(training)
                    student = query_logits(engine.module, data, slot, "student", indices)
                    weights = torch.cat([data[f"sd_{slot}_weights"].unbind()[i] for i in indices])
                    loss = (topk_union_jsd(student, teacher, tu.get(data, "sd_top_k")) * weights).sum() * scale
                weighted = alpha * loss
                if getattr(engine, "scaler", None) is not None:
                    engine.scaler.scale(weighted).backward()
                else:
                    weighted.backward()
                total += loss.detach().item()
                del teacher, student, loss, weighted
            finally:
                engine.module.train(training)
    return {
        "sd/jsd": Metric(value=total, aggregation=AggregationType.SUM),
        "sd/weighted_loss": Metric(value=alpha * total, aggregation=AggregationType.SUM),
        "sd/auxiliary_seconds": Metric(value=time.monotonic() - started, aggregation=AggregationType.SUM),
    }

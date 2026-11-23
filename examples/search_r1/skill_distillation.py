"""Frozen skill self-teachers: overlap-gated forward KL with Reset Support."""

import json
import math
import time
from functools import partial
from pathlib import Path
from uuid import uuid4

import torch

from .skill_distributions import overlap_prefix_gate
from .skill_teachers import TEACHERS, SkillTokenCodec, beta_at_step, load_bank, teacher_masks, validate_settings


def skill_ppo_loss(config, weights, model_output, data, dp_group=None, answer_components=None):
    """Add forward KL to the unchanged task PPO/reference loss."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding

    loss, metrics = ppo_loss(config, model_output, data, dp_group)
    beta = tu.get(data, "skill_beta")
    counts = tu.get(data, "auxiliary_loss_mask_counts")
    for name, weight in weights.items():
        per_token = no_padding_2_padding(model_output[f"skill_{name}_forward_kl"], data)
        components = answer_components if name == "answer" and answer_components else {None: 1.0}
        auxiliary = per_token.sum() * 0
        for component, component_weight in components.items():
            key = f"skill_{name}_{component}_mask" if component else f"skill_{name}_mask"
            mask = data.select(key).to_padded_tensor()[key].bool()
            count = counts[key]
            value = per_token[mask].sum() * data["dp_size"] / count if count > 0 else per_token.sum() * 0
            auxiliary = auxiliary + component_weight * value
            if component:
                metrics[f"skill/{name}/{component}_loss"] = Metric(
                    value=value.detach(), aggregation=AggregationType.SUM
                )
        loss = loss + beta * weight * auxiliary
        metrics[f"skill/{name}/loss"] = Metric(value=auxiliary.detach(), aggregation=AggregationType.SUM)
    metrics["skill/beta"] = beta
    return loss, metrics


class SkillTeacherSupervisor:
    """Score frozen distributions before any update; gates never reset at turns.

    All generated assistant tokens (except chat special tokens) are divergence
    checkpoints, even outside a teacher's loss scope. Tool/padding cannot trip
    a gate. Stop state is independent for every teacher and trajectory.
    """

    def __init__(self, config):
        validate_settings(config)
        self.config = config.skill_distillation
        self.bank, self.contexts, self.bank_sha256 = load_bank(self.config.bank_path)
        self.names = ("combined",) if self.config.mode == "combined" else TEACHERS
        self.weights = {name: float(self.config.weights[name]) for name in self.names}
        self.components = dict(self.config.answer_component_weights) if self.config.mode == "split" else None
        self.codec = None

    def initialize(self, trainer):
        from verl.utils.config import omega_conf_to_dataclass

        self.codec = SkillTokenCodec(trainer.tokenizer)
        manifest = {
            "bank_sha256": self.bank_sha256,
            "bank": self.bank,
            "mode": self.config.mode,
            "weights": self.weights,
            "beta": self.config.beta,
            "warmup_steps": self.config.warmup_steps,
            "objective": self.config.objective,
            "top_k": self.config.top_k,
            "min_topk_overlap": self.config.min_topk_overlap,
            "prefix_fraction": self.config.prefix_fraction,
            "gate_scope": "all_generated_assistant_tokens_excluding_specials",
            "gate_action": "exclude_first_failure_and_all_subsequent_tokens_without_turn_reset",
            "answer_component_weights": self.components,
        }
        path = Path(trainer.config.trainer.default_local_dir) / "skill-teachers.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Existing run has a different skill bank/objective; use a new output directory")
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        actor_config = omega_conf_to_dataclass(trainer.config.actor_rollout_ref.actor)
        trainer.actor_rollout_wg.set_loss_fn(
            partial(skill_ppo_loss, config=actor_config, weights=self.weights, answer_components=self.components)
        )

    def update_metadata(self, batch, step):
        names = [f"skill_{name}_mask" for name in self.names]
        if self.components:
            names.extend(f"skill_answer_{component}_mask" for component in self.components)
        batch.extra_info.update(
            skill_beta=beta_at_step(self.config.beta, self.config.warmup_steps, step),
            auxiliary_loss_mask_names=names,
            skill_forward_kl_names=tuple(self.names),
        )

    def score(self, trainer, batch, metrics):
        import transfer_queue as tq
        from tensordict import TensorDict

        from verl.workers.utils.padding import response_from_nested

        def nested_batch(rows):
            return TensorDict(
                {
                    key: torch.nested.as_nested_tensor([row[key] for row in rows], layout=torch.jagged)
                    for key in rows[0]
                },
                batch_size=[len(rows)],
            )

        started = time.monotonic()
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses", "response_mask"]
        )
        records, scopes, masks, watches = [], [], [], []
        stored = [{} for _ in batch.keys]
        for i, (prompt, response, generated) in enumerate(
            zip(data["prompts"].unbind(), data["responses"].unbind(), data["response_mask"].unbind(), strict=True)
        ):
            is_padding = batch.tags[i].get("is_padding", False)
            generated = torch.zeros_like(generated).bool() if is_padding else generated.bool()
            scope = self.codec.assistant_masks(response, generated)
            scopes.append(scope)
            masks.append(
                teacher_masks(
                    scope["think"], scope["query"], generated, self.config.mode, scope["answer"], scope["decision"]
                )
            )
            watches.append(generated & torch.tensor([int(t) not in self.codec.special_ids for t in response]))
            records.append((prompt.tolist(), response, generated, is_padding))

        def collect(name):
            fields, tags = [], []
            for prompt, response, generated, is_padding in records:
                teacher_prompt = torch.tensor(
                    prompt
                    if name == "student" or is_padding
                    else self.codec.teacher_prompt(prompt, self.contexts[name]),
                    dtype=torch.long,
                )
                ids = torch.cat((teacher_prompt, response))
                if len(ids) > self.config.max_teacher_context_length:
                    raise ValueError(f"{name} input exceeds max_teacher_context_length; no truncation allowed")
                fields.append(
                    {
                        "prompts": teacher_prompt,
                        "responses": response,
                        "input_ids": ids,
                        "position_ids": torch.arange(len(ids)),
                        "response_mask": generated,
                        "loss_mask": generated,
                    }
                )
                tags.append({"seq_len": len(ids), "prompt_len": len(teacher_prompt), "response_len": len(response)})
            partition = f"skill-distribution-{uuid4().hex}"
            try:
                temporary = tq.kv_batch_put(
                    keys=batch.keys, partition_id=partition, tags=tags, fields=nested_batch(fields)
                )
                temporary.extra_info.update(
                    calculate_entropy=False,
                    compute_loss=False,
                    temperature=1.0,
                    use_dynamic_bsz=False,
                    micro_batch_size_per_gpu=self.config.teacher_micro_batch_size_per_gpu,
                    skill_collect_topk=self.config.top_k,
                    skill_forward_kl_names=(),
                )
                output = trainer.actor_rollout_wg.compute_log_prob(temporary)
                if len(output) != len(batch):
                    raise RuntimeError("Skill distribution scoring changed the batch size")
                keys = ["skill_topk_ids", "skill_topk_logps", "skill_tail_logp"]
                result = tq.kv_batch_get(
                    keys=batch.keys, partition_id=partition, select_fields=keys + ["response_mask"]
                )
                rows = [response_from_nested(result[key], result["response_mask"]).unbind() for key in keys]
                return list(zip(*rows, strict=True))
            finally:
                tq.kv_clear(keys=batch.keys, partition_id=partition)

        student_started = time.monotonic()
        student = collect("student")
        metrics["skill/student/scoring_seconds"] = time.monotonic() - student_started
        for name in self.names:
            tick = time.monotonic()
            teacher = collect(name)
            overlaps, stops, before_count, after_count, prefix_drops = [], [], 0, 0, 0
            for i, ((student_ids, _, _), (teacher_ids, logps, tail)) in enumerate(zip(student, teacher, strict=True)):
                expected = (len(records[i][1]), self.config.top_k)
                if student_ids.shape != expected or teacher_ids.shape != expected or logps.shape != expected:
                    raise ValueError("Teacher/student top-k targets are misaligned")
                if tail.shape != expected[:1] or not torch.isfinite(logps).all() or not torch.isfinite(tail).all():
                    raise ValueError("Teacher probabilities are misaligned or nonfinite")
                prefix, overlap, stop = overlap_prefix_gate(
                    student_ids, teacher_ids, watches[i], self.config.min_topk_overlap
                )
                generated = records[i][2]
                cutoff = math.ceil(int(generated.sum()) * self.config.prefix_fraction)
                window = generated.long().cumsum(0) <= cutoff
                base_mask = masks[i][name]
                mask = base_mask & prefix & window
                before_count += int(base_mask.sum())
                after_count += int(mask.sum())
                prefix_drops += int((base_mask & ~window).sum())
                overlaps.append(overlap[watches[i]])
                if watches[i].any():
                    stops.append(stop)
                stored[i].update(
                    {
                        f"skill_{name}_mask": mask,
                        f"skill_{name}_ids": teacher_ids.detach(),
                        f"skill_{name}_logps": logps.detach(),
                        f"skill_{name}_tail": tail.detach(),
                    }
                )
                if name == "answer" and self.components:
                    for component, scope in {"think": "think", "decision": "decision", "content": "answer"}.items():
                        stored[i][f"skill_answer_{component}_mask"] = mask & scopes[i][scope]
            overlap = torch.cat(overlaps)
            triggered = [position for position in stops if position >= 0]
            metrics.update(
                {
                    f"skill/{name}/candidate_tokens": before_count,
                    f"skill/{name}/tokens": after_count,
                    f"skill/{name}/kept_fraction": after_count / max(1, before_count),
                    f"skill/{name}/overlap_mean": overlap.mean().item() if overlap.numel() else 0.0,
                    f"skill/{name}/stopped_trajectories": len(triggered),
                    f"skill/{name}/stopped_fraction": len(triggered) / max(1, len(stops)),
                    f"skill/{name}/first_stop_response_index_mean": sum(triggered) / len(triggered)
                    if triggered
                    else -1.0,
                    f"skill/{name}/prefix_window_dropped_tokens": prefix_drops,
                    f"skill/{name}/scoring_seconds": time.monotonic() - tick,
                }
            )
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=nested_batch(stored))
        batch.fields = None
        metrics.update(
            {
                "skill/scoring_seconds": time.monotonic() - started,
                "skill/think_tokens": sum(int(s["think"].sum()) for s in scopes),
                "skill/query_tokens": sum(int(s["query"].sum()) for s in scopes),
                "skill/answer_tokens": sum(int(s["answer"].sum()) for s in scopes),
                "skill/decision_tokens": sum(int(s["decision"].sum()) for s in scopes),
                "skill/beta": beta_at_step(self.config.beta, self.config.warmup_steps, trainer.global_steps),
            }
        )
        return batch

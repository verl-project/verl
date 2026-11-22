"""V1 driver and loss integration for static Search-R1 self-teachers."""

import json
import time
from functools import partial
from pathlib import Path
from uuid import uuid4

import torch

from .skill_teachers import (
    TEACHERS,
    SkillTokenCodec,
    beta_at_step,
    clipped_skill_loss,
    load_bank,
    teacher_masks,
    validate_settings,
)


def skill_ppo_loss(config, weights, model_output, data, dp_group=None):
    """Retain the original GRPO/entropy/reference loss, add separate skill losses."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding

    loss, metrics = ppo_loss(config, model_output, data, dp_group)
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    beta = tu.get(data, "skill_beta")
    counts = tu.get(data, "auxiliary_loss_mask_counts")
    old = data.select("old_log_probs").to_padded_tensor()["old_log_probs"]
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    for name, weight in weights.items():
        mask_key, advantage_key = f"skill_{name}_mask", f"skill_{name}_advantages"
        padded = data.select(mask_key, advantage_key).to_padded_tensor()
        auxiliary = clipped_skill_loss(
            log_prob,
            old,
            padded[advantage_key],
            padded[mask_key],
            counts[mask_key],
            data["dp_size"],
            clip_low,
            clip_high,
        )
        loss = loss + beta * weight * auxiliary
        metrics[f"skill/{name}/loss"] = Metric(value=auxiliary.detach(), aggregation=AggregationType.SUM)
    metrics["skill/beta"] = beta
    return loss, metrics


class SkillTeacherSupervisor:
    """Scores all three contexts before the actor's first optimizer update.

    The unchanged actor is the old-policy snapshot; its eval-mode forwards are
    no-grad. Teacher inputs live in a separate temporary TransferQueue partition.
    """

    def __init__(self, config):
        validate_settings(config)
        self.config = config.skill_distillation
        self.bank, self.contexts, self.bank_sha256 = load_bank(self.config.bank_path)
        self.names = ("combined",) if self.config.mode == "combined" else TEACHERS
        self.weights = {name: float(self.config.weights[name]) for name in self.names}
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
        }
        path = Path(trainer.config.trainer.default_local_dir) / "skill-teachers.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Existing run has a different skill bank/objective; use a new output directory")
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        actor_config = omega_conf_to_dataclass(trainer.config.actor_rollout_ref.actor)
        trainer.actor_rollout_wg.set_loss_fn(partial(skill_ppo_loss, config=actor_config, weights=self.weights))

    def update_metadata(self, batch, step):
        batch.extra_info.update(
            skill_beta=beta_at_step(self.config.beta, self.config.warmup_steps, step),
            auxiliary_loss_mask_names=[f"skill_{name}_mask" for name in self.names],
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
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["prompts", "responses", "response_mask", "old_log_probs"],
        )
        records, masks, stored = [], [], [{} for _ in batch.keys]
        think_count = query_count = generated_count = 0
        old_probs = data["old_log_probs"].unbind()
        for i, (prompt, response, generated) in enumerate(
            zip(data["prompts"].unbind(), data["responses"].unbind(), data["response_mask"].unbind(), strict=True)
        ):
            response_mask = generated.bool()
            is_padding = batch.tags[i].get("is_padding", False)
            if is_padding:
                response_mask = torch.zeros_like(response_mask)
            think, query = self.codec.span_masks(response, response_mask)
            think_count += int(think.sum())
            query_count += int(query.sum())
            generated_count += int(response_mask.sum())
            masks.append(teacher_masks(think, query, response_mask, self.config.mode))
            records.append((prompt.tolist(), response, response_mask, is_padding))

        for name in self.names:
            teacher_started = time.monotonic()
            fields, tags = [], []
            for prompt, response, response_mask, is_padding in records:
                teacher_prompt = torch.tensor(
                    prompt if is_padding else self.codec.teacher_prompt(prompt, self.contexts[name]), dtype=torch.long
                )
                input_ids = torch.cat((teacher_prompt, response))
                if len(input_ids) > self.config.max_teacher_context_length:
                    raise ValueError(
                        f"{name} teacher input has {len(input_ids)} tokens, exceeding "
                        f"max_teacher_context_length={self.config.max_teacher_context_length}; no truncation allowed"
                    )
                fields.append(
                    {
                        "prompts": teacher_prompt,
                        "responses": response,
                        "input_ids": input_ids,
                        "position_ids": torch.arange(len(input_ids)),
                        "response_mask": response_mask,
                        "loss_mask": response_mask,
                    }
                )
                tags.append(
                    {"seq_len": len(input_ids), "prompt_len": len(teacher_prompt), "response_len": len(response)}
                )
            partition = f"skill-teacher-{uuid4().hex}"
            try:
                teacher_batch = tq.kv_batch_put(
                    keys=batch.keys,
                    partition_id=partition,
                    tags=tags,
                    fields=nested_batch(fields),
                )
                teacher_batch.extra_info.update(
                    calculate_entropy=False,
                    compute_loss=False,
                    temperature=1.0,
                    use_dynamic_bsz=False,
                    micro_batch_size_per_gpu=self.config.teacher_micro_batch_size_per_gpu,
                )
                output = trainer.actor_rollout_wg.compute_log_prob(teacher_batch)
                if len(output) != len(batch):
                    raise RuntimeError("Teacher scoring changed the batch size")
                result = tq.kv_batch_get(
                    keys=batch.keys,
                    partition_id=partition,
                    select_fields=["log_probs", "response_mask"],
                )
                teacher_probs = response_from_nested(result["log_probs"], result["response_mask"]).unbind()
                gaps = []
                for i in range(len(batch)):
                    mask = masks[i][name]
                    advantage = (teacher_probs[i].float() - old_probs[i].float()).detach()
                    if advantage.shape != mask.shape or not torch.isfinite(advantage[mask]).all():
                        raise ValueError("Teacher scores are misaligned or nonfinite")
                    stored[i][f"skill_{name}_mask"] = mask
                    stored[i][f"skill_{name}_advantages"] = advantage.masked_fill(~mask, 0)
                    gaps.append(advantage[mask])
                values = torch.cat(gaps)
                metrics[f"skill/{name}/tokens"] = values.numel()
                metrics[f"skill/{name}/gap_mean"] = values.mean().item() if values.numel() else 0.0
                metrics[f"skill/{name}/gap_abs_mean"] = values.abs().mean().item() if values.numel() else 0.0
                metrics[f"skill/{name}/positive_fraction"] = (
                    (values > 0).float().mean().item() if values.numel() else 0.0
                )
                metrics[f"skill/{name}/scoring_seconds"] = time.monotonic() - teacher_started
                metrics[f"skill/{name}/scoring_tokens"] = sum(tag["seq_len"] for tag in tags)
            finally:
                tq.kv_clear(keys=batch.keys, partition_id=partition)

        if self.config.mode != "combined":
            for other in ("planning", "query"):
                disagreements, overlaps = 0, 0
                for record in stored:
                    overlap = record["skill_entity_mask"] & record[f"skill_{other}_mask"]
                    product = record["skill_entity_advantages"] * record[f"skill_{other}_advantages"]
                    disagreements += int(((product < 0) & overlap).sum())
                    overlaps += int(overlap.sum())
                metrics[f"skill/entity_{other}_conflict_fraction"] = disagreements / max(1, overlaps)
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=nested_batch(stored))
        # Subsequent workers must receive auxiliary columns as well as existing RL data.
        batch.fields = None
        metrics.update(
            {
                "skill/think_tokens": think_count,
                "skill/query_tokens": query_count,
                "skill/generated_tokens": generated_count,
                "skill/scoring_seconds": time.monotonic() - started,
                "skill/beta": beta_at_step(self.config.beta, self.config.warmup_steps, trainer.global_steps),
            }
        )
        return batch

"""Adjacent search-decision repulsion on the actor's final pre-LM-head states."""

import math
import re
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F

from .protocol import parse_action
from .skill_teachers import SkillTokenCodec

PAIR_MASK = "turn_hidden_pair_mask"
PREVIOUS = "turn_hidden_previous"


def validate_settings(config):
    cfg = config.turn_hidden_diversity
    if not math.isfinite(cfg.margin) or not 0 < cfg.margin <= 2:
        raise ValueError("turn_hidden_diversity.margin must be in (0, 2]")
    if not math.isfinite(cfg.coefficient) or cfg.coefficient < 0:
        raise ValueError("turn_hidden_diversity.coefficient must be finite and nonnegative")
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("Turn hidden diversity requires synchronous V1 training")
    actor, model = config.actor_rollout_ref.actor, config.actor_rollout_ref.model
    if actor.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("Turn hidden diversity requires FSDP/FSDP2")
    if model.use_remove_padding or model.get("use_fused_kernels", False):
        raise ValueError("Turn hidden diversity requires padded, unfused model forwards")
    if actor.fsdp_config.ulysses_sequence_parallel_size != 1:
        raise ValueError("Turn hidden diversity does not support sequence parallelism")
    if config.actor_rollout_ref.rollout.agent.default_agent_loop != "search_r1":
        raise ValueError("Turn hidden diversity requires the Search-R1 agent loop")
    if config.get("distillation", {}).get("enabled", False):
        raise ValueError("External distillation is not supported with turn hidden diversity")


def pair_positions(codec, responses, generated, *, return_stats=False):
    """Use the actual token containing the final byte of the opening search tag.

    BPE may merge whitespace/query bytes after '>'; count this explicitly rather
    than silently dropping otherwise valid searches. No retokenization is used.
    """
    ids, mask = responses.tolist(), generated.bool().tolist()
    current_mask = torch.zeros(len(ids), dtype=torch.bool)
    previous_positions = torch.full((len(ids),), -1, dtype=torch.long)
    stats = dict(valid_searches=0, endpoints=0, eligible_pairs=0, captured_pairs=0,
                 boundary_tokens_with_suffix=0, boundary_tokens_with_query=0)
    previous = None
    previous_valid = False
    begin = 0
    while begin < len(ids):
        if not mask[begin] or ids[begin] in codec.special_ids:
            begin += 1
            continue
        end = begin
        while end < len(ids) and mask[end] and ids[end] not in codec.special_ids:
            end += 1
        pieces = [codec.token_bytes(t) for t in ids[begin:end]]
        text = b"".join(pieces)
        valid = parse_action(text.decode("utf-8", errors="replace"))[0] == "search"
        endpoint = None
        if valid:
            stats["valid_searches"] += 1
            hidden_spans = list(re.finditer(rb"<think>.*?</think>", text, re.S))
            tags = [m for m in re.finditer(rb"<search>", text)
                    if not any(h.start() <= m.start() < h.end() for h in hidden_spans)]
            if len(tags) != 1:
                raise ValueError("Valid search has ambiguous opening tag")
            close = tags[0].end()
            offset = 0
            for i, piece in enumerate(pieces, begin):
                finish = offset + len(piece)
                if offset < close <= finish:
                    endpoint = i
                    stats["boundary_tokens_with_suffix"] += int(finish > close)
                    stats["boundary_tokens_with_query"] += int(bool(text[close:finish].strip()))
                    break
                offset = finish
            if endpoint is None:
                raise ValueError("Valid search has no generated token endpoint")
            stats["endpoints"] += 1
        if valid and previous_valid:
            stats["eligible_pairs"] += 1
        if endpoint is not None and previous is not None:
            current_mask[endpoint] = True
            previous_positions[endpoint] = previous
            stats["captured_pairs"] += 1
        previous, previous_valid = endpoint, valid
        begin = end
    result = (current_mask, previous_positions)
    return (*result, stats) if return_stats else result


def distance_sums(current, previous, margin):
    """FP32 normalized cosine, with a detached reference endpoint."""
    a = F.normalize(current.float(), dim=-1, eps=1e-8)
    b = F.normalize(previous.detach().float(), dim=-1, eps=1e-8)
    cosine = (a * b).sum(-1).clamp(-1, 1)
    distance = 1 - cosine
    return {
        "loss_sum": F.relu(margin - distance).sum(),
        "cosine_sum": cosine.sum(),
        "distance_sum": distance.sum(),
        "active_sum": (distance < margin).float().sum(),
    }


@contextmanager
def capture_decisions(module, data):
    """Capture only selected vectors; do not retain every layer's hidden states."""
    from verl.utils import tensordict_utils as tu

    if tu.get(data, "use_remove_padding", default=False) or tu.get(data, "use_fused_kernels", default=False):
        raise ValueError("Decision hidden capture requires padded unfused forwards")
    rows, current, previous = [], [], []
    for row, (prompt, pairs, prior) in enumerate(
        zip(data["prompts"].unbind(), data[PAIR_MASK].unbind(), data[PREVIOUS].unbind(), strict=True)
    ):
        selected = pairs.nonzero().flatten().tolist()
        for index in selected:
            prior_index = int(prior[index])
            if not 0 <= prior_index < index:
                raise ValueError("Invalid adjacent-turn endpoint indices")
            rows.append(row)
            current.append(len(prompt) + index)
            previous.append(len(prompt) + prior_index)
    result = {}

    def capture(_module, args):
        if result:
            raise RuntimeError("Expected one LM-head call per actor forward")
        hidden = args[0]
        if hidden.ndim != 3 or hidden.shape[0] != data.batch_size[0]:
            raise ValueError("Expected right-padded [batch, sequence, hidden] states")
        # Empty indexing yields a graph-connected zero when there are no pairs.
        r = torch.tensor(rows, dtype=torch.long, device=hidden.device)
        c = torch.tensor(current, dtype=torch.long, device=hidden.device)
        p = torch.tensor(previous, dtype=torch.long, device=hidden.device)
        result.update(distance_sums(hidden[r, c], hidden[r, p], tu.get(data, "turn_hidden_margin")))
        result["pairs"] = len(rows)

    head = module.get_output_embeddings()
    if head is None:
        raise ValueError("Model does not expose an LM head")
    handle = head.register_forward_pre_hook(capture)
    try:
        yield result
        if not result:
            raise RuntimeError("LM head bypassed; cannot obtain decision hidden states")
    finally:
        handle.remove()


def turn_ppo_loss(config, skill_weights, model_output, data, dp_group=None):
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.losses import ppo_loss

    if skill_weights is None:
        loss, metrics = ppo_loss(config, model_output, data, dp_group)
    else:
        from .skill_distillation import skill_ppo_loss

        loss, metrics = skill_ppo_loss(config, skill_weights, model_output, data, dp_group)
    sums = model_output["turn_hidden_diversity"]
    count = tu.get(data, "auxiliary_loss_mask_counts")[PAIR_MASK]
    # FSDP averages gradients across DP. Engine sums micro-batch losses.
    scale = data["dp_size"] / max(1, count)
    coefficient = tu.get(data, "turn_hidden_coefficient")
    auxiliary = sums["loss_sum"] * scale
    loss = loss + coefficient * auxiliary
    for key, value in [
        ("loss", auxiliary),
        ("weighted_loss", coefficient * auxiliary),
        ("cosine_mean", sums["cosine_sum"] * scale),
        ("distance_mean", sums["distance_sum"] * scale),
        ("active_fraction", sums["active_sum"] * scale),
        ("pairs", loss.new_tensor(sums["pairs"] * data["dp_size"])),
    ]:
        metrics[f"turn_hidden/{key}"] = Metric(value=value.detach(), aggregation=AggregationType.SUM)
    metrics["turn_hidden/margin"] = tu.get(data, "turn_hidden_margin")
    metrics["turn_hidden/coefficient"] = coefficient
    return loss, metrics


class TurnHiddenSupervisor:
    def __init__(self, config):
        validate_settings(config)
        self.config = config.turn_hidden_diversity

    def initialize(self, trainer):
        from verl.utils.config import omega_conf_to_dataclass

        self.codec = SkillTokenCodec(trainer.tokenizer)
        skills = trainer.skill_teacher_supervisor
        weights = skills.weights if skills is not None else None
        trainer.actor_rollout_wg.set_loss_fn(
            partial(
                turn_ppo_loss,
                config=omega_conf_to_dataclass(trainer.config.actor_rollout_ref.actor),
                skill_weights=weights,
            )
        )

    def prepare(self, batch, metrics):
        import transfer_queue as tq
        from tensordict import TensorDict

        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["responses", "response_mask"]
        )
        masks, previous = [], []
        coverage = {}
        for i, (response, generated) in enumerate(
            zip(data["responses"].unbind(), data["response_mask"].unbind(), strict=True)
        ):
            mask, prior, stats = pair_positions(self.codec, response, generated, return_stats=True)
            if batch.tags[i].get("is_padding", False):
                mask.zero_()
                prior.fill_(-1)
            if not batch.tags[i].get("is_padding", False):
                for key, value in stats.items():
                    coverage[key] = coverage.get(key, 0) + value
            masks.append(mask)
            previous.append(prior)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {
                    PAIR_MASK: torch.nested.as_nested_tensor(masks, layout=torch.jagged),
                    PREVIOUS: torch.nested.as_nested_tensor(previous, layout=torch.jagged),
                },
                batch_size=[len(batch)],
            ),
        )
        batch.fields = None
        metrics["turn_hidden/prepared_pairs"] = sum(int(m.sum()) for m in masks)
        metrics.update({f"turn_hidden/coverage/{k}": v for k, v in coverage.items()})
        for label, numerator, denominator in [("endpoint_fraction", "endpoints", "valid_searches"),
                                               ("pair_fraction", "captured_pairs", "eligible_pairs")]:
            metrics[f"turn_hidden/coverage/{label}"] = coverage.get(numerator, 0) / max(1, coverage.get(denominator, 0))
        return batch

    def update_metadata(self, batch):
        names = list(batch.extra_info.get("auxiliary_loss_mask_names", []))
        batch.extra_info.update(
            turn_hidden_enabled=True,
            turn_hidden_gradient_diagnostics=bool(getattr(self.config, "gradient_diagnostics", False)),
            turn_hidden_margin=float(self.config.margin),
            turn_hidden_coefficient=float(self.config.coefficient),
            auxiliary_loss_mask_names=list(dict.fromkeys(names + [PAIR_MASK])),
        )

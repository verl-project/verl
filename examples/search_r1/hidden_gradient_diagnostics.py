"""Opt-in full-mini-batch, post-FSDP-sync parameter-gradient diagnostics.

Two extra backward passes, no optimizer step. CPU buffers hold local shards.
The normal training backward follows after restoring RNG and clearing gradients.
"""
import math
import time
from functools import partial
import torch


def component_loss(component, config, model_output, data, dp_group=None):
    from verl.utils import tensordict_utils as tu
    if component == 'grpo':
        from verl.trainer.ppo.core_algos import get_policy_loss_fn
        from verl.workers.utils.padding import no_padding_2_padding
        log_prob = no_padding_2_padding(model_output['log_probs'], data)
        padded = data.select('old_log_probs', 'advantages', 'response_mask',
                             *(['rollout_is_weights'] if 'rollout_is_weights' in data else [])).to_padded_tensor()
        config.global_batch_info.update(dp_size=data['dp_size'], batch_num_tokens=data['batch_num_tokens'],
                                        global_batch_size=data['global_batch_size'], loss_scale_factor=config.loss_scale_factor)
        loss, _ = get_policy_loss_fn(config.policy_loss.get('loss_mode', 'vanilla'))(
            old_log_prob=padded['old_log_probs'], log_prob=log_prob, advantages=padded['advantages'],
            response_mask=padded['response_mask'].bool(), loss_agg_mode=config.loss_agg_mode,
            config=config, rollout_is_weights=padded.get('rollout_is_weights', None))
        return loss, {}
    count = tu.get(data, 'auxiliary_loss_mask_counts')['turn_hidden_pair_mask']
    return (model_output['turn_hidden_diversity']['loss_sum'] * data['dp_size']
            / max(1, count) * tu.get(data, 'turn_hidden_coefficient')), {}


def gradient_geometry(first, second):
    """Stable dot products of matching local parameter shards; absent grads=0."""
    aa = bb = ab = 0.0
    for a, b in zip(first, second, strict=True):
        if a is None and b is None:
            continue
        if a is not None:
            aa += torch.dot(a.flatten().double(), a.flatten().double()).item()
        if b is not None:
            bb += torch.dot(b.flatten().double(), b.flatten().double()).item()
        if a is not None and b is not None:
            ab += torch.dot(a.flatten().double(), b.flatten().double()).item()
    return aa, bb, ab


def geometry_metrics(aa, bb, ab):
    a, b = math.sqrt(max(0., aa)), math.sqrt(max(0., bb))
    valid = a > 0 and b > 0
    cosine = max(-1., min(1., ab/(a*b))) if valid else 0.
    return {'grpo_norm': a, 'weighted_hidden_norm': b,
            'hidden_to_grpo_ratio': b/a if a else 0.,
            'ratio_valid': float(a > 0), 'angle_valid': float(valid),
            'cosine': cosine, 'angle_degrees': math.degrees(math.acos(cosine)) if valid else 0.}


def measure_gradients(engine, data, loss_function):
    from verl.utils import tensordict_utils as tu
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
    if getattr(engine, 'scaler', None) is not None:
        raise ValueError('Gradient diagnostics requires unscaled BF16/FP32 backward')
    if isinstance(engine.module, FullyShardedDataParallel):
        if engine.module.sharding_strategy != ShardingStrategy.FULL_SHARD:
            raise ValueError('Gradient diagnostics requires FULL_SHARD (no replicated gradient counting)')
    # The registered turn loss partial carries the actor configuration.
    config = loss_function.keywords['config']
    started = time.monotonic()
    engine._turn_gradient_probe = True
    buffers = []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            rng_cpu = torch.get_rng_state()
            rng_cuda = torch.cuda.get_rng_state()
            for component in ['grpo', 'hidden']:
                torch.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state(rng_cuda)
                engine.optimizer_zero_grad()
                engine.forward_backward_batch(data, partial(component_loss, component, config))
                shards = []
                for parameter in engine.module.parameters():
                    grad = parameter.grad
                    if grad is not None and hasattr(grad, 'to_local'):
                        grad = grad.to_local()
                    shards.append(None if grad is None else grad.detach().float().cpu().clone())
                buffers.append(shards)
        values = torch.tensor(gradient_geometry(*buffers), device=torch.cuda.current_device(), dtype=torch.float64)
        torch.distributed.all_reduce(values, group=engine.get_data_parallel_group())
        result = geometry_metrics(*values.tolist())
        result['seconds'] = time.monotonic()-started
        return {f'turn_hidden/gradient/{key}': value for key, value in result.items()}
    finally:
        buffers.clear()
        engine.optimizer_zero_grad()
        engine._turn_gradient_probe = False

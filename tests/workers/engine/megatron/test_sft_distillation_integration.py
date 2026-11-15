# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real CP2/DP2 SFT teacher export, multi-sample distillation and CE validation.

Run on four GPUs with the paired Megatron-Core/Transformer Engine stack:

    torchrun --nproc-per-node=4 --master-port=29847 \
        tests/workers/engine/megatron/test_sft_distillation_integration.py

Uses the actual GPT model, engine, packing, DDP, schedule and gradient finalizer.
A CP1 full-vocabulary reference provides independent top-k distributions, losses
and parameter gradients. Each DP replica sends two unequal even-length samples in ONE
microbatch, so a lookup that hashes the whole packed batch cannot pass. No model
checkpoint, tokenizer download, Ray actor or mocked collective is used.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.workers.engine.megatron.test_cp_dp_gradient_normalization import (
    _LENGTHS,
    _initialize_parallel,
    _make_model,
    _samples,
)

_COUNTS = (2, 7, 1, 4)
_TOPK = 6
_DUMP_TOPK = 64


def _key(ids):
    return hashlib.sha1(str(ids.tolist()).encode("utf8")).hexdigest()


def _reference():
    config, raw_model, model = _make_model(cp_size=1, per_token=False)
    state = {
        name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in raw_model.state_dict().items()
    }
    lookup, distributions, per_position = {}, [], []
    kl_loss = torch.zeros((), device="cuda", dtype=torch.float32)
    ce_loss = torch.zeros_like(kl_loss)
    for ids, mask in _samples(_COUNTS, "cuda"):
        logits = model(
            input_ids=ids.unsqueeze(0),
            position_ids=torch.arange(ids.numel(), device="cuda").unsqueeze(0),
            attention_mask=None,
        )
        student = logits[0].float().log_softmax(-1)
        # A different teacher keeps the parameter gradients nonzero. The teacher
        # remains fixed during the distributed run and uses the full vocabulary.
        teacher = logits[0].detach().float().roll(5, dims=-1).log_softmax(-1).topk(_TOPK, dim=-1)
        lookup[_key(ids)] = (teacher.indices.cpu(), teacher.values.cpu())
        token_kl = (teacher.values.exp() * (teacher.values - student.gather(-1, teacher.indices))).sum(-1)
        kl_loss = kl_loss + (token_kl[:-1] * mask[1:]).sum() / sum(_COUNTS)
        ce_loss = ce_loss + (F.cross_entropy(logits[0, :-1].float(), ids[1:], reduction="none") * mask[1:]).sum()
        distributions.append(student.detach().cpu())
        per_position.append(token_kl.detach().cpu())
    ce_value = float(ce_loss.detach()) / sum(_COUNTS)
    kl_value = float(kl_loss.detach())
    kl_loss.backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone() for name, parameter in raw_model.named_parameters()
    }
    del model, raw_model, config, logits, student, kl_loss, ce_loss
    gc.collect()
    torch.cuda.empty_cache()
    return state, lookup, distributions, per_position, gradients, kl_value, ce_value


def _engine(state, per_token=False):
    from megatron.core.distributed import DistributedDataParallel, finalize_model_grads
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    config, raw_model, model = _make_model(cp_size=2, per_token=per_token)
    raw_model.load_state_dict(state)
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            average_in_collective=False,
        ),
        module=model,
    )
    config.finalize_model_grads_func = finalize_model_grads
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.module = [ddp]
    engine.tf_config = config
    engine.engine_config = SimpleNamespace(
        cp_seq_chunks=1,
        context_parallel_size=2,
        dynamic_context_parallel=False,
        use_remove_padding=True,
        use_fused_kernels=False,
        pad_bshd_to_minibatch_max=False,
        pad_to_length=False,
        pad_to_length_bucket=None,
        entropy_from_logits_with_chunking=False,
    )
    engine.model_config = SimpleNamespace(
        mtp=SimpleNamespace(enable=False, enable_train=False),
        hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"]),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    engine.enable_routing_replay = False
    return engine, config, raw_model, ddp


def _batch(*, validation=False):
    from megatron.core import parallel_state as mpu
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    first = mpu.get_data_parallel_rank() * 2
    rows = _samples(_COUNTS, "cuda")[first : first + 2]
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor([ids for ids, _ in rows], layout=torch.jagged),
            "loss_mask": torch.nested.nested_tensor([mask for _, mask in rows], layout=torch.jagged),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        gradient_accumulation_steps=1,
        pad_mode=DatasetPadMode.NO_PADDING,
        temperature=1.0,
        needs_vocab_logits=not validation,
        sft_distillation_validation=validation,
    )
    return data


def _global_loss(output):
    value = torch.tensor(sum(output["loss"]), device="cuda", dtype=torch.float64)
    dist.all_reduce(value)
    return float(value / dist.get_world_size())


def _check_all_ranks(check):
    """Report numerical failures together before the next engine collective."""
    result, error = None, None
    try:
        result = check()
    except Exception:
        error = traceback.format_exc()
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise AssertionError("\n".join(f"rank {rank}: {value}" for rank, value in enumerate(errors) if value))
    return result


def _run(output_path=None):
    from megatron.core import parallel_state as mpu

    from verl.workers.utils.distill_sft_loss import SFTDistillationLoss, missing_payload_count
    from verl.workers.utils.teacher_dump import DUMP_IDS_KEY, DUMP_LOGPROBS_KEY, TeacherTopKDump, dump_consistency_stats

    assert int(os.environ.get("WORLD_SIZE", "1")) == 4, "Run with four torchrun ranks"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=4))
    results = []
    failed = False
    try:
        _initialize_parallel(cp_size=1)
        state, lookup, distributions, reference_positions, expected_gradients, kl_loss, ce_loss = _reference()
        mpu.destroy_model_parallel()
        dist.barrier()
        _initialize_parallel(cp_size=2)
        engine, config, raw_model, ddp = _engine(state)
        first = mpu.get_data_parallel_rank() * 2
        ddp.zero_grad_buffer()
        raw_model.eval()
        data = _batch()
        # Export the whole tiny vocabulary so the consistency check covers every
        # next-token target, rather than depending on random top-k overlap.
        exported = engine.infer_batch(data, loss_function=TeacherTopKDump(_DUMP_TOPK))
        print(f"rank={dist.get_rank()} teacher dump forward completed", flush=True)

        def check_dump():
            dump = exported["model_output"]
            assert set((DUMP_IDS_KEY, DUMP_LOGPROBS_KEY, "log_probs")) <= dump.keys()
            dump_max_error = 0.0
            for index, (ids, logps) in enumerate(
                zip(dump[DUMP_IDS_KEY].unbind(), dump[DUMP_LOGPROBS_KEY].unbind(), strict=True)
            ):
                reference = distributions[first + index].to(logps.device)
                selected = reference.gather(-1, ids)
                dump_max_error = max(dump_max_error, float((logps - selected).abs().max()))
                torch.testing.assert_close(logps, selected, rtol=0.0, atol=0.01)
                torch.testing.assert_close(logps, reference.topk(_DUMP_TOPK, dim=-1).values, rtol=0.0, atol=0.01)
            stats = dump_consistency_stats(dump, data["input_ids"])
            assert stats["coverage"] > 0
            assert stats["max_abs_diff"] < 1e-5, stats
            assert all(not torch.count_nonzero(parameter.main_grad) for parameter in raw_model.parameters())
            return {
                "case": "teacher_dump",
                "sequence_lengths": list(_LENGTHS),
                "max_logprob_error_to_cp1": dump_max_error,
                **stats,
            }

        results.append(_check_all_ranks(check_dump))

        distillation = SFTDistillationLoss(
            SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=-10.0)), teacher_lookup=lookup
        )
        missing_before = missing_payload_count()
        raw_model.train()
        for per_token in (False, True):
            if per_token:
                # DDP chooses its buffer scaling factors at construction time;
                # changing only the model config would leave those factors stale.
                del engine, config, raw_model, ddp
                gc.collect()
                torch.cuda.empty_cache()
                engine, config, raw_model, ddp = _engine(state, per_token=True)
            ddp.zero_grad_buffer()
            trained = engine.forward_backward_batch(_batch(), loss_function=distillation)
            actual_loss = _global_loss(trained)

            def check_training(trained=trained, actual_loss=actual_loss, per_token=per_token, raw_model=raw_model):
                assert len(trained["loss"]) == 1, "Both samples must share a single microbatch"
                assert missing_payload_count() == missing_before
                for index, actual in enumerate(trained["model_output"]["distillation_losses"].unbind()):
                    torch.testing.assert_close(actual.cpu(), reference_positions[first + index], rtol=0.03, atol=0.001)
                assert abs(actual_loss - kl_loss) < 0.001, (per_token, actual_loss, kl_loss)
                gradients = {
                    name: parameter.main_grad.detach().float().cpu().clone()
                    for name, parameter in raw_model.named_parameters()
                }
                assert all(torch.isfinite(value).all() for value in gradients.values())
                squared_reference = sum(value.square().sum() for value in expected_gradients.values())
                assert squared_reference > 0
                relative_l2 = float(
                    (
                        sum((gradients[name] - expected_gradients[name]).square().sum() for name in gradients)
                        / squared_reference
                    ).sqrt()
                )
                assert relative_l2 < 0.03, (per_token, relative_l2)
                return relative_l2

            relative_l2 = _check_all_ranks(check_training)
            before_validation = {
                name: parameter.main_grad.detach().clone() for name, parameter in raw_model.named_parameters()
            }
            raw_model.eval()
            validation = engine.infer_batch(_batch(validation=True), loss_function=distillation)
            actual_ce_loss = _global_loss(validation)

            def check_validation(
                actual_ce_loss=actual_ce_loss, before_validation=before_validation, raw_model=raw_model
            ):
                assert abs(actual_ce_loss - ce_loss) < 0.005, (actual_ce_loss, ce_loss)
                for name, parameter in raw_model.named_parameters():
                    assert torch.equal(parameter.main_grad, before_validation[name]), name

            _check_all_ranks(check_validation)
            raw_model.train()
            results.append(
                {
                    "case": "sft_distillation_and_ce_validation",
                    "calculate_per_token_loss": per_token,
                    "cp_size": 2,
                    "dp_size": 2,
                    "samples_per_microbatch": 2,
                    "sequence_lengths": list(_LENGTHS),
                    "missing_teacher_payloads": missing_payload_count() - missing_before,
                    "loss": actual_loss,
                    "reference_loss": kl_loss,
                    "relative_gradient_l2": relative_l2,
                    "validation_ce": actual_ce_loss,
                    "reference_validation_ce": ce_loss,
                }
            )
            dist.barrier()
        if dist.get_rank() == 0:
            print(json.dumps(results, indent=2), flush=True)
            if output_path is not None:
                Path(output_path).write_text(json.dumps(results, indent=2) + "\n")
    except BaseException:
        failed = True
        traceback.print_exc()
        raise
    finally:
        # A failed assertion can occur on only one DP replica. Let torchrun
        # terminate the remaining workers instead of blocking in group teardown.
        if not failed:
            if mpu.model_parallel_is_initialized():
                mpu.destroy_model_parallel()
            dist.destroy_process_group()


def test_sft_distillation_integration():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available():
        pytest.skip("Requires four GPUs and torchrun; see the module docstring")
    _run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    _run(parser.parse_args().output)

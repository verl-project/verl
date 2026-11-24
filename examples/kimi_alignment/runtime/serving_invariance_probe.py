"""Fixed-policy serving controls using the actual scheduler and model entry."""
import asyncio
import copy
import json
import os
import time
from pathlib import Path


async def replay_prefix(agent, request_id, prompt_ids, output_ids, multimodal):
    if os.environ.get("KIMI_ALIGNMENT_SERVING_PROBE") != "1" or len(output_ids) < 5:
        return
    directory = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "serving_invariance"
    directory.mkdir(exist_ok=True)
    try:
        with (directory / "source.json").open("x") as handle:
            json.dump({"request_id": request_id, "prompt_ids": list(prompt_ids),
                       "suffix":list(output_ids[:4]),"target":int(output_ids[4])}, handle)
    except FileExistsError:
        return
    await agent.server_manager.generate(
        request_id="alignment-probe-control", prompt_ids=list(prompt_ids),
        sampling_params={"temperature": 0., "max_tokens": 1}, **multimodal)


async def run_matrix(server, prompt):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    engine = server.engine
    directory = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "serving_invariance"
    await engine.wait_for_requests_to_drain()
    version = server.global_steps
    reports = []

    async def group(label, items, length, budget=None):
        await engine.pause_generation(mode="keep", clear_cache=False)
        previous_budget = None
        if budget is not None:
            previous_budget = await engine.engine_core.call_utility_async("alignment_scheduler_budget", budget)
        collectors = []
        ids = []
        try:
            for role, value in items:
                name = f"alignment-probe-{label}-{role}"
                params = SamplingParams(temperature=0., max_tokens=length,
                                        ignore_eos=True, logprobs=0,
                                        output_kind=RequestOutputKind.FINAL_ONLY)
                collector = await engine.add_request(name, value, params)
                collectors.append(collector)
                ids.append(name)
        finally:
            await engine.resume_generation()
        outputs = await asyncio.wait_for(asyncio.gather(*(c.get() for c in collectors)), 120)
        if previous_budget is not None:
            await engine.pause_generation(mode="keep", clear_cache=False)
            try:
                await engine.engine_core.call_utility_async("alignment_scheduler_budget", previous_budget)
            finally:
                await engine.resume_generation()
        for name, output in zip(ids, outputs):
            assert output.finished and len(output.outputs) == 1
            sample = output.outputs[0]
            reports.append({"id": name, "token_ids":list(sample.token_ids),
                            "logprobs":[float(scores[tok].logprob) for tok,scores in zip(sample.token_ids,sample.logprobs)],
                            "weight_version":version})
        (directory / "matrix-output.json").write_text(json.dumps(reports, indent=2))
        return outputs[0]

    baseline = await group("single", [("target",prompt)], 64)
    await group("repeat", [("target",prompt)], 64)
    peer = copy.deepcopy(prompt)
    # Text-only suffix keeps the original multimodal placeholder geometry intact.
    peer["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + list(baseline.outputs[0].token_ids[:2])
    await group("mixed", [("target",prompt),("peer",peer)], 64)
    shorter_peer = copy.deepcopy(prompt)
    shorter_peer["prompt_token_ids"] = list(prompt["prompt_token_ids"])[:-2]
    await group("reordered", [("peer",shorter_peer),("target",prompt)], 64)
    neighbors = []
    for index in range(15):
        value = copy.deepcopy(prompt)
        value["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + list(baseline.outputs[0].token_ids[:index % 4 + 1])
        neighbors.append((f"peer{index}",value))
    await group("batch16", [("target",prompt), *neighbors], 64)
    full = copy.deepcopy(prompt)
    full["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + list(baseline.outputs[0].token_ids[:4])
    await group("full-prefix", [("target",full)], 1)
    extended = copy.deepcopy(prompt)
    extended["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + list(baseline.outputs[0].token_ids[:8]) * 80
    await group("long-full", [("target",extended)], 1)
    await group("long-chunked", [("target",extended)], 1, budget=512)
    assert server.global_steps == version
    (directory / "matrix-complete.json").write_text(json.dumps({"version":version,"completed":True}))


def capture_probe_logits(logits, batch, rank):
    if rank != 0:
        return
    import torch
    directory = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "serving_invariance"
    for row, request in enumerate(batch.req_ids):
        if not request.startswith("alignment-probe-"):
            continue
        start,end = map(int,batch.query_start_loc[row:row+2].tolist())
        positions = batch.positions.flatten()[start:end]
        record = {"request":request,"positions":positions.cpu(),"batch_ids":list(batch.req_ids),
                  "ids":batch.input_ids.flatten()[start:end].cpu(),
                  "logits":logits[row].detach().cpu(),"total_queries":batch.input_ids.numel()}
        torch.save(record,directory / (request + f"-pos{int(positions[-1])}-{time.time_ns()}.pt"))

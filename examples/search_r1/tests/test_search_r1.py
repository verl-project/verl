import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import ClientResponseError, web
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from examples.search_r1.agent_loop import SearchR1AgentLoop
from examples.search_r1.prepare_data import adapt_row
from examples.search_r1.protocol import SYSTEM_PROMPT, compute_score, parse_action, score_final_turn
from examples.search_r1.search_tool import SearchTool
from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap


@pytest.fixture(scope="module")
def tokenizer():
    default = Path(__file__).resolve().parents[4] / "../sd-search/models/Qwen2.5-3B"
    return AutoTokenizer.from_pretrained(os.environ.get("SEARCH_R1_TOKENIZER", str(default)), local_files_only=True)


class FakeServer:
    def __init__(self, tokenizer, turns):
        self.tokenizer = tokenizer
        self.turns = iter(turns)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        ids = self.tokenizer.encode(next(self.turns), add_special_tokens=False)
        # Model server receives the actual per-turn/context cap.
        ids = ids[: kwargs["sampling_params"]["max_tokens"]]
        version = len(self.calls) - 1
        return SimpleNamespace(
            token_ids=ids,
            log_probs=[-0.5] * len(ids),
            extra_fields={"min_global_steps": version, "max_global_steps": version},
        )


class FakeTool(SearchTool):
    def __init__(self):
        super().__init__({"retrieval_service_url": "http://127.0.0.1:1/retrieve"})
        self.queries = []

    async def execute(self, instance_id, parameters, **kwargs):
        from verl.tools.schemas import ToolResponse

        assert instance_id in self._states
        self.queries.append(parameters)
        return ToolResponse(text="Evidence <answer>Wrong</answer>: Paris is in France."), 0.0, {}


def make_loop(tokenizer, server, tool, response_length=4096, **kwargs):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 1024,
                    "response_length": response_length,
                    "max_model_len": 5120,
                }
            }
        }
    )
    return SearchR1AgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=None,
        data_config=DictConfigWrap(OmegaConf.create({})),
        hf_model_type="qwen2",
        tools=ToolListWrap([tool]),
        **kwargs,
    )


def run_inputs():
    return {
        "raw_prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Capital of France?"}],
        "reward_model": {"ground_truth": {"target": ["Paris"]}},
    }


def test_identical_trajectories_share_uncapped_initial_tokens(tokenizer, monkeypatch):
    from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

    monkeypatch.delattr(tokenizer, "_search_r1_runtime_cache", raising=False)

    def forbid_vocab_copy():
        raise AssertionError("Rollout must not copy the full vocabulary")

    monkeypatch.setattr(tokenizer, "get_vocab", forbid_vocab_copy)
    original = QwenContinuousTokenBuilder.build_initial_tokens
    calls = []

    def record(self, messages, **kwargs):
        calls.append(copy.deepcopy(messages))
        return original(self, messages, **kwargs)

    monkeypatch.setattr(QwenContinuousTokenBuilder, "build_initial_tokens", record)

    async def check():
        tool = FakeTool()
        loops = [make_loop(tokenizer, FakeServer(tokenizer, ["<answer>Paris</answer>"]), tool) for _ in range(5)]
        outputs = await asyncio.gather(*[loop.run({"logprobs": True}, **run_inputs()) for loop in loops])
        canonical = tokenizer.apply_chat_template(run_inputs()["raw_prompt"], tokenize=True, add_generation_prompt=True)
        assert len(calls) == 1
        assert all(output.prompt_ids == canonical and output.reward_score == 1 for output in outputs)
        outputs[0].prompt_ids.append(0)
        assert outputs[1].prompt_ids == canonical
        assert not tool._states
        oversized = run_inputs()
        oversized["raw_prompt"][-1]["content"] = "long question " * 2000
        server = FakeServer(tokenizer, ["<answer>Paris</answer>"])
        with pytest.raises(ValueError, match="Initial Search-R1 prompt exceeds"):
            await make_loop(tokenizer, server, tool).run({}, **oversized)
        assert not server.calls and not tool._states

    asyncio.run(check())


def test_multiturn_context_masks_logprobs_and_final_reward(tokenizer, tmp_path):
    async def check():
        turns = [
            "<think>Find the country.</think><search>France</search>",
            "<think>Find its capital.</think><search>capital of France</search>",
            "<think>Done.</think><answer>Paris</answer>",
        ]
        server, tool = FakeServer(tokenizer, turns), FakeTool()
        loop = make_loop(tokenizer, server, tool)
        inputs = run_inputs()
        original = copy.deepcopy(inputs)
        output = await loop.run({"logprobs": True, "temperature": 1.0}, **inputs)
        assert inputs == original
        messages = output.extra_fields["search_r1_messages"]
        assert [m["role"] for m in messages] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ]
        for i, call in enumerate(server.calls):
            history = messages[: 2 + 2 * i]
            canonical = tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True)
            assert call["prompt_ids"] == canonical
        assert output.reward_score == 1.0
        assert output.extra_fields["min_global_steps"] == 0
        assert output.extra_fields["max_global_steps"] == 2
        assert output.extra_fields["reward_extra_info"]["search_calls"] == 2
        assert len(output.response_ids) == len(output.response_mask) == len(output.response_logprobs)
        # Only exact model output IDs are labelled as sampled, including every search query.
        sampled = [t for t, m in zip(output.response_ids, output.response_mask, strict=False) if m]
        assert sampled == sum((tokenizer.encode(t, add_special_tokens=False) for t in turns), [])
        assert all(
            p == (-0.5 if m else 0.0) for p, m in zip(output.response_logprobs, output.response_mask, strict=False)
        )
        assert not tool._states
        assert tool.queries == [{"query": "France"}, {"query": "capital of France"}]
        (tmp_path / "simulated-trajectory.json").write_text(
            json.dumps(
                {
                    "simulation": True,
                    "output": output.model_dump(),
                    "generation_requests": server.calls,
                },
                indent=2,
            )
        )

    asyncio.run(check())


@pytest.mark.parametrize("case", ["action_budget", "context_budget", "invalid_action", "wrong_answer", "eos"])
def test_termination_and_zero_reward(tokenizer, case):
    async def check():
        turns = ["<search>France</search>", "<answer>Paris</answer>"]
        kwargs, budget = {}, 4096
        if case == "action_budget":
            kwargs["max_actions"] = 1
        elif case == "context_budget":
            budget = len(tokenizer.encode(turns[0], add_special_tokens=False)) + 1
        elif case == "invalid_action":
            turns = ["<think>unfinished <answer>Paris</answer>"]
            kwargs["max_actions"] = 1
        elif case == "eos":
            turns = ["I do not know.<|im_end|>"]
        else:
            turns = ["<answer>Lyon</answer>"]
        server, tool = FakeServer(tokenizer, turns), FakeTool()
        output = await make_loop(tokenizer, server, tool, response_length=budget, **kwargs).run(
            {"logprobs": True},
            **run_inputs(),
        )
        assert output.reward_score == 0.0
        assert len(output.response_ids) <= budget
        assert len(output.response_ids) == len(output.response_mask)
        assert not tool._states
        if case in {"action_budget", "context_budget"}:
            assert output.extra_fields["search_r1_termination"] == case

    asyncio.run(check())


def test_exception_releases_tool_state(tokenizer):
    class BrokenTool(FakeTool):
        async def execute(self, *args, **kwargs):
            raise TimeoutError("retriever timed out")

    async def check():
        tool = BrokenTool()
        loop = make_loop(tokenizer, FakeServer(tokenizer, ["<search>France</search>"]), tool)
        with pytest.raises(TimeoutError):
            await loop.run({"logprobs": True}, **run_inputs())
        assert not tool._states

    asyncio.run(check())


@pytest.mark.parametrize("mode", ["success", "http_error", "bad_payload"])
def test_tool_http_contract_and_cleanup(mode):
    async def check():
        received = []

        async def retrieve(request):
            received.append(await request.json())
            if mode == "http_error":
                return web.Response(status=503)
            if mode == "bad_payload":
                return web.json_response({"result": []})
            return web.json_response(
                {"result": [[{"document": {"contents": "Paris"}, "score": 0.9}]], "profile": {"total_s": 0.01}}
            )

        app = web.Application()
        app.router.add_post("/retrieve", retrieve)
        runner = web.AppRunner(app)
        await runner.setup()
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        site = web.SockSite(runner, sock)
        await site.start()
        tool = SearchTool({"retrieval_service_url": f"http://127.0.0.1:{port}/retrieve"})
        instance, _ = await tool.create()
        try:
            if mode != "success":
                expected = ClientResponseError if mode == "http_error" else ValueError
                with pytest.raises(expected):
                    await tool.execute(instance, {"query": "France"})
                return
            response, reward, metrics = await tool.execute(instance, {"query": " France "})
            assert response.text == "Doc 1(Title: Paris)"
            assert reward == 0.0 and metrics["retrieved_documents"] == 1
            assert received == [{"queries": ["France"], "topk": 3, "return_scores": True}]
            # No duplicate E5 query prefix is introduced by the HTTP client.
            with pytest.raises(ValueError):
                await tool.execute(instance, {"query": " "})
        finally:
            await tool.release(instance)
            await tool.close()
            await runner.cleanup()
        assert not tool._states

    asyncio.run(check())


def test_observation_truncation_and_no_logprobs(tokenizer):
    async def check():
        tool = FakeTool()
        server = FakeServer(tokenizer, ["<search>France</search>", "<answer>Paris</answer>"])
        output = await make_loop(tokenizer, server, tool, max_observation_tokens=2).run({}, **run_inputs())
        assert output.response_logprobs is None
        tool_message = output.extra_fields["search_r1_messages"][3]
        body = tool_message["content"].removeprefix("<information>").removesuffix("</information>")
        assert len(tokenizer.encode(body, add_special_tokens=False)) <= 2
        assert output.reward_score == 1.0

    asyncio.run(check())


def test_reward_provenance_and_protocol():
    assert parse_action("<think><search>fake</search></think><answer>Paris</answer>") == ("answer", "Paris")
    assert parse_action("<answer>A</answer><answer>B</answer>")[0] == "invalid"
    assert score_final_turn("<search>Paris</search>", {"target": ["Paris"]}, "action_budget")["score"] == 0
    assert (
        compute_score(
            "nq",
            "<answer>Paris</answer>",
            {"target": ["Paris"]},
            {
                "search_r1_final_assistant": "<answer>Lyon</answer>",
                "search_r1_termination": "answer",
            },
        )["score"]
        == 0
    )
    with pytest.raises(ValueError, match="provenance"):
        compute_score("nq", "<answer>Paris</answer>", {"target": ["Paris"]})


def test_data_adapter():
    row = {
        "prompt": [{"role": "user", "content": "old rules. Question: Capital of France?"}],
        "data_source": "nq",
        "reward_model": {"ground_truth": {"target": ["Paris"]}},
    }
    adapted = adapt_row(row, 0)
    assert adapted["prompt"][1] == {"role": "user", "content": "Capital of France?"}
    assert adapted["agent_name"] == "search_r1"
    assert "Paris" not in str(adapted["prompt"])
    row["prompt"][0]["content"] = "old rules. Question: Question: Capital of France?"
    assert adapt_row(row, 0)["prompt"][1]["content"] == "Question: Capital of France?"
    row["extra_info"] = {"index": "nq:train:123", "split": "train"}
    preserved = adapt_row(row, 0, preserve_prompt=True)
    assert preserved["prompt"] == row["prompt"]
    assert preserved["extra_info"]["index"] == "nq:train:123"
    assert preserved["extra_info"]["reward_metric"] == "em"
    with pytest.raises(ValueError, match="deferred"):
        adapt_row(row, 0, preserve_prompt=True, reward_metric="f1")


def test_sd_observation_budget_and_deferred_reward(tokenizer):
    async def check():
        server, tool = FakeServer(tokenizer, ["<search>France</search>", "<answer>Paris</answer>"]), FakeTool()
        loop = make_loop(tokenizer, server, tool, observation_tag="documents", max_observation_tokens=12)
        output = await loop.run({}, **run_inputs())
        evidence = output.extra_fields["search_r1_messages"][3]["content"]
        assert evidence.startswith("\n\n<documents>")
        assert len(tokenizer.encode(evidence, add_special_tokens=False)) <= 12
        assert output.reward_score == 1.0
        with pytest.raises(ValueError, match="deferred"):
            await loop.run({}, extra_info={"reward_metric": "f1"}, **run_inputs())

    asyncio.run(check())


def test_data_alignment_audit_preserves_duplicates_and_detects_order(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.search_r1.audit_data import compare

    official = [
        {"data_source": "nq", "question": "Capital?", "golden_answers": ["Paris"]},
        {"data_source": "nq", "question": "Capital?", "golden_answers": ["Paris"]},
        {"data_source": "nq", "question": "Country?", "golden_answers": ["France"]},
    ]
    candidate = [dict(row, question=row["question"].removesuffix("?")) for row in reversed(official)]
    source, target = tmp_path / "official.parquet", tmp_path / "candidate.parquet"
    pq.write_table(pa.Table.from_pylist(official), source)
    pq.write_table(pa.Table.from_pylist(candidate), target)
    result = compare([source], [target])
    assert not result["exact_question"]["question_and_answer_multisets_equal"]
    assert result["question_suffix_normalized"]["question_and_answer_multisets_equal"]
    assert not result["question_suffix_normalized"]["order_equal"]
    pq.write_table(pa.Table.from_pylist(candidate[:2]), target)
    assert compare([source], [target])["question_suffix_normalized"]["missing_rows"] == 1

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from examples.search_r1.profiling import PhaseRecorder


def test_phase_boundaries_keep_original_timer_and_exception(tmp_path):
    calls = []

    @contextmanager
    def original(name, result, **kwargs):
        calls.append((name, "begin"))
        try:
            yield
        finally:
            calls.append((name, "end"))

    recorder = PhaseRecorder(tmp_path / "phases.jsonl")
    timer = recorder.wrap(original)
    with pytest.raises(ValueError, match="original error"), timer("step", {}):
        with timer("update_actor", {}, color="red"):
            raise ValueError("original error")
    recorder.stream.close()
    rows = [json.loads(line) for line in (tmp_path / "phases.jsonl").open()]
    assert [(r["stage"], r["event"], r["step"]) for r in rows] == [
        ("step", "begin", 1), ("update_actor", "begin", 1),
        ("update_actor", "end", 1), ("step", "end", 1),
    ]
    assert calls == [(r["stage"], r["event"]) for r in rows]
    assert rows[0]["time_ns"] <= rows[-1]["time_ns"]


def test_vllm_metrics_parser_and_endpoint_discovery():
    script = Path(__file__).resolve().parents[4] / "scripts/collect_rollout_telemetry.py"
    spec = importlib.util.spec_from_file_location("rollout_telemetry", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.parse_prometheus('''# HELP ignored
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen 3B"} 0.125
vllm:num_requests_running{engine="0"} 123.0
vllm:num_requests_waiting{engine="0"} 4
vllm:request_prefill_time_seconds_sum{engine="0"} 1.5
vllm:request_decode_time_seconds_count{engine="0"} 12
irrelevant_metric 999
''')
    assert data["vllm:kv_cache_usage_perc"][0]["value"] == .125
    assert data["vllm:num_requests_running"][0]["value"] == 123
    assert "irrelevant_metric" not in data
    assert data['vllm:request_prefill_time_seconds_sum'][0]['value'] == 1.5
    assert data['vllm:request_decode_time_seconds_count'][0]['value'] == 12
    assert module.discover_servers("prefix LLMServerManager: ['127.0.0.1:123', '127.0.0.1:456']\n") == [
        "127.0.0.1:123", "127.0.0.1:456",
    ]

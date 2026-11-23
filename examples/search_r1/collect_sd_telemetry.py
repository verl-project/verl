"""Sample the allocation's whole-device memory and vLLM /metrics endpoints offline."""

import argparse
import ast
import datetime
import json
import re
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

METRICS = {
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason",
    "vllm:engine_sleep_state",
    "vllm:num_preemptions_total",
    "vllm:cache_config_info",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:iteration_tokens_total_bucket",
    "vllm:iteration_tokens_total_count",
    "vllm:iteration_tokens_total_sum",
}
for family in [
    "request_queue_time_seconds",
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
    "time_to_first_token_seconds",
    "inter_token_latency_seconds",
    "e2e_request_latency_seconds",
    "request_prefill_kv_computed_tokens",
]:
    METRICS.update("vllm:" + family + suffix for suffix in ["_bucket", "_sum", "_count"])
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+\S+)?$")


def parse_prometheus(body):
    result = {}
    for line in body.split("\n"):
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        if match and match[1] in METRICS:
            result.setdefault(match[1], []).append({"labels": match[2] or "", "value": float(match[3])})
    return result


def discover_servers(log):
    matches = re.findall(r"LLMServerManager: (\[[^\n]*?\])", log)
    return list(dict.fromkeys(address for match in matches for address in ast.literal_eval(match)))


def normalize_gpu_sample(body, logical_uuids, expected_count=4):
    rows = [line.split(",", 2) for line in body.strip().splitlines() if line.strip()]
    by_uuid = {uuid.strip(): (index.strip(), fields.strip()) for uuid, index, fields in rows}
    order = logical_uuids or list(by_uuid)
    assert len(order) == expected_count and len(set(order)) == expected_count, "Unexpected GPU count for this task"
    assert all(uuid in by_uuid for uuid in order), "Task CUDA GPU missing from NVML sample"
    mapping = [{"logical_index": i, "uuid": uuid, "nvml_index": by_uuid[uuid][0]} for i, uuid in enumerate(order)]
    return [f"{i}, {by_uuid[uuid][1]}" for i, uuid in enumerate(order)], mapping


def scrape(address, raw_dir):
    started = time.time_ns()
    row = {"endpoint": address, "sample_start_ns": started}
    try:
        # Internal worker traffic must not inherit a proxy from the development shell.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{address}/metrics", timeout=2) as response:
            body = response.read().decode()
        row["metrics"] = parse_prometheus(body)
        raw = raw_dir / (address.replace(":", "_") + ".prom")
        if not raw.exists() and "vllm:kv_cache_usage_perc" in row["metrics"]:
            raw.write_text(body)
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["sample_end_ns"] = time.time_ns()
    row["time_ns"] = (started + row["sample_end_ns"]) // 2
    row["time"] = datetime.datetime.fromtimestamp(row["time_ns"] / 1e9, datetime.timezone.utc).isoformat()
    return row


def gpu_sample(run, logical_uuids, expected_count):
    p = subprocess.run(
        [
            "nvidia-smi",
            *(["--id=" + ",".join(logical_uuids)] if logical_uuids else []),
            "--query-gpu=uuid,index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    p.check_returncode()
    gpus, device_map = normalize_gpu_sample(p.stdout, logical_uuids, expected_count)
    if not (run / "gpu-device-map.json").exists():
        (run / "gpu-device-map.json").write_text(json.dumps(device_map, indent=2) + "\n")
    return {"returncode": p.returncode, "gpus": gpus}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    run = args.run_dir
    runtime = json.loads((run / "runtime.json").read_text())
    expected_count = len(runtime["gpus"])
    logical_uuids = [gpu.get("uuid") for gpu in runtime["gpus"]]
    if not all(logical_uuids):
        logical_uuids = []
    raw_dir = run / "vllm-metrics-raw"
    raw_dir.mkdir(exist_ok=True)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    addresses = []
    with (
        (run / "gpu-memory.jsonl").open("a", buffering=1) as gpu_out,
        (run / "vllm-metrics.jsonl").open("a", buffering=1) as vllm_out,
        ThreadPoolExecutor(max_workers=expected_count) as pool,
    ):
        while running:
            tick = time.monotonic()
            started = time.time_ns()
            try:
                sample = gpu_sample(run, logical_uuids, expected_count)
            except (subprocess.SubprocessError, OSError, AssertionError) as error:
                # NVML sampling must not disable the independent HTTP collector.
                sample = {"returncode": -1, "gpus": [], "error": f"{type(error).__name__}: {error}"}
            ended = time.time_ns()
            gpu_out.write(
                json.dumps(
                    {
                        "time": datetime.datetime.fromtimestamp(
                            (started + ended) / 2e9, datetime.timezone.utc
                        ).isoformat(),
                        "sample_start_ns": started,
                        "sample_end_ns": ended,
                        "time_ns": (started + ended) // 2,
                        **sample,
                    }
                )
                + "\n"
            )
            if len(addresses) != expected_count:
                addresses = discover_servers((run / "worker.log").read_text(errors="replace"))
                if addresses:
                    (run / "vllm-discovery.json").write_text(
                        json.dumps({"endpoints": addresses, "interval_s": args.interval}, indent=2) + "\n"
                    )
            for row in pool.map(lambda a: scrape(a, raw_dir), addresses):
                vllm_out.write(json.dumps(row) + "\n")
            remaining = args.interval - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()

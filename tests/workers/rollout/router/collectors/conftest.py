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

"""Shared fixtures and helpers for collectors integration tests.

All three test modules (vllm_kv_event_collector, vllm_polling_collector,
route_data_provider_gpu_prefix_hit_rate) share one vLLM process for the
entire pytest session, avoiding redundant model loading.

The session-scoped ``vllm_kv_service`` fixture starts vLLM with ZMQ KV events
enabled.  Polling tests also use this server via the ``vllm_service`` alias —
the polling collector only reads the HTTP ``/metrics`` endpoint, which works
regardless of whether ZMQ events are enabled.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest

# ── Shared configuration constants ───────────────────────────────────────

VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B")
VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
NODE_ID = f"{VLLM_HOST}:{VLLM_PORT}"

ZMQ_SUB_PORT = int(os.environ.get("ZMQ_SUB_PORT", "5555"))
ZMQ_REPLAY_PORT = int(os.environ.get("ZMQ_REPLAY_PORT", "5556"))


# ── Shared helper ────────────────────────────────────────────────────────


def send_inference_request(node_id: str, model: str, prompt: str = "hello") -> bool:
    """POST a chat-completions request to trigger KV cache events.

    Returns True on HTTP 200, False on any error.
    """
    try:
        resp = httpx.post(
            f"http://{node_id}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.7,
            },
            timeout=30.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ── Session-scoped vLLM fixture ──────────────────────────────────────────


@pytest.fixture(scope="session")
def vllm_kv_service():
    """Start one vLLM server with ZMQ KV events for the entire test session.

    Binds:
      - HTTP API at http://<VLLM_HOST>:<VLLM_PORT>   (metrics + completions)
      - ZMQ PUB  at tcp://*:<ZMQ_SUB_PORT>            (KV-cache event stream)
      - ZMQ REP  at tcp://*:<ZMQ_REPLAY_PORT>          (event replay)

    Yields ``node_id`` (``"host:port"``).  The process is SIGTERM'd on teardown.
    """
    kv_events_config = json.dumps(
        {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{ZMQ_SUB_PORT}",
            "replay_endpoint": f"tcp://*:{ZMQ_REPLAY_PORT}",
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        VLLM_MODEL,
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        "--trust-remote-code",
        "--tensor_parallel_size",
        "2",
        "--dtype",
        "bfloat16",
        "--gpu_memory_utilization",
        "0.6",
        "--max-model-len",
        "8192",
        "--override_generation_config",
        '{"temperature": 0.8, "top_k": -1, "top_p": 0.9, "repetition_penalty": 1.0, "max_new_tokens": 4096}',
        "--kv-events-config",
        kv_events_config,
    ]

    proc = subprocess.Popen(cmd)

    metrics_url = f"http://{NODE_ID}/metrics"
    max_wait = 360
    deadline = time.time() + max_wait
    ready = False

    while time.time() < deadline:
        try:
            resp = httpx.get(metrics_url, timeout=5.0)
            if resp.status_code == 200:
                ready = True
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(3)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        pytest.skip(f"vLLM server for {VLLM_MODEL} did not become ready within {max_wait}s")

    yield NODE_ID

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="session")
def vllm_service(vllm_kv_service):
    """Alias: polling tests reuse the session-shared vLLM server.

    The polling collector only reads HTTP ``/metrics``; ZMQ KV events
    being enabled on the same server has no effect on polling tests.
    """
    return vllm_kv_service

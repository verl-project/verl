# Copyright 2026 Individual Contributor: HaitaoWuTJU
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
"""In-process preservation of partial requests during colocated training.

The default abort/re-prefill path does not use this module. The engine and its
scheduler remain alive; only the worker's physical allocations are suspended.
Historical KV is intentionally stale after a weight update.
"""

import asyncio
import time
from functools import wraps
from uuid import uuid4

PROTOCOL = "verl_partial_kv_full_pool_v1"
VERSION_SEGMENTS = "policy_version_segments"


def keep_partial_requests(step, total_steps, save_freq, test_freq):
    """Discard partial work at durable, evaluation and terminal boundaries."""
    return step < total_steps and not any(freq > 0 and step % freq == 0 for freq in (save_freq, test_freq))


class NativeWeightVersions:
    """Record actual EngineCore output versions, including delayed outputs."""

    def __init__(self):
        self.parts = []

    def append(self, token_ids, weight_version):
        """Append original token IDs with their native committed version."""
        if not token_ids:
            return
        if not isinstance(weight_version, str) or not weight_version.isdecimal():
            raise RuntimeError("Partial KV output has no native weight version")
        version = int(weight_version)
        if self.parts and version < self.parts[-1][0]:
            raise RuntimeError("Native output weight version moved backwards")
        if self.parts and self.parts[-1][0] == version:
            self.parts[-1][1].extend(token_ids)
        else:
            self.parts.append((version, list(token_ids)))

    def finish(self, token_ids):
        """Validate returned IDs and trim a stop token omitted by vLLM."""
        native = [token for _, ids in self.parts for token in ids]
        if native[: len(token_ids)] != list(token_ids):
            raise RuntimeError("Native version evidence differs from returned tokens")
        offset, segments = 0, []
        for version, ids in self.parts:
            length = min(len(ids), len(token_ids) - offset)
            if length <= 0:
                break
            segments.append(dict(offset=offset, length=length, policy_version=version))
            offset += length
        if offset != len(token_ids):
            raise RuntimeError("Native version evidence does not cover returned tokens")
        return segments


def serialize_worker_utilities(engine_core):
    """Keep ownership of the worker reply queue through observer cancellation."""
    if getattr(engine_core, "_verl_partial_kv_serialized", False):
        return
    original = engine_core.call_utility_async
    lock = asyncio.Lock()

    @wraps(original)
    async def call(method, *args, **kwargs):
        if method != "collective_rpc":
            return await original(method, *args, **kwargs)
        async with lock:
            pending = asyncio.create_task(original(method, *args, **kwargs))
            cancelled = None
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError as error:
                    cancelled = error
                except Exception:
                    break
            if cancelled is not None:
                # The native RPC cannot be cancelled safely: consume its reply
                # before allowing another reader of the multiprocess queue.
                try:
                    pending.result()
                except BaseException:
                    pass
                raise cancelled
            return pending.result()

    engine_core.call_utility_async = call
    engine_core._verl_partial_kv_serialized = True


class PartialKVOffload:
    """Fail-closed lifecycle owned by one TP replica's AsyncLLM frontend."""

    def __init__(self, server):
        self.server = server
        self.state = "awake"
        self.paused = False
        self.preserve = False
        self.refitted = False
        self.traces = {}
        self.receipts = []

    def install(self):
        """Install native output-version capture after AsyncLLM initialization."""
        engine = self.server.engine
        if not callable(getattr(engine.engine_core, "set_weight_version_async", None)):
            raise RuntimeError("Partial KV offload requires native vLLM weight-version support")
        serialize_worker_utilities(engine.engine_core)
        processor = engine.output_processor
        original = processor.process_outputs

        @wraps(original)
        def process(outputs, *args, **kwargs):
            for output in outputs:
                request = processor.request_states.get(output.request_id)
                external = getattr(request, "external_req_id", output.request_id)
                trace = self.traces.get(external)
                if trace is not None:
                    trace.append(output.new_token_ids, kwargs.get("weight_version"))
            return original(outputs, *args, **kwargs)

        processor.process_outputs = process

    def _healthy(self):
        if self.state == "failed":
            raise RuntimeError("Partial KV handoff failed; restart the engine instead of resuming")

    async def fence(self, timeout=300):
        """Wait for pre-pause outputs, including the last dequeued batch."""
        engine = self.server.engine
        await engine.collective_rpc("partial_kv_stats")
        deadline = time.monotonic() + timeout
        while engine.output_processor.request_states:
            handler = engine.output_handler
            if handler is None or handler.done():
                raise RuntimeError("vLLM output handler stopped with live partial requests")
            queue = engine.engine_core.outputs_queue
            getters = getattr(queue, "_getters", None)
            if getters is None:
                raise RuntimeError("Unsupported vLLM output-queue fence")
            if queue.empty() and any(not future.done() for future in getters):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Paused vLLM outputs did not drain")
            await asyncio.sleep(0.001)

    async def pause(self, preserve, timeout=60):
        """Close admission before acknowledging keep/abort on the scheduler."""
        self._healthy()
        if type(preserve) is not bool:
            raise ValueError("Partial KV preservation must be an explicit boolean")
        if self.paused:
            if self.preserve != preserve:
                raise RuntimeError("Cannot change preservation mode during a handoff")
            return
        if self.state != "awake":
            raise RuntimeError("Pause requests before offloading their memory")
        server = self.server
        server._submission_paused = True
        server._rejecting = False
        server._resume_event.clear()
        self.state = "failed"
        deadline = time.monotonic() + timeout
        while server._admitting:
            if time.monotonic() >= deadline:
                raise TimeoutError("Partial KV admission barrier did not close")
            await asyncio.sleep(0.01)
        await server.engine.pause_generation(mode="keep" if preserve else "abort", clear_cache=not preserve)
        self.paused = True
        self.preserve = preserve
        await self.fence()
        self.state = "awake"

    async def drain(self):
        """Keep live requests when preserved; drain only the discard path."""
        self._healthy()
        if not self.paused:
            # Initial native vLLMReplica.sleep() precedes any trainer pause.
            await self.pause(False)
        if not self.preserve:
            await self.server.engine.wait_for_requests_to_drain()

    def _check_receipts(self, receipts, **expected):
        count = self.server.config.tensor_model_parallel_size
        if len(receipts) != count or any(
            not isinstance(row, dict)
            or row.get("protocol") != PROTOCOL
            or any(row.get(key) != value for key, value in expected.items())
            for row in receipts
        ):
            raise RuntimeError("Incomplete TP acknowledgement for partial KV handoff")
        self.receipts = receipts

    async def sleep(self):
        """Back up full native pools using worker-level sleep, preserving scheduling."""
        self._healthy()
        if self.state in ("asleep", "weights_awake"):
            return
        await self.drain()
        self.refitted = False
        self.state = "failed"
        receipts = await self.server.engine.collective_rpc(
            "offload_partial_kv", args=(self.preserve, self.server.config.partial_rollout_kv_offload_headroom_gb)
        )
        self._check_receipts(receipts, kv_preserved=self.preserve, asleep=True)
        self.state = "asleep"

    async def wake(self, tags):
        """Restore weights before KV; scheduling remains paused throughout."""
        self._healthy()
        if tags is None:
            tags = ["weights", "kv_cache"]
        if not isinstance(tags, list) or not tags or any(tag not in ("weights", "kv_cache") for tag in tags):
            raise ValueError("Partial KV wake tags must contain weights and/or kv_cache")
        if not self.paused:
            raise RuntimeError("Restore partial KV with scheduling paused")
        if self.state == "awake" or self.state == "weights_awake" and set(tags) == {"weights"}:
            return
        if "kv_cache" in tags and ("weights" in tags or self.state != "weights_awake" or not self.refitted):
            raise RuntimeError("Complete weight refit before restoring the KV pool")
        self.state = "failed"
        receipts = await self.server.engine.collective_rpc("restore_partial_kv", args=(tags,))
        self._check_receipts(receipts, tags=tags, kv_preserved=self.preserve, runtime_restored=True)
        self.state = "awake" if "kv_cache" in tags else "weights_awake"

    async def commit_version(self, version):
        """Publish the native version only after the weight receiver succeeds."""
        self._healthy()
        if not self.paused or self.state != "weights_awake":
            raise RuntimeError("Commit weights while paused, before KV restoration")
        if type(version) is not int or version < 0:
            raise ValueError("Weight version must be a nonnegative integer")
        previous = self.server.global_steps
        if previous is not None and version < previous:
            raise ValueError("Weight version moved backwards")
        self.state = "failed"
        await self.server.engine.engine_core.set_weight_version_async(str(version))
        self.refitted = True
        self.state = "weights_awake"

    async def resume(self):
        """Resume only after all pools and the new weight version are ready."""
        self._healthy()
        if self.state != "awake" or not self.refitted:
            raise RuntimeError("Complete KV restoration and weight refit before resuming")
        if self.paused:
            await self.server.engine.resume_generation()
            self.paused = False

    async def generate(self, **kwargs):
        """Retain native tokens/logprobs and isolate each physical request's prefix."""
        request_id = kwargs["request_id"]
        if request_id in self.traces:
            raise RuntimeError("Duplicate partial-rollout request identity")
        trace = NativeWeightVersions()
        self.traces[request_id] = trace
        # Salt at actual admission, after the server's submission gate. This
        # also isolates a fresh retry with the same external request_id.
        kwargs["prompt"]["cache_salt"] = f"verl:{request_id}:{self.server.global_steps}:{uuid4().hex}"
        try:
            output = await self.server._generate_admitted(**kwargs)
            ids = output.outputs[0].token_ids if output.outputs else []
            return output, trace.finish(ids)
        finally:
            self.traces.pop(request_id, None)

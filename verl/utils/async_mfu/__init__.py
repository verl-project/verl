"""Dependency-free, execution-based estimated MFU instrumentation."""
import atexit
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.request
import uuid

VERSION = "qwen3-moe-verl-training-reference-causal-generation-v2"
_writer = None


def enabled():
    return os.environ.get("ASYNC_MFU_ENABLED", "0") == "1"


class Writer:
    def __init__(self):
        self.directory = Path(os.environ["ASYNC_MFU_DIR"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.producer = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
        self.lock = threading.Lock()
        self.sequence = 0
        self.anchor = time.monotonic()
        self.offset, self.uncertainty = self.calibrate()
        self.file = (self.directory / f"events-{self.producer}.jsonl").open("a", buffering=1)
        self.emit("producer", uncertainty_s=self.uncertainty, hostname=socket.gethostname())
        atexit.register(self.file.close)
        threading.Thread(target=self.heartbeat, daemon=True).start()

    def heartbeat(self):
        last_check = time.monotonic()
        while True:
            time.sleep(2)
            try:
                self.emit("watermark")
                if time.monotonic() - last_check >= 60:
                    offset, uncertainty = self.calibrate()
                    # Keep the original time axis; report drift rather than jump timestamps.
                    self.emit("clock_check", offset_change_s=offset - self.offset,
                              uncertainty_s=uncertainty)
                    last_check = time.monotonic()
            except Exception:
                # Final replay will reject insufficient watermarks; training need not stall.
                return

    def calibrate(self):
        samples = []
        for _ in range(5):
            start = time.monotonic()
            with urllib.request.urlopen(os.environ["ASYNC_MFU_CLOCK_URL"], timeout=5) as response:
                server = json.load(response)["monotonic"]
            end = time.monotonic()
            samples.append((end - start, server - (start + end) / 2))
        rtt, offset = min(samples)
        return offset, rtt / 2

    def now(self):
        return time.monotonic() + self.offset

    def emit(self, kind, **data):
        with self.lock:
            self.sequence += 1
            record = dict(kind=kind, producer=self.producer, sequence=self.sequence,
                          timestamp=self.now(), **data)
            self.file.write(json.dumps(record, allow_nan=False) + "\n")
        return record


def writer():
    global _writer
    if _writer is None:
        _writer = Writer()
    return _writer


def actor_start():
    if not enabled():
        return None
    event_id = uuid.uuid4().hex
    record = writer().emit("begin", event_id=event_id, role="actor")
    return event_id, record["timestamp"]


def actor_end(ticket, flops, tokens, device_name):
    if ticket is not None:
        writer().emit("execution", event_id=ticket[0], role="actor", start=ticket[1],
                      end=writer().now(), actor_flops=flops, actor_tokens=tokens,
                      device_name=device_name)


def reference_start():
    if not enabled():
        return None
    event_id = uuid.uuid4().hex
    record = writer().emit("begin", event_id=event_id, role="reference")
    return event_id, record["timestamp"]


def reference_end(ticket, counter, sequence_lengths, device_name):
    if ticket is not None:
        if counter is None or not sequence_lengths:
            raise ValueError("Async MFU requires Reference sequence lengths and a FLOPs counter")
        # Match verl's forward_only MFU: the counter estimates forward + backward,
        # while the frozen Reference runs one forward pass without recomputation.
        estimated, _ = counter.estimate_flops(sequence_lengths, 1.0)
        writer().emit("execution", event_id=ticket[0], role="reference", start=ticket[1],
                      end=writer().now(), reference_flops=estimated * 1e12 / 3.0,
                      reference_tokens=sum(sequence_lengths), device_name=device_name)


def generation_work(config, scheduled, requests):
    """Estimate executed model work; scheduled request counters already advanced."""
    if config.model_type != "qwen3_moe":
        raise ValueError("Async MFU currently supports qwen3_moe only")
    h, layers = config.hidden_size, config.num_hidden_layers
    d, heads, kv = config.head_dim, config.num_attention_heads, config.num_key_value_heads
    linear = layers * (h * (2 * heads * d + 2 * kv * d)
                       + 3 * h * config.num_experts_per_tok * config.moe_intermediate_size
                       + h * config.num_experts)
    result = dict(prefill_flops=0.0, decode_flops=0.0, prefill_tokens=0, decode_tokens=0,
                  cached_prefix_tokens=0, requests=[])
    for request_id, count in scheduled.num_scheduled_tokens.items():
        request = requests[request_id]
        end = request.num_computed_tokens
        start = end - count
        if start < 0:
            raise ValueError("Scheduled token position is negative")
        prompt = request.num_prompt_tokens
        prefill = max(0, min(end, prompt) - min(start, prompt))
        decode = count - prefill
        # Arithmetic series of visible KV lengths, with causal attention.
        for phase, first, n in (("prefill", start, prefill), ("decode", start + prefill, decode)):
            pairs = n * (2 * first + n + 1) / 2
            flops = 2 * linear * n + 4 * layers * heads * d * pairs
            # vLLM samples at the last prompt position and each decode position.
            logits = n if phase == "decode" else int(n > 0 and end >= prompt)
            flops += 2 * h * config.vocab_size * logits
            result[f"{phase}_flops"] += flops
            result[f"{phase}_tokens"] += n
        result["requests"].append(dict(request_id=request_id, start_position=start,
                                       computed_tokens=count, prompt_tokens=prompt))
    result["cached_prefix_tokens"] = sum(
        min(r.num_computed_tokens, len(r.prompt_token_ids or []))
        for r in scheduled.scheduled_new_reqs
    )
    return result


def rollout_start(core, scheduled):
    if not enabled() or not scheduled.total_num_scheduled_tokens:
        return None
    if core.use_spec_decode or not core.is_ec_consumer:
        raise ValueError("Async MFU does not support speculative decode or non-consumer engines")
    work = generation_work(core.vllm_config.model_config.hf_config, scheduled, core.scheduler.requests)
    event_id = uuid.uuid4().hex
    record = writer().emit("begin", event_id=event_id, role="rollout")
    return event_id, record["timestamp"], work


def rollout_end(ticket):
    if ticket is not None:
        writer().emit("execution", event_id=ticket[0], start=ticket[1], end=writer().now(),
                      role="rollout", **ticket[2])


def window_start(step):
    if enabled():
        record = writer().emit("window_begin", step=step)
        return record["timestamp"]
    return None


def window_end(step, start):
    if start is not None:
        writer().emit("window", step=step, start=start, end=writer().now())

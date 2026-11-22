"""Optional wall-clock phase markers; the original verl timers remain in use."""

import datetime
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseRecorder:
    stages = {"step", "gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights", "save_checkpoint"}

    def __init__(self, path):
        self.stream = Path(path).open("a", buffering=1)
        self.step = 0

    def emit(self, stage, event):
        now = time.time_ns()
        self.stream.write(json.dumps({
            "time_ns": now,
            "time": datetime.datetime.fromtimestamp(now / 1e9, datetime.timezone.utc).isoformat(),
            "step": self.step, "stage": stage, "event": event, "pid": os.getpid(),
        }) + "\n")

    def wrap(self, original):
        @contextmanager
        def marked_timer(name, *args, **kwargs):
            if name not in self.stages:
                with original(name, *args, **kwargs):
                    yield
                return
            if name == "step":
                self.step += 1
            self.emit(name, "begin")
            try:
                with original(name, *args, **kwargs):
                    yield
            finally:
                self.emit(name, "end")
        return marked_timer


def install_phase_timers(run_dir):
    from verl.trainer.ppo.v1 import trainer_base, trainer_sync

    if getattr(trainer_base.marked_timer, "search_r1_phase_recorder", None):
        return
    recorder = PhaseRecorder(Path(run_dir) / "phase-events.jsonl")
    for module in [trainer_base, trainer_sync]:
        wrapped = recorder.wrap(module.marked_timer)
        wrapped.search_r1_phase_recorder = recorder
        module.marked_timer = wrapped

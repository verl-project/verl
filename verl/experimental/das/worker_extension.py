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
"""vLLM worker extension that hosts DAS inside every TP-rank GPU worker.

``vLLMHttpServer._get_worker_extension_cls`` returns
``vLLMDASColocateWorkerExtension`` when ``rollout.das.enable`` is set; vLLM
mixes it into the Worker class of each TP-rank process, so (1) ``__new__``
runs in that process before the model runner and its drafter exist, which is
where the suffix proposer patch is installed, and (2) the ``das_*`` methods are
reachable through ``collective_rpc``, which is how delta batches arrive.

The DAS config reaches the worker process as JSON in ``VERL_DAS_CONFIG``, set
by ``vLLMHttpServer.__init__`` and inherited by its subprocesses (the same hop
``VERL_SEED`` uses).
"""

from __future__ import annotations

import logging
import os

from verl.experimental.das.drafter_state import get_active_state
from verl.workers.config.das import DASConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DAS_CONFIG_ENV = "VERL_DAS_CONFIG"
_warned_no_suffix = False


def load_worker_config(environ=None) -> DASConfig | None:
    """The ``DASConfig`` handed down by the server actor, or None when DAS is off."""
    payload = (environ if environ is not None else os.environ).get(DAS_CONFIG_ENV)
    if not payload:
        return None
    try:
        cfg = DASConfig.from_json(payload)
    except Exception as e:
        logger.warning("DAS: ignoring malformed %s (%s)", DAS_CONFIG_ENV, e)
        return None
    return cfg if cfg.enable else None


def maybe_install_proposer_patch(vllm_config, environ=None) -> bool:
    """Install the DAS proposer patch if DAS is enabled and vLLM drafts with ``method=suffix``."""
    global _warned_no_suffix
    cfg = load_worker_config(environ)
    if cfg is None:
        return False
    spec = getattr(vllm_config, "speculative_config", None)
    method = getattr(spec, "method", None)
    if method != "suffix":
        if not _warned_no_suffix:
            _warned_no_suffix = True
            logger.warning(
                "DAS: rollout.das.enable=True but vLLM speculative method is %r (need 'suffix'); DAS inactive",
                method,
            )
        return False
    from verl.experimental.das.proposer_patch import install_proposer_patch

    return install_proposer_patch(cfg)


class DASWorkerExtensionMixin:
    """DAS hooks for a vLLM worker: patch installation at construction plus the delta RPCs."""

    def __new__(cls, **kwargs):
        try:
            maybe_install_proposer_patch(kwargs.get("vllm_config"))
        except Exception:
            logger.exception("DAS: proposer patch installation failed; running stock decoding")
        try:
            return super().__new__(cls, **kwargs)
        except TypeError:
            # Next class in the MRO is ``object`` (tests): it takes no kwargs.
            return super().__new__(cls)

    def das_apply_deltas(self, payload: bytes) -> int:
        """Apply a canonical delta batch; returns the new engine-side state version (-1 if inactive)."""
        state = get_active_state()
        if state is None:
            return -1
        state.apply_delta_batch(payload)
        stats = state.tree_stats()
        logger.info(
            "DAS: applied delta batch -> store version %d (iteration %s, trees=%s, drafts_emitted=%s)",
            state.state_version(),
            stats.get("iteration"),
            stats.get("problem_trees"),
            stats.get("drafts_emitted"),
        )
        return state.state_version()


try:
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
except Exception:  # vLLM is not importable here (CPU tests); the concrete class is only used in engines.
    vLLMColocateWorkerExtension = None  # type: ignore[assignment,misc]

if vLLMColocateWorkerExtension is not None:

    class vLLMDASColocateWorkerExtension(DASWorkerExtensionMixin, vLLMColocateWorkerExtension):
        """verl's colocate worker extension (weight sync) plus the DAS hooks."""

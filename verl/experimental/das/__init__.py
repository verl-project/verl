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
"""DAS: distribution-aware speculative decoding for RL rollouts (arXiv 2511.13841).

Built on vLLM's suffix-decoding drafter (``speculative_config.method=suffix``).
DAS adds (1) per-problem suffix trees, fed by a central job-scoped store that
collects every rollout trajectory and redistributes them to all engines as
versioned deltas, and (2) length-aware draft budgets per problem. The decode
hot path never touches the network: trees live inside each TP-rank worker.

Enable with ``actor_rollout_ref.rollout.das.enable=true``; see
``docs/advance/das.md``. Submodules are imported lazily by the integration
points, so importing this package pulls in neither Ray nor vLLM.

Components (one module each):

- ``problem_id``: first-turn prompt hash -> ``dasp{hash}-{uuid}`` engine request id.
- ``push_client``: batched, fire-and-forget trajectory pushes from agent-loop workers.
- ``service``: ``SuffixDataService``, the versioned sliding-window trajectory store.
- ``drafter_state``: engine-local per-problem trees, transient self-inserts, budgets.
- ``proposer_patch``: in-place extension of vLLM's ``SuffixDecodingProposer``.
- ``worker_extension``: TP-rank RPC receiver for delta batches.
- ``pump``: per-engine delta pump (store -> ``collective_rpc``).
- ``client`` / ``agent_loop`` / ``agent_loop_tq``: verl server-client and agent-loop integration.
"""

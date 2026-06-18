# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Vendored instruction-following constraint library from swiss-ai/if-gym.

Only ``instructions_registry``, ``instructions`` and ``instructions_util`` are
vendored; in-package imports were rewritten to relative form. This is the
constraint checker used both by the IFGym agent loop (per-turn rollout scoring)
and by ``ifgym_mt_reward.compute_score``.
"""

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
"""Rollout servers that delegate `generate` to a remote RL backend.

Each sub-package wraps an in-process rollout server (e.g.
``vLLMHttpServer``) and routes prompts to the backend's `generate`
endpoint instead of running the model locally. Used by adapters that
co-train sampling and training in a separate cluster.
"""

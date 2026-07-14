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
"""Teacher worker package for FSDP-served distillation teachers.

The :class:`TeacherFSDPWorker` exposed here is the FSDP-backed alternative to
the vLLM teacher rollout managed by
:class:`verl.experimental.teacher_loop.MultiTeacherModelManager`. It is wired
into the trainer when ``distillation.teacher_backend == 'fsdp'``.
"""

from .fsdp_teacher_worker import TeacherFSDPWorker
from .utils import chunked_gather_logprobs

__all__ = ["TeacherFSDPWorker", "chunked_gather_logprobs"]

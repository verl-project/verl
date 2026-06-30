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

import torch

from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer


def test_build_optimizer_with_soap():
    model = torch.nn.Linear(2, 2, bias=False)
    config = FSDPOptimizerConfig(
        lr=3e-3,
        betas=(0.95, 0.95),
        optimizer="SOAP",
        optimizer_impl="verl.optimizers.soap",
        override_optimizer_config={
            "precondition_frequency": 2,
            "max_precond_dim": 8,
            "merge_dims": False,
            "precondition_1d": True,
        },
    )

    optimizer = build_optimizer(model.parameters(), config)
    assert optimizer.__class__.__name__ == "SOAP"

    loss = model(torch.randn(4, 2)).sum()
    loss.backward()
    optimizer.step()

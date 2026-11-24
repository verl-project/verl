# SPDX-License-Identifier: Apache-2.0
"""Native K3.1 configuration exports with existing checkpoint loader aliases."""
from fsdp_turbo.models.kimi.kimi_k31.configuration_kimi_k31 import (
    KimiK31Config,
    KimiK31VisionConfig,
    KimiLinearConfig,
)

KimiK3Config = KimiK31Config
KimiK3VisionConfig = KimiK31VisionConfig

__all__ = ["KimiK31Config", "KimiK31VisionConfig", "KimiK3Config", "KimiK3VisionConfig", "KimiLinearConfig"]

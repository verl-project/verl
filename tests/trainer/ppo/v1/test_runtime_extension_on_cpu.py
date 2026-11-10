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

from dataclasses import dataclass, field

import pytest

from verl.trainer.runtime_extension import (
    RUNTIME_EXTENSION_API_VERSION,
    RuntimeCapability,
    RuntimeEvent,
    RuntimeExtensionContext,
    RuntimeExtensionRegistry,
    RuntimeExtensionRestartRequired,
    SafePoint,
    SafePointDecision,
)


@dataclass
class RecordingExtension:
    name: str = "recording"
    api_version: int = RUNTIME_EXTENSION_API_VERSION
    required_capabilities: tuple[RuntimeCapability, ...] = (
        RuntimeCapability.EVENTS,
        RuntimeCapability.SAFE_POINTS,
    )
    events: list[RuntimeEvent] = field(default_factory=list)
    safe_points: list[SafePoint] = field(default_factory=list)
    decision: SafePointDecision = SafePointDecision.CONTINUE

    def on_event(self, context: RuntimeExtensionContext, event: RuntimeEvent) -> None:
        assert context.api_version == RUNTIME_EXTENSION_API_VERSION
        self.events.append(event)

    def on_safe_point(self, context: RuntimeExtensionContext, safe_point: SafePoint) -> SafePointDecision:
        assert context.run_id
        self.safe_points.append(safe_point)
        return self.decision


def test_register_deliver_and_close_extension():
    registry = RuntimeExtensionRegistry(
        capabilities=(RuntimeCapability.EVENTS, RuntimeCapability.SAFE_POINTS), run_id="run-1"
    )
    extension = RecordingExtension()

    assert not registry
    registration = registry.register(extension)
    assert registry
    event = registry.publish("trainer.started", step=0)
    registry.reach(SafePoint.BEFORE_FIT)

    assert extension.events == [event]
    assert event.payload == {"step": 0}
    assert extension.safe_points == [SafePoint.BEFORE_FIT]

    registration.close()
    registration.close()
    assert not registry
    registry.publish("trainer.stopped")
    registry.reach(SafePoint.AFTER_FIT)
    assert extension.events == [event]
    assert extension.safe_points == [SafePoint.BEFORE_FIT]


def test_registration_rejects_duplicate_version_and_missing_capability():
    registry = RuntimeExtensionRegistry(capabilities=(RuntimeCapability.EVENTS,))
    extension = RecordingExtension(required_capabilities=(RuntimeCapability.EVENTS,))
    registry.register(extension)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(extension)

    incompatible = RecordingExtension(name="incompatible", api_version=99)
    with pytest.raises(ValueError, match="requires API version 99"):
        registry.register(incompatible)

    missing = RecordingExtension(name="missing", required_capabilities=(RuntimeCapability.SAFE_POINTS,))
    with pytest.raises(ValueError, match="unavailable capabilities: safe_points"):
        registry.register(missing)


def test_restart_decision_stops_at_safe_point():
    registry = RuntimeExtensionRegistry(capabilities=(RuntimeCapability.SAFE_POINTS,))
    extension = RecordingExtension(
        required_capabilities=(RuntimeCapability.SAFE_POINTS,),
        decision=SafePointDecision.RESTART_REQUIRED,
    )
    registry.register(extension)

    with pytest.raises(RuntimeExtensionRestartRequired, match="requested restart at step_boundary"):
        registry.reach(SafePoint.STEP_BOUNDARY)

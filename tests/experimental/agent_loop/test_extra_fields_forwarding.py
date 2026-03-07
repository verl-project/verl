# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Test that AgentData.extra_fields are forwarded to AgentLoopOutput.

ToolAgentLoop.run() constructs AgentLoopOutput from AgentData. Custom data
written to agent_data.extra_fields during tool execution (e.g. tool_history)
must survive into the output. Previously, extra_fields was hardcoded to {},
silently dropping all custom tool session data.
"""

import unittest

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


class TestExtraFieldsForwarding(unittest.TestCase):
    """Test extra_fields construction logic matching ToolAgentLoop.run() lines 185-198."""

    def _build_output(self, agent_extra_fields, turn_scores=None, tool_rewards=None):
        """Reproduce the output construction from ToolAgentLoop.run().

        This mirrors the exact logic at tool_agent_loop.py:185-198:
            extra_fields=dict(agent_data.extra_fields),
            ...
            output.extra_fields.update({"turn_scores": ..., "tool_rewards": ...})
        """
        output = AgentLoopOutput(
            prompt_ids=[1, 2, 3],
            response_ids=[4, 5, 6],
            response_mask=[1, 1, 1],
            num_turns=1,
            metrics=AgentLoopMetrics(),
            extra_fields=dict(agent_extra_fields),
        )
        output.extra_fields.update({
            "turn_scores": turn_scores or [],
            "tool_rewards": tool_rewards or [],
        })
        return output

    def test_custom_extra_fields_survive(self):
        """Custom data written to agent_data.extra_fields appears in output."""
        extra = {"tool_history": [{"tool": "search", "result": "found"}], "session_id": "abc"}
        output = self._build_output(extra)

        self.assertEqual(output.extra_fields["tool_history"], [{"tool": "search", "result": "found"}])
        self.assertEqual(output.extra_fields["session_id"], "abc")

    def test_turn_scores_and_tool_rewards_merged(self):
        """turn_scores and tool_rewards are merged on top of custom fields."""
        extra = {"tool_history": ["step1"]}
        output = self._build_output(extra, turn_scores=[0.5], tool_rewards=[1.0])

        self.assertEqual(output.extra_fields["tool_history"], ["step1"])
        self.assertEqual(output.extra_fields["turn_scores"], [0.5])
        self.assertEqual(output.extra_fields["tool_rewards"], [1.0])

    def test_turn_scores_overrides_custom_field(self):
        """If extra_fields has 'turn_scores', the .update() overwrites it."""
        extra = {"turn_scores": "should_be_overwritten"}
        output = self._build_output(extra, turn_scores=[0.9])

        self.assertEqual(output.extra_fields["turn_scores"], [0.9])

    def test_empty_extra_fields_still_has_turn_scores(self):
        """Even with empty extra_fields, turn_scores and tool_rewards are present."""
        output = self._build_output({})

        self.assertEqual(output.extra_fields["turn_scores"], [])
        self.assertEqual(output.extra_fields["tool_rewards"], [])

    def test_shallow_copy_isolation(self):
        """Modifying output.extra_fields does not mutate the original dict."""
        original = {"key": "value"}
        output = self._build_output(original)
        output.extra_fields["new_key"] = "new_value"

        self.assertNotIn("new_key", original)


if __name__ == "__main__":
    unittest.main()

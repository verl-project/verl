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
"""Structured mock trajectories for CT-vs-legacy agent-loop tokenization checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema


@dataclass(frozen=True)
class TrajectoryStep:
    """One assistant generation followed by optional environment messages."""

    assistant: dict[str, Any]
    appended_messages: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SingleTurnTrajectory:
    """A single prompt/assistant-response trajectory for SingleTurnAgentLoop."""

    name: str
    description: str
    raw_prompt: tuple[dict[str, Any], ...]
    assistant_response: str
    expected_num_turns: int = 2


@dataclass(frozen=True)
class ToolAgentTrajectory:
    """A messages-level trajectory independent of any rollout server."""

    name: str
    description: str
    raw_prompt: tuple[dict[str, Any], ...]
    steps: tuple[TrajectoryStep, ...]
    tools: tuple[FunctionTool, ...]
    max_parallel_calls: int = 1

    @property
    def expected_generation_turns(self) -> int:
        return len(self.steps)

    @property
    def expected_num_turns(self) -> int:
        environment_turns = sum(1 for step in self.steps if step.appended_messages)
        return 1 + self.expected_generation_turns + environment_turns

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in self.tools]

    def full_messages(self) -> list[dict[str, Any]]:
        messages = [dict(message) for message in self.raw_prompt]
        for step in self.steps:
            messages.append(_clone_message(step.assistant))
            messages.extend(_clone_message(message) for message in step.appended_messages)
        return messages


MockTrajectory = SingleTurnTrajectory | ToolAgentTrajectory


def _clone_message(message: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(message, ensure_ascii=False))


def _schema(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    )


def _assistant(content: str, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _tool_message(call_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    }


def get_weather(city: str, day: str) -> dict[str, Any]:
    """Return a deterministic weather snapshot for a city and day."""
    table = {
        ("Seattle", "today"): {"condition": "rain", "high_c": 12, "wind_kph": 22},
        ("Portland", "today"): {"condition": "cloudy", "high_c": 16, "wind_kph": 11},
        ("Denver", "tomorrow"): {"condition": "clear", "high_c": 24, "wind_kph": 18},
        ("Boulder", "tomorrow"): {"condition": "storms", "high_c": 21, "wind_kph": 31},
    }
    return {"city": city, "day": day, **table.get((city, day), {"condition": "unknown", "high_c": None})}


def estimate_travel(origin: str, destination: str, depart_window: str) -> dict[str, Any]:
    """Estimate deterministic travel time for a simple route."""
    route = f"{origin}->{destination}"
    table = {
        ("Depot A", "West Clinic", "morning"): {"minutes": 32, "risk": "low"},
        ("Depot A", "East Clinic", "morning"): {"minutes": 51, "risk": "medium"},
        ("Depot B", "West Clinic", "afternoon"): {"minutes": 44, "risk": "medium"},
        ("Depot B", "East Clinic", "afternoon"): {"minutes": 28, "risk": "low"},
    }
    return {"route": route, "depart_window": depart_window, **table.get((origin, destination, depart_window), {})}


def reserve_room(room: str, start_time: str, end_time: str | None = None) -> dict[str, Any]:
    """Reserve a room if the request has a complete time range."""
    if not end_time:
        return {"ok": False, "error": "end_time is required", "room": room, "start_time": start_time}
    return {"ok": True, "room": room, "start_time": start_time, "end_time": end_time, "confirmation": "room-4821"}


def lookup_incident(incident_id: str) -> dict[str, Any]:
    """Look up a deterministic support incident."""
    if incident_id != "INC-1042":
        return {"ok": False, "error": "incident_id must use the INC-#### format", "received": incident_id}
    return {"ok": True, "incident_id": incident_id, "status": "mitigated", "owner": "edge-oncall"}


def build_common_tools() -> tuple[FunctionTool, ...]:
    return (
        FunctionTool(
            name="get_weather",
            fn=get_weather,
            tool_schema=_schema(
                "get_weather",
                "Return a deterministic weather snapshot.",
                {
                    "city": {"type": "string", "description": "City to check."},
                    "day": {"type": "string", "description": "Day label such as today or tomorrow."},
                },
                ["city", "day"],
            ),
        ),
        FunctionTool(
            name="estimate_travel",
            fn=estimate_travel,
            tool_schema=_schema(
                "estimate_travel",
                "Estimate deterministic travel time for a route.",
                {
                    "origin": {"type": "string", "description": "Starting location."},
                    "destination": {"type": "string", "description": "Destination location."},
                    "depart_window": {"type": "string", "description": "Departure window label."},
                },
                ["origin", "destination", "depart_window"],
            ),
        ),
        FunctionTool(
            name="reserve_room",
            fn=reserve_room,
            tool_schema=_schema(
                "reserve_room",
                "Reserve a meeting room with an explicit time range.",
                {
                    "room": {"type": "string", "description": "Room name."},
                    "start_time": {"type": "string", "description": "Local start time."},
                    "end_time": {"type": ["string", "null"], "description": "Local end time."},
                },
                ["room", "start_time"],
            ),
        ),
        FunctionTool(
            name="lookup_incident",
            fn=lookup_incident,
            tool_schema=_schema(
                "lookup_incident",
                "Look up an incident status.",
                {
                    "incident_id": {"type": "string", "description": "Incident id in INC-#### format."},
                },
                ["incident_id"],
            ),
        ),
    )


TOOLS = build_common_tools()


# system + user -> assistant
SINGLE_TURN_CHAT = SingleTurnTrajectory(
    name="singleturnchat",
    description="One user request followed by one plain assistant response.",
    raw_prompt=(
        {"role": "system", "content": "You answer scheduling questions in one concise sentence."},
        {"role": "user", "content": "Can I move the Friday demo to 15:30 if the room is free?"},
    ),
    assistant_response="Yes, move the Friday demo to 15:30 if the room remains free and notify the attendees.",
)


# system + user
# -> assistant(tool_call: get_weather)
# -> tool(get_weather)
# -> assistant(tool_call: get_weather)
# -> tool(get_weather)
# -> assistant
MULTITURN_SINGLE_TOOL = ToolAgentTrajectory(
    name="multiturnsingletool",
    description="Two sequential assistant tool calls, one tool response per turn, then a final answer.",
    raw_prompt=(
        {"role": "system", "content": "You are a concise trip-planning assistant. Use tools for live facts."},
        {
            "role": "user",
            "content": "Pick a better city for an outdoor team lunch today: Seattle or Portland. Check both first.",
        },
    ),
    tools=TOOLS,
    steps=(
        TrajectoryStep(
            assistant=_assistant(
                "I will check Seattle first, then compare it with Portland.",
                [_tool_call("call_weather_seattle", "get_weather", {"city": "Seattle", "day": "today"})],
            ),
            appended_messages=(
                _tool_message(
                    "call_weather_seattle",
                    "get_weather",
                    {"city": "Seattle", "day": "today", "condition": "rain", "high_c": 12, "wind_kph": 22},
                ),
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "Seattle looks wet, so I will check Portland before deciding.",
                [_tool_call("call_weather_portland", "get_weather", {"city": "Portland", "day": "today"})],
            ),
            appended_messages=(
                _tool_message(
                    "call_weather_portland",
                    "get_weather",
                    {"city": "Portland", "day": "today", "condition": "cloudy", "high_c": 16, "wind_kph": 11},
                ),
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "Choose Portland. It is cloudy rather than rainy, warmer, and less windy; "
                "keep Seattle as the indoor fallback."
            )
        ),
    ),
)


# system + user
# -> assistant(two parallel tool_calls: estimate_travel + get_weather)
# -> tool(estimate_travel) + tool(get_weather)
# -> assistant(two parallel tool_calls: estimate_travel + get_weather)
# -> tool(estimate_travel) + tool(get_weather)
# -> assistant
MULTITURN_MULTI_TOOL = ToolAgentTrajectory(
    name="multiturnmultitool",
    description="Two assistant turns with parallel tool calls, followed by a final comparison.",
    raw_prompt=(
        {"role": "system", "content": "You coordinate clinic deliveries and summarize tradeoffs briefly."},
        {
            "role": "user",
            "content": (
                "Compare morning deliveries from Depot A to the West and East clinics, then compare the afternoon "
                "Depot B options. Use weather and travel estimates before recommending a route."
            ),
        },
    ),
    tools=TOOLS,
    max_parallel_calls=2,
    steps=(
        TrajectoryStep(
            assistant=_assistant(
                "I will check the morning route and destination weather together.",
                [
                    _tool_call(
                        "call_travel_west",
                        "estimate_travel",
                        {"origin": "Depot A", "destination": "West Clinic", "depart_window": "morning"},
                    ),
                    _tool_call("call_weather_denver", "get_weather", {"city": "Denver", "day": "tomorrow"}),
                ],
            ),
            appended_messages=(
                _tool_message(
                    "call_travel_west",
                    "estimate_travel",
                    {"route": "Depot A->West Clinic", "depart_window": "morning", "minutes": 32, "risk": "low"},
                ),
                _tool_message(
                    "call_weather_denver",
                    "get_weather",
                    {"city": "Denver", "day": "tomorrow", "condition": "clear", "high_c": 24, "wind_kph": 18},
                ),
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "Now I will check the afternoon alternative and the other destination weather.",
                [
                    _tool_call(
                        "call_travel_east",
                        "estimate_travel",
                        {"origin": "Depot B", "destination": "East Clinic", "depart_window": "afternoon"},
                    ),
                    _tool_call("call_weather_boulder", "get_weather", {"city": "Boulder", "day": "tomorrow"}),
                ],
            ),
            appended_messages=(
                _tool_message(
                    "call_travel_east",
                    "estimate_travel",
                    {"route": "Depot B->East Clinic", "depart_window": "afternoon", "minutes": 28, "risk": "low"},
                ),
                _tool_message(
                    "call_weather_boulder",
                    "get_weather",
                    {"city": "Boulder", "day": "tomorrow", "condition": "storms", "high_c": 21, "wind_kph": 31},
                ),
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "Use Depot A to West Clinic in the morning. It is slightly longer than the best afternoon estimate, "
                "but the weather is clear and the travel risk is low; avoid the Boulder storm window."
            )
        ),
    ),
)


# system + user
# -> assistant(tool_call: reserve_room, invalid arguments)
# -> user(retry instruction)
# -> assistant(tool_call: reserve_room, corrected arguments)
# -> tool(reserve_room)
# -> assistant
MULTITURN_RETRY_USER = ToolAgentTrajectory(
    name="multiturnretryuser",
    description="A failed tool call is followed by a user retry instruction, then the assistant retries correctly.",
    raw_prompt=(
        {"role": "system", "content": "You reserve rooms. If a request is incomplete, recover with one concise retry."},
        {"role": "user", "content": "Reserve Maple Room tomorrow from 10:00 to 10:30 and confirm the booking."},
    ),
    tools=TOOLS,
    steps=(
        TrajectoryStep(
            assistant=_assistant(
                "I will try to reserve the room now.",
                [_tool_call("call_room_incomplete", "reserve_room", {"room": "Maple Room", "start_time": "10:00"})],
            ),
            appended_messages=(
                {
                    "role": "user",
                    "content": (
                        "The room reservation tool rejected the previous call because end_time was missing. "
                        "Retry with room=Maple Room, start_time=10:00, end_time=10:30."
                    ),
                },
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "I missed the end time. I will retry with the complete time range.",
                [
                    _tool_call(
                        "call_room_retry",
                        "reserve_room",
                        {"room": "Maple Room", "start_time": "10:00", "end_time": "10:30"},
                    )
                ],
            ),
            appended_messages=(
                _tool_message(
                    "call_room_retry",
                    "reserve_room",
                    {
                        "ok": True,
                        "room": "Maple Room",
                        "start_time": "10:00",
                        "end_time": "10:30",
                        "confirmation": "room-4821",
                    },
                ),
            ),
        ),
        TrajectoryStep(assistant=_assistant("Maple Room is reserved from 10:00 to 10:30. Confirmation: room-4821.")),
    ),
)


# system + user
# -> assistant(tool_call: lookup_incident, invalid arguments)
# -> system(retry directive)
# -> assistant(tool_call: lookup_incident, corrected arguments)
# -> tool(lookup_incident)
# -> assistant
MULTITURN_RETRY_SYSTEM = ToolAgentTrajectory(
    name="multiturnretrysystem",
    description="A failed tool call is followed by a system retry directive, then the assistant retries correctly.",
    raw_prompt=(
        {"role": "system", "content": "You are an incident assistant. Respect system retry directives exactly."},
        {"role": "user", "content": "Check the mitigation status for incident 1042 and summarize the owner."},
    ),
    tools=TOOLS,
    steps=(
        TrajectoryStep(
            assistant=_assistant(
                "I will look up the incident status.",
                [_tool_call("call_incident_bad", "lookup_incident", {"incident_id": "1042"})],
            ),
            appended_messages=(
                {
                    "role": "system",
                    "content": (
                        "Tool validation failed: incident_id must use the INC-#### format. "
                        "Retry with incident_id=INC-1042 before answering."
                    ),
                },
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "I will retry with the required incident id format.",
                [_tool_call("call_incident_retry", "lookup_incident", {"incident_id": "INC-1042"})],
            ),
            appended_messages=(
                _tool_message(
                    "call_incident_retry",
                    "lookup_incident",
                    {"ok": True, "incident_id": "INC-1042", "status": "mitigated", "owner": "edge-oncall"},
                ),
            ),
        ),
        TrajectoryStep(assistant=_assistant("Incident INC-1042 is mitigated, and the current owner is edge-oncall.")),
    ),
)


SINGLE_TURN_TRAJECTORIES: tuple[SingleTurnTrajectory, ...] = (SINGLE_TURN_CHAT,)


TOOL_AGENT_TRAJECTORIES: tuple[ToolAgentTrajectory, ...] = (
    MULTITURN_SINGLE_TOOL,
    MULTITURN_MULTI_TOOL,
    MULTITURN_RETRY_USER,
    MULTITURN_RETRY_SYSTEM,
)


TRAJECTORIES: tuple[MockTrajectory, ...] = SINGLE_TURN_TRAJECTORIES + TOOL_AGENT_TRAJECTORIES


TRAJECTORY_BY_NAME: dict[str, MockTrajectory] = {trajectory.name: trajectory for trajectory in TRAJECTORIES}
TOOL_AGENT_TRAJECTORY_BY_NAME = {trajectory.name: trajectory for trajectory in TOOL_AGENT_TRAJECTORIES}


def get_trajectory(name: str) -> MockTrajectory:
    try:
        return TRAJECTORY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown trajectory {name!r}. Choices: {sorted(TRAJECTORY_BY_NAME)}") from exc


def select_trajectories(names: list[str] | None = None) -> list[MockTrajectory]:
    if not names:
        return list(TRAJECTORIES)
    return [get_trajectory(name) for name in names]


def get_tool_agent_trajectory(name: str) -> ToolAgentTrajectory:
    try:
        return TOOL_AGENT_TRAJECTORY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tool-agent trajectory {name!r}. Choices: {sorted(TOOL_AGENT_TRAJECTORY_BY_NAME)}"
        ) from exc


def select_tool_agent_trajectories(names: list[str] | None = None) -> list[ToolAgentTrajectory]:
    if not names:
        return list(TOOL_AGENT_TRAJECTORIES)
    return [get_tool_agent_trajectory(name) for name in names]

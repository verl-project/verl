from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

# Define ChatCompletionContentPartParam locally to avoid openai dependency
ChatCompletionContentPartParam = Dict[str, Any]

Content = Union[str, List[ChatCompletionContentPartParam]]
Item = Any
number = int | float
UUID = str


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: Content
    reward: Optional[float]


class AgentStep(TypedDict, total=False):
    """Represents a single step in an agent's history.

    Attributes:
        step: The step number.
        messages: A list of messages exchanged during the step.
        reward: The reward received at this step.
    """

    step: int
    messages: List[Message]
    reward: float


# AgentHistory maps agent ids (e.g. "Player 1", "Player 2") to their respective list of steps.
AgentHistory = Dict[str, List[AgentStep]]


class Observation(TypedDict):
    """Represents an observation in a game history.

    Attributes:
        raw: The raw observation data (as a dictionary).
        rendered: The rendered string of the observation suitable for input into an LLM.
    """

    raw: Dict[str, Any]
    rendered: Content


class GameStep(TypedDict):
    """Represents a single step in a game history. Essentially an (s,a,r) triple with metadata.

    Attributes:
        step: The step number.
        agent: The agent who took the action (optional for final steps).
        observation: The observation at this step.
        action: The action taken by the agent (if any).
        reward: The reward received; can be a float or a dictionary mapping agent names to rewards.
        done: A flag indicating whether the game has ended after this step.
        info: Additional information related to the step.
    """

    step: int
    agent_id: str
    observation: Observation
    action: str
    reward: float | Dict[str, float]
    done: bool
    info: Dict[str, Any]


# GameHistory is represented as a list of game steps.
GameHistory = List[GameStep]


class WeightedSFTBatch(TypedDict):
    """Represents a batch for weighted SFT training.

    Attributes:
        tokens: Token sequences for the batch.
        loss_masks: Masks indicating which tokens to include in loss computation.
        advantages: Token-level advantages for weighting the loss.
        labels: Target labels for cross-entropy computation (optional, can be derived from tokens).
    """
    tokens: List[List[int]]
    loss_masks: List[List[int]]
    advantages: List[List[float]]
    labels: Optional[List[List[int]]]


class WeightedSFTConfig(TypedDict):
    """Configuration for weighted SFT training.

    Attributes:
        loss_reduction: How to reduce the loss ('mean', 'sum', 'none').
        ignore_index: Token index to ignore in loss computation (typically -100).
        advantage_normalization: Whether to normalize advantages ('batch', 'sequence', 'none').
        temperature: Temperature for softmax (default 1.0).
    """
    loss_reduction: Literal["mean", "sum", "none"]
    ignore_index: int
    advantage_normalization: Literal["batch", "sequence", "none"]
    temperature: float

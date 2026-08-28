from typing import Any, Dict, Optional

from pydantic import BaseModel


class ReActStep(BaseModel):
    """
    Single step in the ReAct reasoning trace.

    This model is used to record each thought-action-observation cycle
    in the ReAct loop. The trace is stored in the ChatCompletionResponse's
    Choice.provider_specific_fields["react_trace"] as a list of ReActStep dicts.
    """

    thought: Optional[str] = None
    """The reasoning/thought from the LLM before taking action."""

    action: str
    """The name of the tool that was called."""

    action_input: Dict[str, Any]
    """The input arguments passed to the tool."""

    observation: str
    """The result returned from the tool execution."""

    outcome: Optional[str] = None
    """Interpreted outcome disposition when the tool exposes one."""

    error_code: Optional[str] = None
    """Stable interpreted error code, when present."""

    attempt: int = 1
    """One-based attempt number for this tool and outcome class."""

    executed: bool = True
    """False for virtual control calls or when safe handling suppresses execution."""

    model_round: int = 1
    """One-based model round that emitted this tool call."""

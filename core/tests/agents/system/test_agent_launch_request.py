from pydantic import ValidationError
import pytest

from rustic_ai.core.agents.system.models import AgentLaunchRequest
from rustic_ai.core.guild import AgentSpec


def agent_spec() -> AgentSpec:
    return AgentSpec(
        id="dynamic",
        name="Dynamic",
        description="Dynamic agent",
        class_name="example.DynamicAgent",
        properties={},
    )


def test_agent_launch_request_accepts_dependency_selections():
    request = AgentLaunchRequest(
        agent_spec=agent_spec(),
        dependency_selections={"llm": {"catalog_key": "dynamic_models", "selector": "gpt"}},
    )

    assert request.dependency_selections["llm"].selector == "gpt"


def test_agent_launch_request_keeps_legacy_shape():
    assert AgentLaunchRequest(agent_spec=agent_spec()).dependency_selections == {}


def test_agent_launch_request_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentLaunchRequest(agent_spec=agent_spec(), dependency_selection={})

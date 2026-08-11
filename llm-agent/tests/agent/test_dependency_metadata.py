from rustic_ai.core.guild.metaprog.agent_registry import AgentRegistry
from rustic_ai.llm_agent.llm_agent import LLMAgent


def test_llm_agent_publishes_canonical_llm_dependency_type():
    entry = AgentRegistry.get_agent(
        f"{LLMAgent.__module__}.{LLMAgent.__name__}"
    )
    assert entry is not None

    llm = next(
        dependency
        for dependency in entry.agent_dependencies
        if dependency.dependency_key == "llm"
    )
    assert (
        llm.required_type
        == "rustic_ai.core.guild.agent_ext.depends.llm.llm.LLM"
    )


def test_llm_agent_model_property_is_optional():
    entry = AgentRegistry.get_agent(
        f"{LLMAgent.__module__}.{LLMAgent.__name__}"
    )
    assert entry is not None
    assert "model" not in entry.agent_props_schema.get("required", [])

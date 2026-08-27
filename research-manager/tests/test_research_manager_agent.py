"""Tests for the research manager's public agent classes."""

from rustic_ai.core.utils.basic_class_utils import get_qualified_class_name
from rustic_ai.research_manager.agent import ResearchManager, ResearchUpdates


def test_research_manager_has_standalone_class_names():
    assert get_qualified_class_name(ResearchManager) == "rustic_ai.research_manager.agent.ResearchManager"
    assert get_qualified_class_name(ResearchUpdates) == "rustic_ai.research_manager.agent.ResearchUpdates"

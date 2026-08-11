from typing import get_args, get_origin

from rustic_ai.core.guild.agent_ext.depends import DependencyResolver
from rustic_ai.core.guild.agent_ext.depends.llm import LLM
from rustic_ai.litellm.agent_ext.llm import LiteLLMResolver


def test_litellm_resolver_declares_llm_contract():
    for base in LiteLLMResolver.__orig_bases__:
        if get_origin(base) is DependencyResolver:
            assert get_args(base)[0] is LLM
            return
    raise AssertionError("LiteLLMResolver does not declare DependencyResolver[T]")

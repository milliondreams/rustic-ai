from typing import get_args, get_origin

from rustic_ai.chroma.agent_ext.vectorstore import ChromaResolver
from rustic_ai.core.guild.agent_ext.depends import DependencyResolver
from rustic_ai.core.guild.agent_ext.depends.vectorstore import VectorStore


def test_chroma_resolver_declares_vectorstore_contract():
    for base in ChromaResolver.__orig_bases__:
        if get_origin(base) is DependencyResolver:
            assert get_args(base)[0] is VectorStore
            return
    raise AssertionError("ChromaResolver does not declare DependencyResolver[T]")

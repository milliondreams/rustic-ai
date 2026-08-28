from typing import get_args, get_origin
from unittest.mock import Mock

from rustic_ai.core.guild.agent_ext.depends import DependencyResolver
from rustic_ai.core.guild.agent_ext.depends.embeddings import Embeddings
from rustic_ai.core.guild.agent_ext.depends.text_splitter import TextSplitter
from rustic_ai.langchain.agent_ext.embeddings import openai
from rustic_ai.langchain.agent_ext.embeddings.openai import OpenAIEmbeddingsResolver
from rustic_ai.langchain.agent_ext.text_splitter.recursive_splitter import (
    RecursiveSplitterResolver,
)


def _provided_type(resolver: type[DependencyResolver]) -> type:
    for base in resolver.__orig_bases__:
        if get_origin(base) is DependencyResolver:
            return get_args(base)[0]
    raise AssertionError(f"{resolver.__name__} does not declare DependencyResolver[T]")


def test_langchain_resolvers_declare_interface_contracts():
    assert _provided_type(OpenAIEmbeddingsResolver) is Embeddings
    assert _provided_type(RecursiveSplitterResolver) is TextSplitter


def test_openai_embeddings_resolver_maps_local_compatible_configuration(monkeypatch):
    client = Mock()
    client.embed_documents.return_value = [[1.0, 0.0], [0.0, 1.0]]
    constructor = Mock(return_value=client)
    monkeypatch.setattr(openai, "LangchainOAIE", constructor)

    resolver = OpenAIEmbeddingsResolver(
        model_name="rustic/nomic-embed-default",
        model_conf={
            "openai_api_base": "http://localhost:55262/v1",
            "openai_api_key": "local-not-required",
        },
    )

    assert resolver.resolve("org", "guild", "agent").embed(["one", "two"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    constructor.assert_called_once_with(
        model="rustic/nomic-embed-default",
        deployment="rustic/nomic-embed-default",
        openai_api_base="http://localhost:55262/v1",
        openai_api_key="local-not-required",
    )

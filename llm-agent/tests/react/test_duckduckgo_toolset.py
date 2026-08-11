import json
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from pydantic import ValidationError
import pytest

from rustic_ai.llm_agent.react import (
    DuckDuckGoInstantAnswerArgs,
    DuckDuckGoInstantAnswerToolset,
    ReActAgentConfig,
)


class FakeResponse:
    def __init__(self, payload):
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def read(self, limit):
        return self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def execute_with_payload(payload, *, query="Python", max_related_topics=5):
    opener = Mock()
    opener.open.return_value = FakeResponse(payload)
    toolset = DuckDuckGoInstantAnswerToolset()
    args = DuckDuckGoInstantAnswerArgs(query=query, max_related_topics=max_related_topics)
    with patch("rustic_ai.llm_agent.react.toolsets.duckduckgo.build_opener", return_value=opener):
        result = json.loads(toolset.execute("duckduckgo_instant_answer", args))
    return result, opener


def test_tool_spec_exposes_typed_query_schema():
    toolset = DuckDuckGoInstantAnswerToolset()

    [tool] = toolset.chat_tools
    parameters = tool.function.parameters.model_dump()

    assert tool.function.name == "duckduckgo_instant_answer"
    assert parameters["required"] == ["query"]
    assert parameters["properties"]["query"]["maxLength"] == 500
    assert parameters["properties"]["max_related_topics"]["maximum"] == 10


def test_toolset_serializes_and_deserializes_in_react_config():
    config = ReActAgentConfig(
        toolset={
            "kind": "rustic_ai.llm_agent.react.toolsets.duckduckgo.DuckDuckGoInstantAnswerToolset",
            "timeout_seconds": 4,
        }
    )

    assert isinstance(config.toolset, DuckDuckGoInstantAnswerToolset)
    assert config.toolset.timeout_seconds == 4
    assert config.model_dump()["toolset"]["kind"].endswith("DuckDuckGoInstantAnswerToolset")


def test_query_is_trimmed_and_blank_queries_are_rejected():
    assert DuckDuckGoInstantAnswerArgs(query="  Python  ").query == "Python"

    with pytest.raises(ValidationError):
        DuckDuckGoInstantAnswerArgs(query="   ")


def test_executes_fixed_api_request_with_expected_parameters():
    result, opener = execute_with_payload({"Answer": "42", "AnswerType": "calc"}, query="6 * 7")

    request = opener.open.call_args.args[0]
    params = parse_qs(urlsplit(request.full_url).query)
    assert request.full_url.startswith("https://api.duckduckgo.com/?")
    assert params == {
        "q": ["6 * 7"],
        "format": ["json"],
        "no_html": ["1"],
        "no_redirect": ["1"],
        "skip_disambig": ["0"],
        "t": ["rustic-ai"],
    }
    assert opener.open.call_args.kwargs == {"timeout": 10.0}
    assert result["answer"] == {"text": "42", "type": "calc"}


def test_formats_abstract_and_rejects_unsafe_source_url():
    result, _ = execute_with_payload(
        {
            "Heading": "Python",
            "AbstractText": "A programming language.",
            "AbstractSource": "Wikipedia",
            "AbstractURL": "javascript:alert(1)",
        }
    )

    assert result["status"] == "ok"
    assert result["heading"] == "Python"
    assert result["abstract"] == {
        "text": "A programming language.",
        "source": "Wikipedia",
        "url": "",
    }


def test_flattens_limits_and_deduplicates_related_topics():
    topic = {"Text": "Python language", "FirstURL": "https://duckduckgo.com/Python"}
    result, _ = execute_with_payload(
        {
            "RelatedTopics": [
                {"Name": "Languages", "Topics": [topic, topic]},
                {"Text": "Monty Python", "FirstURL": "https://duckduckgo.com/Monty_Python"},
            ]
        },
        max_related_topics=1,
    )

    assert result["related_topics"] == [{"text": "Python language", "url": "https://duckduckgo.com/Python"}]


def test_zero_related_topic_limit_omits_topics():
    result, _ = execute_with_payload(
        {"RelatedTopics": [{"Text": "Python language", "FirstURL": "https://duckduckgo.com/Python"}]},
        max_related_topics=0,
    )

    assert result["status"] == "no_result"
    assert "related_topics" not in result


def test_returns_explicit_no_result():
    result, _ = execute_with_payload({"RelatedTopics": []}, query="unanswerable long-tail query")

    assert result == {
        "query": "unanswerable long-tail query",
        "status": "no_result",
        "notice": "DuckDuckGo Instant Answers is not full web search and may not cover this query.",
        "retry_hint": "Retry once with a concise entity or topic name, not a question.",
    }


def test_interprets_success_no_result_and_network_failure():
    toolset = DuckDuckGoInstantAnswerToolset()
    args = DuckDuckGoInstantAnswerArgs(query="Python")

    success = toolset.interpret_result(
        "duckduckgo_instant_answer",
        args,
        json.dumps({"status": "ok", "abstract": {"text": "A language.", "source": "Wikipedia"}}),
    )
    no_result = toolset.interpret_result(
        "duckduckgo_instant_answer",
        args,
        json.dumps({"status": "no_result"}),
    )
    network_error = toolset.interpret_result(
        "duckduckgo_instant_answer",
        args,
        json.dumps({"status": "error", "error": {"code": "network_error", "message": "Unavailable"}}),
    )

    assert success is not None and success.disposition == "success"
    assert success.verified_summary == "A language. (source: Wikipedia)"
    assert no_result is not None and no_result.disposition == "no_result"
    assert network_error is not None and network_error.disposition == "retryable_error"


@pytest.mark.parametrize(
    ("side_effect", "expected_code"),
    [
        (URLError("offline"), "network_error"),
        (HTTPError("https://api.duckduckgo.com/", 429, "rate limited", {}, None), "http_error"),
    ],
)
def test_returns_bounded_network_errors(side_effect, expected_code):
    opener = Mock()
    opener.open.side_effect = side_effect
    args = DuckDuckGoInstantAnswerArgs(query="Python")

    with patch("rustic_ai.llm_agent.react.toolsets.duckduckgo.build_opener", return_value=opener):
        result = json.loads(DuckDuckGoInstantAnswerToolset().execute("duckduckgo_instant_answer", args))

    assert result["status"] == "error"
    assert result["error"]["code"] == expected_code
    assert "offline" not in result["error"]["message"]


def test_rejects_oversized_and_malformed_responses():
    oversized, _ = execute_with_payload(b"x" * (DuckDuckGoInstantAnswerToolset.MAX_RESPONSE_BYTES + 1))
    malformed, _ = execute_with_payload(b"not json")

    assert oversized["error"]["code"] == "invalid_response"
    assert malformed["error"]["code"] == "invalid_response"


def test_rejects_unknown_tool_and_wrong_argument_type():
    toolset = DuckDuckGoInstantAnswerToolset()

    with pytest.raises(ValueError, match="Unknown tool"):
        toolset.execute("web_search", DuckDuckGoInstantAnswerArgs(query="Python"))
    with pytest.raises(TypeError, match="requires DuckDuckGoInstantAnswerArgs"):
        toolset.execute("duckduckgo_instant_answer", Mock())

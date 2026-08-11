import json
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from pydantic import ValidationError
import pytest

from rustic_ai.llm_agent.react import (
    CompositeToolset,
    MathToolset,
    MediaWikiSearchArgs,
    MediaWikiSearchToolset,
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


def execute_with_payload(payload, *, query="Python programming language", max_results=3):
    opener = Mock()
    opener.open.return_value = FakeResponse(payload)
    toolset = MediaWikiSearchToolset()
    args = MediaWikiSearchArgs(query=query, max_results=max_results)
    with patch("rustic_ai.llm_agent.react.toolsets.mediawiki.build_opener", return_value=opener):
        result = json.loads(toolset.execute("mediawiki_search", args))
    return result, opener


def page(title, index, extract="An encyclopedia extract.", fullurl=None, *, disambiguation=False):
    value = {
        "pageid": index,
        "index": index,
        "title": title,
        "extract": extract,
        "fullurl": fullurl or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    }
    if disambiguation:
        value["pageprops"] = {"disambiguation": ""}
    return value


def test_tool_spec_exposes_strict_typed_schema():
    [tool] = MediaWikiSearchToolset().chat_tools
    parameters = tool.function.parameters.model_dump()

    assert tool.function.name == "mediawiki_search"
    assert parameters["required"] == ["query"]
    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["query"]["maxLength"] == 300
    assert parameters["properties"]["max_results"]["maximum"] == 5


def test_argument_model_normalizes_query_and_rejects_extras():
    assert MediaWikiSearchArgs(query="  Python   programming  ").query == "Python programming"

    with pytest.raises(ValidationError):
        MediaWikiSearchArgs(query="   ")
    with pytest.raises(ValidationError):
        MediaWikiSearchArgs(query="Python", language="en")


def test_toolset_serializes_deserializes_and_composes():
    config = ReActAgentConfig(
        toolset={
            "kind": "rustic_ai.llm_agent.react.toolsets.mediawiki.MediaWikiSearchToolset",
            "timeout_seconds": 4,
        }
    )
    composite = CompositeToolset(toolsets=[MathToolset(), config.toolset])

    assert isinstance(config.toolset, MediaWikiSearchToolset)
    assert config.toolset.timeout_seconds == 4
    assert set(composite.tool_names) == {"calculate", "convert_units", "mediawiki_search"}


def test_executes_fixed_mediawiki_request_with_expected_parameters():
    result, opener = execute_with_payload({"query": {"pages": [page("Python (programming language)", 1)]}})

    request = opener.open.call_args.args[0]
    params = parse_qs(urlsplit(request.full_url).query)
    assert request.full_url.startswith("https://en.wikipedia.org/w/api.php?")
    assert params == {
        "action": ["query"],
        "generator": ["search"],
        "gsrsearch": ["Python programming language"],
        "gsrnamespace": ["0"],
        "gsrlimit": ["3"],
        "prop": ["extracts|info|pageprops"],
        "exintro": ["1"],
        "explaintext": ["1"],
        "inprop": ["url"],
        "ppprop": ["disambiguation"],
        "redirects": ["1"],
        "format": ["json"],
        "formatversion": ["2"],
    }
    assert request.headers["Accept"] == "application/json"
    assert "RusticAI-ReAct" in request.headers["User-agent"]
    assert opener.open.call_args.kwargs == {"timeout": 10.0}
    assert result["status"] == "ok"


def test_orders_limits_and_bounds_results():
    long_extract = "word " * 1_000
    result, _ = execute_with_payload(
        {
            "query": {
                "pages": [
                    page("Third", 3),
                    page("First", 1, extract=long_extract),
                    page("Second", 2),
                ]
            }
        },
        max_results=2,
    )

    assert [item["title"] for item in result["results"]] == ["First", "Second"]
    assert len(result["results"][0]["extract"]) == MediaWikiSearchToolset.MAX_EXTRACT_CHARS
    assert result["source"] == "English Wikipedia"


def test_rejects_untrusted_article_urls():
    result, _ = execute_with_payload(
        {
            "query": {
                "pages": [
                    page("Unsafe", 1, fullurl="https://evil.example/wiki/Unsafe"),
                    page("Credentials", 2, fullurl="https://user:pass@en.wikipedia.org/wiki/Credentials"),
                ]
            }
        }
    )

    assert [item["url"] for item in result["results"]] == ["", ""]


def test_returns_explicit_no_result_for_missing_or_empty_pages():
    missing, _ = execute_with_payload({"batchcomplete": True}, query="Asterwick")
    empty, _ = execute_with_payload({"query": {"pages": []}}, query="Asterwick")

    assert (
        missing
        == empty
        == {
            "query": "Asterwick",
            "status": "no_result",
            "notice": "English Wikipedia returned no matching article.",
            "retry_hint": "Retry once with a concise article title or different identifying keywords.",
        }
    )


def test_marks_top_disambiguation_result_as_ambiguous():
    result, _ = execute_with_payload(
        {
            "query": {
                "pages": [
                    page("Mercury", 1, disambiguation=True),
                    page("Mercury (planet)", 2),
                    page("Mercury (element)", 3),
                ]
            }
        },
        query="Mercury",
    )

    assert result["status"] == "ambiguous"
    assert result["results"][0]["disambiguation"] is True
    assert "ask which meaning" in result["notice"]


def test_interprets_success_no_result_ambiguity_and_errors():
    toolset = MediaWikiSearchToolset()
    args = MediaWikiSearchArgs(query="Python")
    success = toolset.interpret_result(
        "mediawiki_search",
        args,
        json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "Python",
                        "extract": "A programming language.",
                        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                    }
                ],
            }
        ),
    )
    no_result = toolset.interpret_result("mediawiki_search", args, json.dumps({"status": "no_result"}))
    ambiguous = toolset.interpret_result(
        "mediawiki_search",
        args,
        json.dumps({"status": "ambiguous", "results": [{"title": "Python"}, {"title": "Pythonidae"}]}),
    )
    retryable = toolset.interpret_result(
        "mediawiki_search",
        args,
        json.dumps({"status": "error", "error": {"code": "rate_limited", "message": "Slow down"}}),
    )
    terminal = toolset.interpret_result(
        "mediawiki_search",
        args,
        json.dumps({"status": "error", "error": {"code": "invalid_response", "message": "Invalid"}}),
    )

    assert success is not None and success.disposition == "success"
    assert success.verified_summary == (
        "Python: A programming language. "
        "(source: English Wikipedia, https://en.wikipedia.org/wiki/Python_(programming_language))"
    )
    assert no_result is not None and no_result.disposition == "no_result"
    assert ambiguous is not None and ambiguous.disposition == "clarification_required"
    assert ambiguous.code == "ambiguous_topic"
    assert "Pythonidae" in ambiguous.message
    assert retryable is not None and retryable.disposition == "retryable_error"
    assert terminal is not None and terminal.disposition == "terminal_error"


@pytest.mark.parametrize(
    ("side_effect", "expected_code"),
    [
        (URLError("offline"), "network_error"),
        (HTTPError(MediaWikiSearchToolset.API_URL, 429, "limited", {}, None), "rate_limited"),
        (HTTPError(MediaWikiSearchToolset.API_URL, 503, "down", {}, None), "server_error"),
        (HTTPError(MediaWikiSearchToolset.API_URL, 403, "forbidden", {}, None), "http_error"),
    ],
)
def test_returns_bounded_network_errors(side_effect, expected_code):
    opener = Mock()
    opener.open.side_effect = side_effect
    args = MediaWikiSearchArgs(query="Python")

    with patch("rustic_ai.llm_agent.react.toolsets.mediawiki.build_opener", return_value=opener):
        result = json.loads(MediaWikiSearchToolset().execute("mediawiki_search", args))

    assert result["status"] == "error"
    assert result["error"]["code"] == expected_code
    assert "offline" not in result["error"]["message"]


def test_rejects_oversized_malformed_and_structurally_invalid_responses():
    oversized, _ = execute_with_payload(b"x" * (MediaWikiSearchToolset.MAX_RESPONSE_BYTES + 1))
    malformed, _ = execute_with_payload(b"not json")
    invalid, _ = execute_with_payload({"query": {"pages": {"1": page("Python", 1)}}})

    assert oversized["error"]["code"] == "invalid_response"
    assert malformed["error"]["code"] == "invalid_response"
    assert invalid["error"]["code"] == "invalid_response"


def test_rejects_unknown_tool_and_wrong_argument_type():
    toolset = MediaWikiSearchToolset()

    with pytest.raises(ValueError, match="Unknown tool"):
        toolset.execute("web_search", MediaWikiSearchArgs(query="Python"))
    with pytest.raises(TypeError, match="requires MediaWikiSearchArgs"):
        toolset.execute("mediawiki_search", Mock())

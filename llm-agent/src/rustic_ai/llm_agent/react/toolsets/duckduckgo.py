"""DuckDuckGo Instant Answer toolset for ReAct agents."""

from collections.abc import Iterator
import json
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, Field, field_validator

from rustic_ai.core.guild.agent_ext.depends.llm.tools_manager import ToolSpec
from rustic_ai.llm_agent.react.toolset import (
    ReActSkillSpec,
    ReActToolOutcome,
    ReActToolset,
    ToolOutcomeDisposition,
)


class DuckDuckGoInstantAnswerArgs(BaseModel):
    """Arguments accepted by the DuckDuckGo Instant Answer tool."""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="A concise factual query to look up using DuckDuckGo Instant Answers.",
    )
    max_related_topics: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Maximum number of related topics to include in the result.",
    )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be blank")
        return query


class _RejectRedirects(HTTPRedirectHandler):
    """Prevent the fixed API endpoint from redirecting requests elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class DuckDuckGoInstantAnswerToolset(ReActToolset):
    """Look up bounded, structured results from DuckDuckGo Instant Answers."""

    API_URL: ClassVar[str] = "https://api.duckduckgo.com/"
    MAX_RESPONSE_BYTES: ClassVar[int] = 1_048_576
    MAX_ABSTRACT_CHARS: ClassVar[int] = 4_000
    MAX_ANSWER_CHARS: ClassVar[int] = 2_000
    MAX_TOPIC_CHARS: ClassVar[int] = 1_000

    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
        description="Timeout for the DuckDuckGo request.",
    )

    def get_skill_specs(self) -> list[ReActSkillSpec]:
        return [
            ReActSkillSpec(
                name="knowledge_lookup",
                description="Stable encyclopedia lookup or concise Instant Answer lookup.",
                tool_names=["duckduckgo_instant_answer"],
                instructions="Use only when Instant Answers is requested or appropriate; it is not general web search.",
                examples=["France", "gold chemical element"],
                order=31,
            )
        ]

    def get_toolspecs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="duckduckgo_instant_answer",
                description=(
                    "Look up a concise entity or topic using DuckDuckGo Instant Answers. This is not general web "
                    "search and may return no result for broad, current, or long-tail queries."
                ),
                parameter_class=DuckDuckGoInstantAnswerArgs,
            )
        ]

    def execute(self, tool_name: str, args: BaseModel) -> str:
        if tool_name != "duckduckgo_instant_answer":
            raise ValueError(f"Unknown tool: {tool_name}")
        if not isinstance(args, DuckDuckGoInstantAnswerArgs):
            raise TypeError("duckduckgo_instant_answer requires DuckDuckGoInstantAnswerArgs")

        try:
            payload = self._request(args.query)
            result = self._format_result(payload, args)
        except HTTPError as exc:
            result = self._error_result(args.query, "http_error", f"DuckDuckGo returned HTTP {exc.code}")
        except (TimeoutError, URLError):
            result = self._error_result(args.query, "network_error", "DuckDuckGo could not be reached")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            result = self._error_result(args.query, "invalid_response", str(exc))

        return json.dumps(result, ensure_ascii=True, separators=(",", ":"))

    def interpret_result(self, tool_name: str, args: BaseModel, result: str) -> ReActToolOutcome | None:
        if tool_name != "duckduckgo_instant_answer":
            return None
        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        status = payload.get("status")
        if status == "ok":
            return ReActToolOutcome(
                disposition=ToolOutcomeDisposition.SUCCESS,
                verified_summary=self._verified_summary(payload),
            )
        if status == "no_result":
            return ReActToolOutcome(
                disposition=ToolOutcomeDisposition.NO_RESULT,
                code="no_result",
                message=(
                    "DuckDuckGo Instant Answers returned no result. Retry once with a concise entity/topic query; "
                    "do not answer from memory if verification still fails."
                ),
            )
        error = payload.get("error")
        if not isinstance(error, dict):
            return None
        code = error.get("code")
        message = error.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            return None
        disposition = (
            ToolOutcomeDisposition.RETRYABLE_ERROR if code == "network_error" else ToolOutcomeDisposition.TERMINAL_ERROR
        )
        return ReActToolOutcome(disposition=disposition, code=code, message=message)

    @staticmethod
    def _verified_summary(payload: dict[str, Any]) -> str:
        for key in ("answer", "abstract", "definition"):
            value = payload.get(key)
            if isinstance(value, dict) and isinstance(value.get("text"), str):
                source = value.get("source")
                return f"{value['text']} (source: {source})" if source else value["text"]
        topics = payload.get("related_topics")
        if isinstance(topics, list) and topics and isinstance(topics[0], dict):
            text = topics[0].get("text")
            if isinstance(text, str):
                return text
        return "DuckDuckGo returned a verified Instant Answer result."

    def _request(self, query: str) -> dict[str, Any]:
        params = urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": "1",
                "no_redirect": "1",
                "skip_disambig": "0",
                "t": "rustic-ai",
            }
        )
        request = Request(
            f"{self.API_URL}?{params}",
            headers={
                "Accept": "application/json",
                "User-Agent": "RusticAI-ReAct/1.0",
            },
            method="GET",
        )

        opener = build_opener(_RejectRedirects())
        with opener.open(request, timeout=self.timeout_seconds) as response:
            body = response.read(self.MAX_RESPONSE_BYTES + 1)

        if len(body) > self.MAX_RESPONSE_BYTES:
            raise ValueError("DuckDuckGo response exceeded the size limit")

        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("DuckDuckGo returned a non-object response")
        return payload

    def _format_result(
        self,
        payload: dict[str, Any],
        args: DuckDuckGoInstantAnswerArgs,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "query": args.query,
            "status": "no_result",
            "notice": "DuckDuckGo Instant Answers is not full web search and may not cover this query.",
            "retry_hint": "Retry once with a concise entity or topic name, not a question.",
        }

        heading = self._bounded_text(payload.get("Heading"), 300)
        if heading:
            result["heading"] = heading

        answer = self._bounded_text(payload.get("Answer"), self.MAX_ANSWER_CHARS)
        if answer:
            result["answer"] = {
                "text": answer,
                "type": self._bounded_text(payload.get("AnswerType"), 100),
            }

        abstract = self._bounded_text(
            payload.get("AbstractText") or payload.get("Abstract"),
            self.MAX_ABSTRACT_CHARS,
        )
        if abstract:
            result["abstract"] = {
                "text": abstract,
                "source": self._bounded_text(payload.get("AbstractSource"), 200),
                "url": self._safe_url(payload.get("AbstractURL")),
            }

        definition = self._bounded_text(payload.get("Definition"), self.MAX_ANSWER_CHARS)
        if definition:
            result["definition"] = {
                "text": definition,
                "source": self._bounded_text(payload.get("DefinitionSource"), 200),
                "url": self._safe_url(payload.get("DefinitionURL")),
            }

        related_topics = self._related_topics(payload.get("RelatedTopics"), args.max_related_topics)
        if related_topics:
            result["related_topics"] = related_topics

        if any(key in result for key in ("answer", "abstract", "definition", "related_topics")):
            result["status"] = "ok"

        return result

    def _related_topics(self, value: Any, limit: int) -> list[dict[str, str]]:
        if limit == 0:
            return []

        topics: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for item in self._iter_topic_items(value):
            text = self._bounded_text(item.get("Text"), self.MAX_TOPIC_CHARS)
            url = self._safe_url(item.get("FirstURL"))
            if not text:
                continue

            identity = (text, url)
            if identity in seen:
                continue
            seen.add(identity)
            topics.append({"text": text, "url": url})
            if len(topics) >= limit:
                break

        return topics

    @classmethod
    def _iter_topic_items(cls, value: Any) -> Iterator[dict[str, Any]]:
        if not isinstance(value, list):
            return
        for item in value:
            if not isinstance(item, dict):
                continue
            nested = item.get("Topics")
            if isinstance(nested, list):
                yield from cls._iter_topic_items(nested)
            else:
                yield item

    @staticmethod
    def _bounded_text(value: Any, limit: int) -> str:
        if not isinstance(value, str):
            return ""
        text = " ".join(value.split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    @staticmethod
    def _safe_url(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return value

    @staticmethod
    def _error_result(query: str, code: str, message: str) -> dict[str, Any]:
        return {
            "query": query,
            "status": "error",
            "error": {"code": code, "message": message},
        }

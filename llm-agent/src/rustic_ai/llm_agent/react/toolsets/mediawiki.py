"""MediaWiki search toolset for grounded encyclopedic lookups."""

import json
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rustic_ai.core.guild.agent_ext.depends.llm.tools_manager import ToolSpec
from rustic_ai.llm_agent.react.toolset import (
    ReActSkillSpec,
    ReActToolOutcome,
    ReActToolset,
    ToolOutcomeDisposition,
)


class MediaWikiSearchArgs(BaseModel):
    """Arguments accepted by the English Wikipedia search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "A concise encyclopedia topic, title, person, place, work, or keyword query, such as "
            "'Nineteen Eighty-Four novel' or 'Python programming language'."
        ),
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of matching English Wikipedia articles to return.",
    )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = " ".join(value.split())
        if not query:
            raise ValueError("query must not be blank")
        return query


class _RejectRedirects(HTTPRedirectHandler):
    """Prevent the fixed API endpoint from redirecting elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class MediaWikiSearchToolset(ReActToolset):
    """Search English Wikipedia and return bounded introductory extracts."""

    API_URL: ClassVar[str] = "https://en.wikipedia.org/w/api.php"
    ALLOWED_HOST: ClassVar[str] = "en.wikipedia.org"
    MAX_RESPONSE_BYTES: ClassVar[int] = 1_048_576
    MAX_TITLE_CHARS: ClassVar[int] = 300
    MAX_EXTRACT_CHARS: ClassVar[int] = 2_000

    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
        description="Timeout for the MediaWiki request.",
    )

    def get_skill_specs(self) -> list[ReActSkillSpec]:
        return [
            ReActSkillSpec(
                name="knowledge_lookup",
                description="Stable encyclopedia lookup or concise Instant Answer lookup.",
                tool_names=["mediawiki_search"],
                instructions="Use a concise article title or topic, not a long question. This is not current web search.",
                examples=["Python programming language", "Nineteen Eighty-Four novel"],
                order=30,
            )
        ]

    def get_toolspecs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="mediawiki_search",
                description=(
                    "Search English Wikipedia for stable facts about people, places, works, concepts, and history. "
                    "Use a concise title or topic; this is not current or general web search."
                ),
                parameter_class=MediaWikiSearchArgs,
            )
        ]

    def execute(self, tool_name: str, args: BaseModel) -> str:
        if tool_name != "mediawiki_search":
            raise ValueError(f"Unknown tool: {tool_name}")
        if not isinstance(args, MediaWikiSearchArgs):
            raise TypeError("mediawiki_search requires MediaWikiSearchArgs")

        try:
            payload = self._request(args.query, args.max_results)
            result = self._format_result(payload, args)
        except HTTPError as exc:
            if exc.code == 429:
                result = self._error_result(
                    args.query,
                    "rate_limited",
                    "Wikipedia temporarily rate limited the request",
                )
            elif exc.code >= 500:
                result = self._error_result(args.query, "server_error", "Wikipedia is temporarily unavailable")
            else:
                result = self._error_result(args.query, "http_error", f"Wikipedia returned HTTP {exc.code}")
        except (TimeoutError, URLError):
            result = self._error_result(args.query, "network_error", "Wikipedia could not be reached")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            result = self._error_result(args.query, "invalid_response", str(exc))

        return json.dumps(result, ensure_ascii=True, separators=(",", ":"))

    def interpret_result(self, tool_name: str, args: BaseModel, result: str) -> ReActToolOutcome | None:
        if tool_name != "mediawiki_search":
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
        if status == "ambiguous":
            return ReActToolOutcome(
                disposition=ToolOutcomeDisposition.CLARIFICATION_REQUIRED,
                code="ambiguous_topic",
                message=self._ambiguity_message(payload),
            )
        if status == "no_result":
            return ReActToolOutcome(
                disposition=ToolOutcomeDisposition.NO_RESULT,
                code="no_result",
                message="Wikipedia returned no matching article. Retry once with a concise title or different keywords.",
            )

        error = payload.get("error")
        if not isinstance(error, dict):
            return None
        code = error.get("code")
        message = error.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            return None
        retryable = {"network_error", "rate_limited", "server_error"}
        disposition = (
            ToolOutcomeDisposition.RETRYABLE_ERROR if code in retryable else ToolOutcomeDisposition.TERMINAL_ERROR
        )
        return ReActToolOutcome(disposition=disposition, code=code, message=message)

    def _request(self, query: str, max_results: int) -> dict[str, Any]:
        params = urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "0",
                "gsrlimit": str(max_results),
                "prop": "extracts|info|pageprops",
                "exintro": "1",
                "explaintext": "1",
                "inprop": "url",
                "ppprop": "disambiguation",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            }
        )
        request = Request(
            f"{self.API_URL}?{params}",
            headers={
                "Accept": "application/json",
                "User-Agent": "RusticAI-ReAct/1.0 (https://github.com/rustic-ai/rustic-ai)",
            },
            method="GET",
        )

        opener = build_opener(_RejectRedirects())
        with opener.open(request, timeout=self.timeout_seconds) as response:
            body = response.read(self.MAX_RESPONSE_BYTES + 1)
        if len(body) > self.MAX_RESPONSE_BYTES:
            raise ValueError("Wikipedia response exceeded the size limit")

        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Wikipedia returned a non-object response")
        return payload

    def _format_result(self, payload: dict[str, Any], args: MediaWikiSearchArgs) -> dict[str, Any]:
        query_payload = payload.get("query")
        if query_payload is None:
            return self._no_result(args.query)
        if not isinstance(query_payload, dict):
            raise ValueError("Wikipedia response contained an invalid query object")

        pages = query_payload.get("pages")
        if pages is None:
            return self._no_result(args.query)
        if not isinstance(pages, list):
            raise ValueError("Wikipedia response contained an invalid pages list")

        results = [result for page in pages if (result := self._format_page(page)) is not None]
        results.sort(key=lambda item: item.pop("_index"))
        results = results[: args.max_results]
        if not results:
            return self._no_result(args.query)

        status = "ambiguous" if results[0]["disambiguation"] else "ok"
        formatted: dict[str, Any] = {
            "query": args.query,
            "status": status,
            "source": "English Wikipedia",
            "results": results,
        }
        if status == "ambiguous":
            formatted["notice"] = "The top Wikipedia result is a disambiguation page; ask which meaning is intended."
        return formatted

    def _format_page(self, page: Any) -> dict[str, Any] | None:
        if not isinstance(page, dict):
            return None
        title = self._bounded_text(page.get("title"), self.MAX_TITLE_CHARS)
        if not title:
            return None
        index = page.get("index")
        if not isinstance(index, int):
            index = 2**31 - 1
        pageprops = page.get("pageprops")
        disambiguation = isinstance(pageprops, dict) and "disambiguation" in pageprops
        return {
            "_index": index,
            "title": title,
            "extract": self._bounded_text(page.get("extract"), self.MAX_EXTRACT_CHARS),
            "url": self._safe_article_url(page.get("fullurl")),
            "disambiguation": disambiguation,
        }

    @classmethod
    def _safe_article_url(cls, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname != cls.ALLOWED_HOST
            or parsed.username is not None
            or parsed.password is not None
        ):
            return ""
        return value

    @staticmethod
    def _bounded_text(value: Any, limit: int) -> str:
        if not isinstance(value, str):
            return ""
        text = " ".join(value.split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3].rstrip()}..."

    @staticmethod
    def _no_result(query: str) -> dict[str, Any]:
        return {
            "query": query,
            "status": "no_result",
            "notice": "English Wikipedia returned no matching article.",
            "retry_hint": "Retry once with a concise article title or different identifying keywords.",
        }

    @staticmethod
    def _error_result(query: str, code: str, message: str) -> dict[str, Any]:
        return {
            "query": query,
            "status": "error",
            "error": {"code": code, "message": message},
        }

    @staticmethod
    def _verified_summary(payload: dict[str, Any]) -> str:
        results = payload.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            first = results[0]
            title = first.get("title")
            extract = first.get("extract")
            url = first.get("url")
            if isinstance(title, str) and isinstance(extract, str) and extract:
                suffix = f" (source: English Wikipedia, {url})" if isinstance(url, str) and url else ""
                return f"{title}: {extract}{suffix}"
        return "Wikipedia returned a verified encyclopedic result."

    @staticmethod
    def _ambiguity_message(payload: dict[str, Any]) -> str:
        results = payload.get("results")
        if isinstance(results, list):
            titles: list[str] = []
            for item in results:
                if isinstance(item, dict):
                    title = item.get("title")
                    if isinstance(title, str):
                        titles.append(title)
            if titles:
                return f"The topic is ambiguous. Ask the user to choose among: {', '.join(titles)}."
        return "The topic is ambiguous. Ask the user which meaning they intend."

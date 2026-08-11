from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from rustic_ai.core.guild.agent_ext.depends.llm.models import ChatCompletionTool
from rustic_ai.core.guild.agent_ext.depends.llm.tools_manager import ToolSpec
from rustic_ai.core.utils.basic_class_utils import get_class_from_name

if TYPE_CHECKING:
    from rustic_ai.llm_agent.plugins.request_preprocessor import RequestPreprocessor
    from rustic_ai.llm_agent.plugins.tool_call_wrapper import ToolCallWrapper


class ToolOutcomeDisposition(str, Enum):
    """How the ReAct executor should handle an interpreted tool result."""

    SUCCESS = "success"
    RETRYABLE_ERROR = "retryable_error"
    TERMINAL_ERROR = "terminal_error"
    CLARIFICATION_REQUIRED = "clarification_required"
    NO_RESULT = "no_result"


class ReActToolOutcome(BaseModel):
    """Structured interpretation of a tool's string result."""

    disposition: ToolOutcomeDisposition
    code: Optional[str] = None
    message: Optional[str] = None
    verified_summary: Optional[str] = None


class ReActSkillSpec(BaseModel):
    """A progressively disclosed group of related ReAct tools."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    description: str = Field(min_length=1, max_length=500)
    tool_names: List[str] = Field(min_length=1)
    instructions: str = Field(default="", max_length=2_000)
    examples: List[str] = Field(default_factory=list, max_length=8)
    order: int = Field(default=100, ge=0, exclude=True)

    @field_validator("tool_names")
    @classmethod
    def _unique_tool_names(cls, value: List[str]) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("skill tool_names must be unique")
        return value


class ReActToolset(BaseModel, ABC):
    """
    Abstract base class for tool providers with execution capability.

    A ReActToolset defines both the tools available to the agent (via get_toolspecs)
    and the execution logic for those tools (via execute).

    Subclasses must implement:
    - get_toolspecs(): Return list of ToolSpec objects defining available tools
    - execute(): Execute a tool by name with parsed arguments

    The toolset is fully serializable via the `kind` field which stores the
    fully qualified class name (FQCN) for runtime class resolution.

    Built-in implementations such as ``MathToolset`` provide safe tools that can
    be used directly or combined with ``CompositeToolset``.
    """

    kind: Optional[str] = Field(default=None, frozen=True, description="FQCN of the toolset class")

    def model_post_init(self, __context) -> None:
        if not self.kind:
            object.__setattr__(
                self,
                "kind",
                f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            )

    @model_validator(mode="after")
    def _enforce_kind_matches_class(self):
        fqcn = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        if self.kind and self.kind != fqcn:
            raise ValueError(f"`kind` must be {fqcn!r}, got {self.kind!r}")
        return self

    @abstractmethod
    def get_toolspecs(self) -> List[ToolSpec]:
        """
        Return the list of tool specifications available in this toolset.

        Returns:
            List of ToolSpec objects defining the tools.
        """
        pass

    @abstractmethod
    def execute(self, tool_name: str, args: BaseModel) -> str:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: The name of the tool to execute.
            args: The parsed arguments as a Pydantic model instance.

        Returns:
            The result of the tool execution as a string.

        Raises:
            ValueError: If the tool name is not recognized.
        """
        pass

    def bind_agent_context(self, org_id: str, guild_id: str, agent_id: str) -> None:
        """
        Optional hook called by ReActAgent to provide guild context to the toolset.

        Toolsets that need access to guild-scoped resources (like filesystems)
        can override this method to configure themselves with the correct paths.

        Args:
            org_id: The organization ID.
            guild_id: The guild ID.
            agent_id: The agent ID.
        """
        pass

    def validate_plugins(
        self,
        request_preprocessors: List["RequestPreprocessor"],
        tool_wrappers: List["ToolCallWrapper"],
    ) -> None:
        """
        Validate that required plugins are configured for this toolset.

        Override this method in subclasses to enforce plugin dependencies.
        Raise a ValueError with a descriptive message if required plugins
        are missing.

        This method is called by ReActAgent before processing begins,
        allowing early detection of configuration errors.

        Args:
            request_preprocessors: The configured request preprocessors.
            tool_wrappers: The configured tool wrappers.

        Raises:
            ValueError: If required plugins are not configured.
        """
        pass

    def interpret_result(self, tool_name: str, args: BaseModel, result: str) -> Optional[ReActToolOutcome]:
        """Interpret a result for safe retry and failure handling.

        Custom toolsets remain opaque by default. Implementations should only
        return an outcome when they own and understand the result contract.
        """
        return None

    def get_skill_specs(self) -> List[ReActSkillSpec]:
        """Return optional progressive-disclosure metadata for this toolset."""
        return []

    @cached_property
    def toolspecs_by_name(self) -> Dict[str, ToolSpec]:
        """Return a dictionary mapping tool names to their specifications."""
        return {spec.name: spec for spec in self.get_toolspecs()}

    def get_toolspec(self, name: str) -> Optional[ToolSpec]:
        """
        Get a tool specification by name.

        Args:
            name: The name of the tool.

        Returns:
            The ToolSpec if found, None otherwise.
        """
        return self.toolspecs_by_name.get(name)

    @cached_property
    def chat_tools(self) -> List[ChatCompletionTool]:
        """
        Convert tool specifications to ChatCompletionTool format for LLM API.

        Returns:
            List of ChatCompletionTool objects.
        """
        return [spec.chat_tool for spec in self.get_toolspecs()]

    @cached_property
    def tool_names(self) -> List[str]:
        """Return list of available tool names."""
        return [spec.name for spec in self.get_toolspecs()]

    @cached_property
    def tool_count(self) -> int:
        """Return the number of tools in this toolset."""
        return len(self.get_toolspecs())


class CompositeToolset(ReActToolset):
    """
    A toolset that combines multiple toolsets into one.

    This allows composing toolsets from different sources while presenting
    them as a single unified toolset to the ReActAgent.

    Example:
        composite = CompositeToolset(
            toolsets=[
                CalculatorToolset(),
                SearchToolset(api_key="...")
            ]
        )

    YAML serialization example:
        toolset:
          kind: rustic_ai.llm_agent.react.toolset.CompositeToolset
          toolsets:
            - kind: rustic_ai.skills.toolset.SkillToolset
              skill_paths:
                - /path/to/skill
            - kind: rustic_ai.skills.toolset.MarketplaceSkillToolset
              source: anthropic
              skill_names:
                - pdf
    """

    toolsets: List[ReActToolset] = Field(min_length=1, description="List of toolsets to combine")

    @field_validator("toolsets", mode="before")
    @classmethod
    def _deserialize_toolsets(cls, value: Any) -> List[ReActToolset]:
        """
        Deserialize toolsets from YAML/JSON dicts using the 'kind' field.

        This validator handles polymorphic deserialization of nested toolsets.
        Each toolset dict must have a 'kind' field with the FQCN of the toolset class.
        """
        if not isinstance(value, list):
            raise ValueError("toolsets must be a list")

        deserialized = []
        for item in value:
            if isinstance(item, ReActToolset):
                # Already a toolset instance
                deserialized.append(item)
            elif isinstance(item, dict):
                # Dict representation from YAML/JSON
                kind = item.get("kind")
                if not kind:
                    raise ValueError("Each toolset dict must have a 'kind' field with the class FQCN")

                try:
                    toolset_class = get_class_from_name(kind)
                except Exception as e:
                    raise ValueError(f"Failed to load toolset class '{kind}': {e}") from e

                if not issubclass(toolset_class, ReActToolset):
                    raise ValueError(f"Class '{kind}' is not a ReActToolset subclass")

                # Instantiate the toolset with the remaining fields
                try:
                    toolset_instance = toolset_class(**item)
                    deserialized.append(toolset_instance)
                except Exception as e:
                    raise ValueError(f"Failed to instantiate toolset '{kind}': {e}") from e
            else:
                raise ValueError(f"Invalid toolset type: {type(item)}. Must be ReActToolset or dict.")

        return deserialized

    def get_toolspecs(self) -> List[ToolSpec]:
        """Return combined tool specifications from all toolsets."""
        specs: List[ToolSpec] = []
        for toolset in self.toolsets:
            specs.extend(toolset.get_toolspecs())
        return specs

    def get_skill_specs(self) -> List[ReActSkillSpec]:
        """Return child skills, merging explicitly compatible shared groups."""
        contributions: Dict[str, List[ReActSkillSpec]] = {}
        for toolset in self.toolsets:
            owned_tools = set(toolset.tool_names)
            for skill in toolset.get_skill_specs():
                unknown = set(skill.tool_names) - owned_tools
                if unknown:
                    raise ValueError(
                        f"Skill {skill.name!r} references tools not owned by its toolset: {sorted(unknown)}"
                    )
                existing = contributions.setdefault(skill.name, [])
                if existing and existing[0].description != skill.description:
                    raise ValueError(f"ReAct skill {skill.name!r} has conflicting descriptions across child toolsets")
                existing.append(skill)

        skills: List[ReActSkillSpec] = []
        for name, parts in contributions.items():
            ordered = sorted(enumerate(parts), key=lambda item: (item[1].order, item[0]))
            skills.append(
                ReActSkillSpec(
                    name=name,
                    description=parts[0].description,
                    tool_names=[tool for _, part in ordered for tool in part.tool_names],
                    instructions=" ".join(dict.fromkeys(part.instructions for _, part in ordered if part.instructions)),
                    examples=list(dict.fromkeys(example for _, part in ordered for example in part.examples)),
                    order=min(part.order for part in parts),
                )
            )
        return sorted(skills, key=lambda skill: skill.order)

    def execute(self, tool_name: str, args: BaseModel) -> str:
        """
        Execute a tool by delegating to the appropriate child toolset.

        Args:
            tool_name: The name of the tool to execute.
            args: The parsed arguments.

        Returns:
            The result of the tool execution.

        Raises:
            ValueError: If the tool name is not found in any child toolset.
        """
        for toolset in self.toolsets:
            if tool_name in toolset.tool_names:
                return toolset.execute(tool_name, args)
        raise ValueError(f"Unknown tool: {tool_name}")

    def bind_agent_context(self, org_id: str, guild_id: str, agent_id: str) -> None:
        """
        Propagate agent context to all child toolsets.

        Args:
            org_id: The organization ID.
            guild_id: The guild ID.
            agent_id: The agent ID.
        """
        for toolset in self.toolsets:
            toolset.bind_agent_context(org_id, guild_id, agent_id)

    def interpret_result(self, tool_name: str, args: BaseModel, result: str) -> Optional[ReActToolOutcome]:
        """Delegate result interpretation to the toolset that owns the tool."""
        for toolset in self.toolsets:
            if tool_name in toolset.tool_names:
                return toolset.interpret_result(tool_name, args, result)
        return None

    def validate_plugins(
        self,
        request_preprocessors: List["RequestPreprocessor"],
        tool_wrappers: List["ToolCallWrapper"],
    ) -> None:
        """
        Propagate plugin validation to all child toolsets.

        Args:
            request_preprocessors: The configured request preprocessors.
            tool_wrappers: The configured tool wrappers.

        Raises:
            ValueError: If any child toolset's required plugins are not configured.
        """
        for toolset in self.toolsets:
            toolset.validate_plugins(request_preprocessors, tool_wrappers)

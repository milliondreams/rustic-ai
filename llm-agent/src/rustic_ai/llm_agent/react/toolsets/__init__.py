"""Built-in toolsets for the Rustic AI ReAct agent."""

from .duckduckgo import DuckDuckGoInstantAnswerArgs, DuckDuckGoInstantAnswerToolset
from .math import CalculateArgs, ConvertUnitsArgs, MathToolset
from .mediawiki import MediaWikiSearchArgs, MediaWikiSearchToolset
from .temporal import (
    AddBusinessDaysArgs,
    AddCalendarPeriodArgs,
    ConvertDatetimeArgs,
    DateRangeArgs,
    GetCurrentTimeArgs,
    GetDateInfoArgs,
    TemporalToolset,
)

__all__ = [
    "AddBusinessDaysArgs",
    "AddCalendarPeriodArgs",
    "CalculateArgs",
    "ConvertUnitsArgs",
    "ConvertDatetimeArgs",
    "DateRangeArgs",
    "DuckDuckGoInstantAnswerArgs",
    "DuckDuckGoInstantAnswerToolset",
    "GetDateInfoArgs",
    "MathToolset",
    "MediaWikiSearchArgs",
    "MediaWikiSearchToolset",
    "GetCurrentTimeArgs",
    "TemporalToolset",
]

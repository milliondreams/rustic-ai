"""Deterministic calendar and timezone tools for ReAct agents."""

import calendar
from datetime import date, datetime, timedelta, timezone
import json
import re
from typing import Callable, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tzlocal import get_localzone_name

from rustic_ai.core.guild.agent_ext.depends.llm.tools_manager import ToolSpec
from rustic_ai.llm_agent.react.toolset import (
    ReActSkillSpec,
    ReActToolOutcome,
    ReActToolset,
    ToolOutcomeDisposition,
)


class GetCurrentTimeArgs(BaseModel):
    """Arguments for reading the current time."""

    model_config = ConfigDict(extra="forbid")

    timezone: str | None = Field(
        default=None,
        max_length=255,
        description="Optional IANA timezone such as America/Vancouver. Omit it to use the system timezone.",
    )

    @field_validator("timezone")
    @classmethod
    def normalize_timezone(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "timezone")


class ConvertDatetimeArgs(BaseModel):
    """Arguments for converting one instant between timezones."""

    model_config = ConfigDict(extra="forbid")

    datetime: str = Field(
        min_length=16,
        max_length=64,
        description=(
            "ISO datetime such as 2026-08-06T14:30:00, 2026-08-06T14:30:00-07:00, " "or 2026-08-06T21:30:00Z."
        ),
    )
    from_timezone: str | None = Field(
        default=None,
        max_length=255,
        description="IANA timezone for a datetime without an offset. Omit it to use the system timezone.",
    )
    to_timezone: str | None = Field(
        default=None,
        max_length=255,
        description="Target IANA timezone. Omit it to use the system timezone.",
    )

    @field_validator("datetime")
    @classmethod
    def normalize_datetime(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("datetime must not be blank")
        return value

    @field_validator("from_timezone", "to_timezone")
    @classmethod
    def normalize_timezone(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value, "timezone")


class GetDateInfoArgs(BaseModel):
    """Arguments for inspecting a calendar date."""

    model_config = ConfigDict(extra="forbid")

    date: str = Field(min_length=10, max_length=10, description="Calendar date in YYYY-MM-DD format.")

    @field_validator("date")
    @classmethod
    def normalize_date(cls, value: str) -> str:
        return value.strip()


class AddCalendarPeriodArgs(BaseModel):
    """Arguments for adding calendar units to a date."""

    model_config = ConfigDict(extra="forbid")

    date: str = Field(min_length=10, max_length=10, description="Starting date in YYYY-MM-DD format.")
    years: int = Field(default=0, ge=-9_999, le=9_999)
    months: int = Field(default=0, ge=-1_000_000, le=1_000_000)
    weeks: int = Field(default=0, ge=-1_000_000, le=1_000_000)
    days: int = Field(default=0, ge=-1_000_000, le=1_000_000)

    @field_validator("date")
    @classmethod
    def normalize_date(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def require_nonzero_period(self):
        if not any((self.years, self.months, self.weeks, self.days)):
            raise ValueError("at least one calendar period must be nonzero")
        return self


class AddBusinessDaysArgs(BaseModel):
    """Arguments for adding Monday-to-Friday days to a date."""

    model_config = ConfigDict(extra="forbid")

    date: str = Field(min_length=10, max_length=10, description="Starting date in YYYY-MM-DD format.")
    days: int = Field(
        ge=-100_000,
        le=100_000,
        description="Signed Monday-to-Friday days to add; holidays are not included.",
    )

    @field_validator("date")
    @classmethod
    def normalize_date(cls, value: str) -> str:
        return value.strip()

    @field_validator("days")
    @classmethod
    def require_nonzero_days(cls, value: int) -> int:
        if value == 0:
            raise ValueError("days must be nonzero")
        return value


class DateRangeArgs(BaseModel):
    """Arguments for measuring a signed interval between dates."""

    model_config = ConfigDict(extra="forbid")

    start_date: str = Field(min_length=10, max_length=10, description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(min_length=10, max_length=10, description="End date in YYYY-MM-DD format.")

    @field_validator("start_date", "end_date")
    @classmethod
    def normalize_date(cls, value: str) -> str:
        return value.strip()


class _TemporalToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})?$")
_TIMEZONE_RE = re.compile(r"^[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)+$")


class TemporalToolset(ReActToolset):
    """Provide deterministic, offline calendar and timezone operations."""

    now_utc: ClassVar[Callable[[], datetime]] = staticmethod(lambda: datetime.now(timezone.utc))
    local_timezone_name: ClassVar[Callable[[], str]] = staticmethod(get_localzone_name)

    def get_skill_specs(self) -> list[ReActSkillSpec]:
        return [
            ReActSkillSpec(
                name="dates_and_time",
                description="Current time, timezones, supplied dates, and calendar or business-day operations.",
                tool_names=[
                    "get_current_time",
                    "convert_datetime",
                    "get_date_info",
                    "add_calendar_period",
                    "add_business_days",
                    "calendar_days_between",
                    "business_days_between",
                ],
                instructions=(
                    "Use get_current_time only for now, get_date_info for a supplied date, add tools for future or "
                    "past dates, and *_days_between only to compare two known dates. Omit timezone for system-local time."
                ),
                order=20,
            )
        ]

    def get_toolspecs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="get_current_time",
                description="Return the current date and time in an IANA timezone, or the system timezone when omitted.",
                parameter_class=GetCurrentTimeArgs,
            ),
            ToolSpec(
                name="convert_datetime",
                description=(
                    "Convert an ISO datetime to an IANA timezone. A datetime without an offset uses from_timezone or "
                    "the system timezone; an omitted target also uses the system timezone."
                ),
                parameter_class=ConvertDatetimeArgs,
            ),
            ToolSpec(
                name="get_date_info",
                description="Return the weekday and weekend status for a YYYY-MM-DD date.",
                parameter_class=GetDateInfoArgs,
            ),
            ToolSpec(
                name="add_calendar_period",
                description=(
                    "Add signed calendar years, months, weeks, and days to a date. Month-end results clamp to the last "
                    "valid day."
                ),
                parameter_class=AddCalendarPeriodArgs,
            ),
            ToolSpec(
                name="add_business_days",
                description="Add signed Monday-to-Friday days to a date. Holidays are not included.",
                parameter_class=AddBusinessDaysArgs,
            ),
            ToolSpec(
                name="calendar_days_between",
                description="Return signed calendar days from the start date up to, but not including, the end date.",
                parameter_class=DateRangeArgs,
            ),
            ToolSpec(
                name="business_days_between",
                description=(
                    "Return signed Monday-to-Friday days from the start date up to, but not including, the end date. "
                    "Holidays are not included."
                ),
                parameter_class=DateRangeArgs,
            ),
        ]

    def execute(self, tool_name: str, args: BaseModel) -> str:
        try:
            if tool_name == "get_current_time":
                if not isinstance(args, GetCurrentTimeArgs):
                    raise TypeError("get_current_time requires GetCurrentTimeArgs")
                result = self._get_current_time(args)
            elif tool_name == "convert_datetime":
                if not isinstance(args, ConvertDatetimeArgs):
                    raise TypeError("convert_datetime requires ConvertDatetimeArgs")
                result = self._convert_datetime(args)
            elif tool_name == "get_date_info":
                if not isinstance(args, GetDateInfoArgs):
                    raise TypeError("get_date_info requires GetDateInfoArgs")
                result = self._get_date_info(args)
            elif tool_name == "add_calendar_period":
                if not isinstance(args, AddCalendarPeriodArgs):
                    raise TypeError("add_calendar_period requires AddCalendarPeriodArgs")
                result = self._add_calendar_period(args)
            elif tool_name == "add_business_days":
                if not isinstance(args, AddBusinessDaysArgs):
                    raise TypeError("add_business_days requires AddBusinessDaysArgs")
                result = self._add_business_days(args)
            elif tool_name == "calendar_days_between":
                if not isinstance(args, DateRangeArgs):
                    raise TypeError("calendar_days_between requires DateRangeArgs")
                result = self._calendar_days_between(args)
            elif tool_name == "business_days_between":
                if not isinstance(args, DateRangeArgs):
                    raise TypeError("business_days_between requires DateRangeArgs")
                result = self._business_days_between(args)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except _TemporalToolError as exc:
            result = _error_result(exc.code, exc.message)
        return json.dumps(result, ensure_ascii=True, separators=(",", ":"))

    def interpret_result(self, tool_name: str, args: BaseModel, result: str) -> ReActToolOutcome | None:
        if tool_name not in {
            "get_current_time",
            "convert_datetime",
            "get_date_info",
            "add_calendar_period",
            "add_business_days",
            "calendar_days_between",
            "business_days_between",
        }:
            return None
        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("status") == "ok":
            if tool_name == "get_current_time":
                summary = f"{payload.get('datetime', '')} in {payload.get('timezone', '')}"
            elif tool_name == "convert_datetime":
                source = payload.get("input", {})
                summary = (
                    f"{source.get('datetime', '')} in {source.get('timezone', '')} = "
                    f"{payload.get('datetime', '')} in {payload.get('timezone', '')}"
                )
            elif tool_name in {
                "get_date_info",
                "add_calendar_period",
                "add_business_days",
            }:
                summary = f"{payload.get('date', '')} is {payload.get('weekday', '')}"
            elif tool_name == "calendar_days_between":
                summary = (
                    f"{payload.get('start_date', '')} to {payload.get('end_date', '')}: "
                    f"{payload.get('calendar_days', '')} calendar days"
                )
            else:
                summary = (
                    f"{payload.get('start_date', '')} to {payload.get('end_date', '')}: "
                    f"{payload.get('business_days', '')} business days"
                )
            return ReActToolOutcome(
                disposition=ToolOutcomeDisposition.SUCCESS,
                verified_summary=summary.strip(),
            )

        error = payload.get("error")
        if not isinstance(error, dict):
            return None
        code = error.get("code")
        message = error.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            return None
        if code in {"ambiguous_local_time", "nonexistent_local_time"}:
            disposition = ToolOutcomeDisposition.CLARIFICATION_REQUIRED
        elif code in {
            "invalid_date",
            "invalid_datetime",
            "unknown_timezone",
            "timezone_offset_mismatch",
            "local_timezone_unavailable",
        }:
            disposition = ToolOutcomeDisposition.RETRYABLE_ERROR
        else:
            disposition = ToolOutcomeDisposition.TERMINAL_ERROR
        return ReActToolOutcome(disposition=disposition, code=code, message=message)

    def _timezone(self, requested: str | None) -> tuple[str, ZoneInfo]:
        name = requested
        if name is None:
            try:
                name = type(self).local_timezone_name()
            except Exception as exc:
                raise _TemporalToolError(
                    "local_timezone_unavailable",
                    "The system timezone could not be detected; supply an IANA timezone explicitly.",
                ) from exc
        return _resolve_timezone(name)

    def _get_current_time(self, args: GetCurrentTimeArgs) -> dict[str, object]:
        zone_name, zone = self._timezone(args.timezone)
        now = type(self).now_utc()
        if now.tzinfo is None or now.utcoffset() is None:
            raise _TemporalToolError(
                "date_range_exceeded",
                "The system clock did not return an aware UTC datetime.",
            )
        current = now.astimezone(timezone.utc).astimezone(zone)
        return {
            "status": "ok",
            "datetime": _iso_datetime(current),
            "timezone": zone_name,
            "utc_offset": _format_offset(current),
            "weekday": calendar.day_name[current.weekday()],
            "utc_datetime": _utc_datetime(current),
        }

    def _convert_datetime(self, args: ConvertDatetimeArgs) -> dict[str, object]:
        parsed = _parse_datetime(args.datetime)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            source_name, source_zone = self._timezone(args.from_timezone)
            source = _localize_wall_time(parsed, source_name, source_zone)
        else:
            source = parsed
            if args.from_timezone is not None:
                source_name, source_zone = self._timezone(args.from_timezone)
                rendered_source = parsed.astimezone(timezone.utc).astimezone(source_zone)
                if (
                    rendered_source.replace(tzinfo=None) != parsed.replace(tzinfo=None)
                    or rendered_source.utcoffset() != parsed.utcoffset()
                ):
                    raise _TemporalToolError(
                        "timezone_offset_mismatch",
                        "The datetime offset does not match from_timezone at that local time.",
                    )
                source = rendered_source
            else:
                source_name = _fixed_offset_name(parsed)

        target_name, target_zone = self._timezone(args.to_timezone)
        converted = source.astimezone(timezone.utc).astimezone(target_zone)
        return {
            "status": "ok",
            "input": {
                "datetime": _iso_datetime(source),
                "timezone": source_name,
                "utc_offset": _format_offset(source),
            },
            "datetime": _iso_datetime(converted),
            "timezone": target_name,
            "utc_offset": _format_offset(converted),
            "weekday": calendar.day_name[converted.weekday()],
            "utc_datetime": _utc_datetime(converted),
        }

    def _get_date_info(self, args: GetDateInfoArgs) -> dict[str, object]:
        value = _parse_date(args.date)
        return {
            "status": "ok",
            "date": value.isoformat(),
            "weekday": calendar.day_name[value.weekday()],
            "is_weekend": value.weekday() >= 5,
        }

    def _add_calendar_period(self, args: AddCalendarPeriodArgs) -> dict[str, object]:
        original = _parse_date(args.date)
        try:
            shifted = _add_months(original, args.years * 12 + args.months)
            shifted += timedelta(weeks=args.weeks, days=args.days)
        except (OverflowError, ValueError) as exc:
            raise _TemporalToolError(
                "date_range_exceeded",
                "The calculated date is outside the supported range.",
            ) from exc
        return {
            "status": "ok",
            "input_date": original.isoformat(),
            "date": shifted.isoformat(),
            "weekday": calendar.day_name[shifted.weekday()],
            "is_weekend": shifted.weekday() >= 5,
            "delta": {
                "years": args.years,
                "months": args.months,
                "weeks": args.weeks,
                "days": args.days,
            },
        }

    def _add_business_days(self, args: AddBusinessDaysArgs) -> dict[str, object]:
        original = _parse_date(args.date)
        try:
            shifted = _add_business_days(original, args.days)
        except (OverflowError, ValueError) as exc:
            raise _TemporalToolError(
                "date_range_exceeded",
                "The calculated date is outside the supported range.",
            ) from exc
        return {
            "status": "ok",
            "input_date": original.isoformat(),
            "date": shifted.isoformat(),
            "weekday": calendar.day_name[shifted.weekday()],
            "is_weekend": shifted.weekday() >= 5,
            "business_days": args.days,
        }

    def _calendar_days_between(self, args: DateRangeArgs) -> dict[str, object]:
        start = _parse_date(args.start_date)
        end = _parse_date(args.end_date)
        return {
            "status": "ok",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "calendar_days": (end - start).days,
        }

    def _business_days_between(self, args: DateRangeArgs) -> dict[str, object]:
        start = _parse_date(args.start_date)
        end = _parse_date(args.end_date)
        return {
            "status": "ok",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "business_days": _business_day_difference(start, end),
        }


def _normalize_optional_text(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be blank")
    return value


def _resolve_timezone(value: str) -> tuple[str, ZoneInfo]:
    name = value.strip()
    if name != "UTC" and (not _TIMEZONE_RE.fullmatch(name) or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise _TemporalToolError(
            "unknown_timezone",
            f"Unsupported timezone '{value}'; use an IANA name such as America/Vancouver or UTC, or omit it for "
            "system local time.",
        )
    try:
        return name, ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise _TemporalToolError(
            "unknown_timezone",
            f"Unknown IANA timezone '{value}'; omit it for system local time.",
        ) from exc


def _parse_date(value: str) -> date:
    if not _DATE_RE.fullmatch(value):
        raise _TemporalToolError("invalid_date", "Dates must use YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _TemporalToolError("invalid_date", f"Invalid calendar date '{value}'.") from exc


def _parse_datetime(value: str) -> datetime:
    if not _DATETIME_RE.fullmatch(value):
        raise _TemporalToolError("invalid_datetime", "Datetimes must use ISO format with a T separator.")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise _TemporalToolError("invalid_datetime", f"Invalid ISO datetime '{value}'.") from exc


def _localize_wall_time(value: datetime, zone_name: str, zone: ZoneInfo) -> datetime:
    candidates: list[datetime] = []
    seen_instants: set[datetime] = set()
    for fold in (0, 1):
        candidate = value.replace(tzinfo=zone, fold=fold)
        instant = candidate.astimezone(timezone.utc)
        if instant.astimezone(zone).replace(tzinfo=None) == value and instant not in seen_instants:
            candidates.append(candidate)
            seen_instants.add(instant)
    if not candidates:
        raise _TemporalToolError(
            "nonexistent_local_time",
            f"{value.isoformat()} does not exist in {zone_name} because of a clock change; provide another time.",
        )
    if len(candidates) > 1:
        raise _TemporalToolError(
            "ambiguous_local_time",
            f"{value.isoformat()} occurs twice in {zone_name}; provide an explicit UTC offset.",
        )
    return candidates[0]


def _add_months(value: date, months: int) -> date:
    month_index = (value.year - 1) * 12 + value.month - 1 + months
    if month_index < 0 or month_index > 9_999 * 12 - 1:
        raise OverflowError
    zero_based_year, zero_based_month = divmod(month_index, 12)
    year = zero_based_year + 1
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _add_business_days(value: date, count: int) -> date:
    direction = 1 if count >= 0 else -1
    remaining = abs(count)
    result = value
    while remaining:
        result += timedelta(days=direction)
        if result.weekday() < 5:
            remaining -= 1
    return result


def _business_day_difference(start: date, end: date) -> int:
    if end < start:
        return -_business_day_difference(end, start)
    day_count = (end - start).days
    full_weeks, remainder = divmod(day_count, 7)
    return full_weeks * 5 + sum(1 for offset in range(remainder) if (start.weekday() + offset) % 7 < 5)


def _iso_datetime(value: datetime) -> str:
    timespec = "microseconds" if value.microsecond else "seconds"
    return value.isoformat(timespec=timespec)


def _utc_datetime(value: datetime) -> str:
    timespec = "microseconds" if value.microsecond else "seconds"
    return value.astimezone(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")


def _format_offset(value: datetime) -> str:
    offset = value.utcoffset()
    if offset is None:
        raise _TemporalToolError("invalid_datetime", "The datetime does not have a UTC offset.")
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    suffix = f":{seconds:02d}" if seconds else ""
    return f"{sign}{hours:02d}:{minutes:02d}{suffix}"


def _fixed_offset_name(value: datetime) -> str:
    offset = _format_offset(value)
    return "UTC" if offset == "+00:00" else f"UTC{offset}"


def _error_result(code: str, message: str) -> dict[str, object]:
    return {"status": "error", "error": {"code": code, "message": message}}

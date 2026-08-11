from datetime import datetime, timezone
import json
from unittest.mock import Mock

from pydantic import ValidationError
import pytest

from rustic_ai.llm_agent.react import (
    AddBusinessDaysArgs,
    AddCalendarPeriodArgs,
    CompositeToolset,
    ConvertDatetimeArgs,
    DateRangeArgs,
    GetCurrentTimeArgs,
    GetDateInfoArgs,
    ReActAgentConfig,
    TemporalToolset,
)


def execute(name, args):
    return json.loads(TemporalToolset().execute(name, args))


def test_tool_specs_expose_strict_typed_schemas():
    tools = {tool.function.name: tool for tool in TemporalToolset().chat_tools}

    assert set(tools) == {
        "get_current_time",
        "convert_datetime",
        "get_date_info",
        "add_calendar_period",
        "add_business_days",
        "calendar_days_between",
        "business_days_between",
    }
    for tool in tools.values():
        assert tool.function.parameters.model_dump()["additionalProperties"] is False
    assert tools["get_current_time"].function.parameters.model_dump().get("required", []) == []
    assert tools["convert_datetime"].function.parameters.model_dump()["required"] == ["datetime"]
    assert tools["get_date_info"].function.parameters.model_dump()["required"] == ["date"]
    assert tools["add_calendar_period"].function.parameters.model_dump()["required"] == ["date"]
    assert tools["add_business_days"].function.parameters.model_dump()["required"] == ["date", "days"]
    assert tools["calendar_days_between"].function.parameters.model_dump()["required"] == [
        "start_date",
        "end_date",
    ]
    assert tools["business_days_between"].function.parameters.model_dump()["required"] == [
        "start_date",
        "end_date",
    ]


def test_argument_models_reject_unknown_properties_and_blank_timezones():
    with pytest.raises(ValidationError):
        GetCurrentTimeArgs(timezone="UTC", locale="en")
    with pytest.raises(ValidationError):
        GetCurrentTimeArgs(timezone=" ")

    with pytest.raises(ValidationError, match="at least one calendar period"):
        AddCalendarPeriodArgs(date="2026-08-06")
    with pytest.raises(ValidationError, match="days must be nonzero"):
        AddBusinessDaysArgs(date="2026-08-06", days=0)


def test_toolset_serializes_and_composes():
    config = ReActAgentConfig(toolset={"kind": "rustic_ai.llm_agent.react.toolsets.temporal.TemporalToolset"})
    composite = CompositeToolset(toolsets=[config.toolset])

    assert isinstance(config.toolset, TemporalToolset)
    result = json.loads(composite.execute("get_date_info", GetDateInfoArgs(date="2026-08-06")))
    assert result["weekday"] == "Thursday"


def test_current_time_uses_explicit_timezone_and_one_utc_instant(monkeypatch):
    monkeypatch.setattr(
        TemporalToolset,
        "now_utc",
        staticmethod(lambda: datetime(2026, 8, 6, 12, 34, 56, tzinfo=timezone.utc)),
    )

    result = execute("get_current_time", GetCurrentTimeArgs(timezone="America/Vancouver"))

    assert result == {
        "status": "ok",
        "datetime": "2026-08-06T05:34:56-07:00",
        "timezone": "America/Vancouver",
        "utc_offset": "-07:00",
        "weekday": "Thursday",
        "utc_datetime": "2026-08-06T12:34:56Z",
    }


def test_current_time_defaults_to_detected_system_timezone(monkeypatch):
    monkeypatch.setattr(TemporalToolset, "local_timezone_name", staticmethod(lambda: "Asia/Kolkata"))
    monkeypatch.setattr(
        TemporalToolset,
        "now_utc",
        staticmethod(lambda: datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc)),
    )

    result = execute("get_current_time", GetCurrentTimeArgs())

    assert result["datetime"] == "2026-01-02T04:30:00+05:30"
    assert result["weekday"] == "Friday"
    assert result["timezone"] == "Asia/Kolkata"


def test_local_timezone_detection_failure_is_actionable(monkeypatch):
    def fail():
        raise RuntimeError("unavailable")

    monkeypatch.setattr(TemporalToolset, "local_timezone_name", staticmethod(fail))
    result = execute("get_current_time", GetCurrentTimeArgs())

    assert result["error"]["code"] == "local_timezone_unavailable"


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        ("2026-08-06T09:00:00", "America/Vancouver", "Europe/London", "2026-08-06T17:00:00+01:00"),
        ("2026-08-06T09:00:00-07:00", "America/Vancouver", "UTC", "2026-08-06T16:00:00+00:00"),
        ("2026-01-01T23:30:00Z", None, "Asia/Kolkata", "2026-01-02T05:00:00+05:30"),
    ],
)
def test_converts_datetimes(value, source, target, expected):
    result = execute(
        "convert_datetime",
        ConvertDatetimeArgs(datetime=value, from_timezone=source, to_timezone=target),
    )

    assert result["status"] == "ok"
    assert result["datetime"] == expected


def test_naive_conversion_defaults_both_timezones_to_local(monkeypatch):
    monkeypatch.setattr(TemporalToolset, "local_timezone_name", staticmethod(lambda: "America/Vancouver"))

    result = execute("convert_datetime", ConvertDatetimeArgs(datetime="2026-08-06T09:00:00"))

    assert result["input"]["timezone"] == "America/Vancouver"
    assert result["timezone"] == "America/Vancouver"
    assert result["datetime"] == "2026-08-06T09:00:00-07:00"


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("2026-03-08T02:30:00", "nonexistent_local_time"),
        ("2026-11-01T01:30:00", "ambiguous_local_time"),
    ],
)
def test_dst_wall_time_requires_clarification(value, code):
    zone = "America/New_York" if code == "ambiguous_local_time" else "America/Vancouver"
    args = ConvertDatetimeArgs(
        datetime=value,
        from_timezone=zone,
        to_timezone="UTC",
    )
    result = execute("convert_datetime", args)
    outcome = TemporalToolset().interpret_result("convert_datetime", args, json.dumps(result))

    assert result["error"]["code"] == code
    assert outcome is not None
    assert outcome.disposition == "clarification_required"


def test_explicit_offset_resolves_dst_ambiguity():
    result = execute(
        "convert_datetime",
        ConvertDatetimeArgs(
            datetime="2026-11-01T01:30:00-07:00",
            from_timezone="America/Los_Angeles",
            to_timezone="UTC",
        ),
    )

    assert result["datetime"] == "2026-11-01T08:30:00+00:00"


def test_rejects_mismatched_offset_and_timezone():
    result = execute(
        "convert_datetime",
        ConvertDatetimeArgs(
            datetime="2026-08-06T09:00:00-08:00",
            from_timezone="America/Vancouver",
            to_timezone="UTC",
        ),
    )

    assert result["error"]["code"] == "timezone_offset_mismatch"


@pytest.mark.parametrize("timezone_name", ["PST", "../UTC", "/usr/share/zoneinfo/UTC", "America/../UTC"])
def test_rejects_abbreviations_and_unsafe_timezone_names(timezone_name):
    result = execute("get_current_time", GetCurrentTimeArgs(timezone=timezone_name))

    assert result["error"]["code"] == "unknown_timezone"


@pytest.mark.parametrize(
    ("args", "expected_date", "weekday"),
    [
        (AddCalendarPeriodArgs(date="2024-01-31", months=1), "2024-02-29", "Thursday"),
        (AddCalendarPeriodArgs(date="2023-01-31", years=1, months=1), "2024-02-29", "Thursday"),
        (AddCalendarPeriodArgs(date="2026-08-06", weeks=2, days=3), "2026-08-23", "Sunday"),
    ],
)
def test_adds_calendar_periods_in_documented_order(args, expected_date, weekday):
    result = execute("add_calendar_period", args)

    assert result["date"] == expected_date
    assert result["weekday"] == weekday
    assert result["is_weekend"] is (weekday in {"Saturday", "Sunday"})


@pytest.mark.parametrize(
    ("date_value", "days", "expected_date", "weekday"),
    [
        ("2026-08-08", 1, "2026-08-10", "Monday"),
        ("2026-08-10", -1, "2026-08-07", "Friday"),
    ],
)
def test_adds_business_days_separately(date_value, days, expected_date, weekday):
    result = execute("add_business_days", AddBusinessDaysArgs(date=date_value, days=days))

    assert result["date"] == expected_date
    assert result["weekday"] == weekday
    assert result["business_days"] == days


def test_mixed_calendar_and_business_arithmetic_chains_results():
    calendar_result = execute(
        "add_calendar_period",
        AddCalendarPeriodArgs(date="2026-08-03", weeks=2, days=3),
    )
    business_result = execute(
        "add_business_days",
        AddBusinessDaysArgs(date=calendar_result["date"], days=4),
    )

    assert calendar_result["date"] == "2026-08-20"
    assert business_result["date"] == "2026-08-26"
    assert business_result["weekday"] == "Wednesday"


@pytest.mark.parametrize(
    ("date_value", "weekday", "is_weekend"),
    [
        ("2026-08-06", "Thursday", False),
        ("2026-08-08", "Saturday", True),
        ("0001-01-01", "Monday", False),
        ("9999-12-31", "Friday", False),
    ],
)
def test_gets_date_info_without_arithmetic(date_value, weekday, is_weekend):
    result = execute("get_date_info", GetDateInfoArgs(date=date_value))

    assert result == {
        "status": "ok",
        "date": date_value,
        "weekday": weekday,
        "is_weekend": is_weekend,
    }


@pytest.mark.parametrize(
    ("start", "end", "calendar_days", "business_days"),
    [
        ("2026-08-03", "2026-08-04", 1, 1),
        ("2026-08-07", "2026-08-10", 3, 1),
        ("2026-08-08", "2026-08-10", 2, 0),
        ("2026-08-10", "2026-08-07", -3, -1),
        ("2026-08-06", "2026-08-06", 0, 0),
    ],
)
def test_difference_tools_use_signed_half_open_intervals(start, end, calendar_days, business_days):
    args = DateRangeArgs(start_date=start, end_date=end)
    calendar_result = execute("calendar_days_between", args)
    business_result = execute("business_days_between", args)

    assert calendar_result["calendar_days"] == calendar_days
    assert "business_days" not in calendar_result
    assert business_result["business_days"] == business_days
    assert "calendar_days" not in business_result


@pytest.mark.parametrize(
    ("name", "args", "code"),
    [
        ("get_date_info", GetDateInfoArgs(date="2026-02-30"), "invalid_date"),
        (
            "convert_datetime",
            ConvertDatetimeArgs(datetime="2026-08-06 10:00", to_timezone="UTC"),
            "invalid_datetime",
        ),
        (
            "add_calendar_period",
            AddCalendarPeriodArgs(date="9999-12-31", days=1),
            "date_range_exceeded",
        ),
        ("add_business_days", AddBusinessDaysArgs(date="9999-12-31", days=1), "date_range_exceeded"),
    ],
)
def test_returns_stable_errors(name, args, code):
    result = execute(name, args)
    assert result["error"]["code"] == code


def test_interprets_success_retryable_and_terminal_results():
    toolset = TemporalToolset()
    success_args = GetDateInfoArgs(date="2026-08-06")
    success_result = toolset.execute("get_date_info", success_args)
    success = toolset.interpret_result("get_date_info", success_args, success_result)
    retryable = toolset.interpret_result(
        "get_current_time",
        GetCurrentTimeArgs(timezone="Mars/Olympus"),
        json.dumps({"status": "error", "error": {"code": "unknown_timezone", "message": "Unknown."}}),
    )
    terminal = toolset.interpret_result(
        "get_date_info",
        success_args,
        json.dumps({"status": "error", "error": {"code": "date_range_exceeded", "message": "Range."}}),
    )

    assert success is not None and success.disposition == "success"
    assert success.verified_summary == "2026-08-06 is Thursday"
    assert retryable is not None and retryable.disposition == "retryable_error"
    assert terminal is not None and terminal.disposition == "terminal_error"


def test_outputs_are_compact_ascii_json():
    success = TemporalToolset().execute("get_current_time", GetCurrentTimeArgs(timezone="UTC"))
    error = TemporalToolset().execute("get_date_info", GetDateInfoArgs(date="not-a-date"))

    assert ": " not in success
    assert ": " not in error
    success.encode("ascii")
    error.encode("ascii")


def test_rejects_unknown_tool_and_wrong_argument_models():
    toolset = TemporalToolset()

    with pytest.raises(ValueError, match="Unknown tool"):
        toolset.execute("schedule", GetCurrentTimeArgs())
    with pytest.raises(TypeError, match="requires GetCurrentTimeArgs"):
        toolset.execute("get_current_time", Mock())
    with pytest.raises(TypeError, match="requires ConvertDatetimeArgs"):
        toolset.execute("convert_datetime", Mock())
    with pytest.raises(TypeError, match="requires GetDateInfoArgs"):
        toolset.execute("get_date_info", Mock())
    with pytest.raises(TypeError, match="requires AddCalendarPeriodArgs"):
        toolset.execute("add_calendar_period", Mock())
    with pytest.raises(TypeError, match="requires AddBusinessDaysArgs"):
        toolset.execute("add_business_days", Mock())
    with pytest.raises(TypeError, match="requires DateRangeArgs"):
        toolset.execute("calendar_days_between", Mock())
    with pytest.raises(TypeError, match="requires DateRangeArgs"):
        toolset.execute("business_days_between", Mock())

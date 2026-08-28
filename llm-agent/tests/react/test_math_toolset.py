import json
from unittest.mock import Mock

from pydantic import ValidationError
import pytest

from rustic_ai.llm_agent.react import (
    CalculateArgs,
    CompositeToolset,
    ConvertUnitsArgs,
    MathToolset,
    ReActAgentConfig,
)
from rustic_ai.llm_agent.react.toolsets import math as math_toolset_module


def calculate(expression: str) -> dict:
    return json.loads(MathToolset().execute("calculate", CalculateArgs(expression=expression)))


def convert(value, from_unit: str, to_unit: str) -> dict:
    args = ConvertUnitsArgs(value=value, from_unit=from_unit, to_unit=to_unit)
    return json.loads(MathToolset().execute("convert_units", args))


def test_tool_specs_expose_strict_typed_schemas():
    tools = {tool.function.name: tool for tool in MathToolset().chat_tools}

    assert set(tools) == {"calculate", "convert_units"}
    calculate_schema = tools["calculate"].function.parameters.model_dump()
    convert_schema = tools["convert_units"].function.parameters.model_dump()
    assert calculate_schema["required"] == ["expression"]
    assert calculate_schema["additionalProperties"] is False
    assert calculate_schema["properties"]["expression"]["maxLength"] == 256
    assert convert_schema["required"] == ["value", "from_unit", "to_unit"]
    assert convert_schema["additionalProperties"] is False
    assert convert_schema["properties"]["value"]["anyOf"] == [
        {"type": "number"},
        {"type": "string"},
    ]
    assert convert_schema["properties"]["from_unit"]["enum"] == convert_schema["properties"]["to_unit"]["enum"]
    assert convert_schema["properties"]["from_unit"]["enum"] == [unit.key for unit in math_toolset_module._UNITS]
    assert "pattern" not in json.dumps(convert_schema)


def test_argument_models_reject_extra_and_non_finite_values():
    with pytest.raises(ValidationError):
        CalculateArgs(expression="2+2", precision=2)
    with pytest.raises(ValidationError):
        ConvertUnitsArgs(value="NaN", from_unit="m", to_unit="ft")
    with pytest.raises(ValidationError, match="must not exceed 40 characters"):
        ConvertUnitsArgs(value="1", from_unit="m" * 41, to_unit="ft")


def test_toolset_serializes_and_composes():
    config = ReActAgentConfig(toolset={"kind": "rustic_ai.llm_agent.react.toolsets.math.MathToolset"})
    composite = CompositeToolset(toolsets=[config.toolset])

    assert isinstance(config.toolset, MathToolset)
    assert json.loads(composite.execute("calculate", CalculateArgs(expression="5*5")))["value"] == "25"


def test_interprets_success_retryable_terminal_and_clarification_results():
    toolset = MathToolset()

    success_args = CalculateArgs(expression="2+2")
    success_result = toolset.execute("calculate", success_args)
    success = toolset.interpret_result("calculate", success_args, success_result)
    retryable = toolset.interpret_result(
        "calculate",
        CalculateArgs(expression="unknown+1"),
        json.dumps({"status": "error", "error": {"code": "unknown_identifier", "message": "Unknown."}}),
    )
    terminal = toolset.interpret_result(
        "calculate",
        CalculateArgs(expression="1/0"),
        json.dumps({"status": "error", "error": {"code": "division_by_zero", "message": "Zero."}}),
    )
    conversion_args = ConvertUnitsArgs(value=1, from_unit="gallon", to_unit="l")
    clarification = toolset.interpret_result(
        "convert_units",
        conversion_args,
        json.dumps({"status": "error", "error": {"code": "ambiguous_unit", "message": "Specify system."}}),
    )

    assert success is not None and success.disposition == "success"
    assert success.verified_summary == "2+2 = 4"
    assert retryable is not None and retryable.disposition == "retryable_error"
    assert terminal is not None and terminal.disposition == "terminal_error"
    assert clarification is not None and clarification.disposition == "clarification_required"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("-2^2", "-4"),
        ("2^-2", "0.25"),
        ("2^3^2", "512"),
        ("1e3 + .5", "1000.5"),
        ("abs(-3)", "3"),
        ("sqrt(144)", "12"),
        ("min(4,2,9)", "2"),
        ("max(4,2,9)", "9"),
        ("sum(1,2,3,4)", "10"),
        ("mean(2,4,9)", "5"),
        ("median(9,2,5)", "5"),
        ("median(1,8,3,4)", "3.5"),
        ("floor(3.8)", "3"),
        ("floor(-3.2)", "-4"),
        ("ceil(3.2)", "4"),
        ("ceil(-3.8)", "-3"),
        ("round(10/3,4)", "3.3333"),
        ("15 / 100 * 240", "36"),
    ],
)
def test_calculates_supported_expressions(expression, expected):
    result = calculate(expression)

    assert result["status"] == "ok"
    assert result["value"] == expected


def test_normalizes_safe_operator_variants():
    result = calculate("6 × 7 − 2 ÷ 2")

    assert result == {
        "status": "ok",
        "expression": "6*7-2/2",
        "value": "41",
        "approximate": False,
    }


def test_reports_exact_and_approximate_results():
    assert calculate("1/8")["approximate"] is False
    repeating = calculate("1/3")
    assert repeating["approximate"] is True
    assert repeating["value"] == "0." + "3" * 50
    assert calculate("sqrt(2)")["approximate"] is True
    assert calculate("pi")["approximate"] is True


@pytest.mark.parametrize(
    ("expression", "code"),
    [
        ("1/0", "division_by_zero"),
        ("sqrt(-1)", "domain_error"),
        ("2**1.5", "domain_error"),
        ("2**1001", "numeric_limit_exceeded"),
        ("round(1,13)", "domain_error"),
        ("unknown + 1", "unknown_identifier"),
        ("pow(2,3)", "unknown_function"),
        ("mean()", "invalid_expression"),
        ("floor(1,2)", "invalid_expression"),
        ("2(3+4)", "unsupported_syntax"),
        ("10%", "unsupported_syntax"),
        ('__import__("os")', "unsupported_syntax"),
        ("(1).__class__", "unsupported_syntax"),
        ("[1,2]", "unsupported_syntax"),
        ("x=1", "unsupported_syntax"),
        ("1 # comment", "unsupported_syntax"),
    ],
)
def test_rejects_unsafe_or_unsupported_expressions(expression, code):
    result = calculate(expression)

    assert result["status"] == "error"
    assert result["error"]["code"] == code


def test_enforces_expression_resource_limits():
    literal_result = calculate("1" * 101)
    token_result = calculate("+".join(["1"] * 65))
    nesting_result = calculate("(" * 25 + "1" + ")" * 25)

    assert literal_result["error"]["code"] == "numeric_limit_exceeded"
    assert token_result["error"]["code"] == "numeric_limit_exceeded"
    assert nesting_result["error"]["code"] == "numeric_limit_exceeded"


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (1, "km", "m", "1000"),
        (1, "foot", "inches", "12"),
        (1, "lb", "g", "453.59237"),
        (1, "day", "hours", "24"),
        (1, "hectare", "m²", "10000"),
        (36, "km/h", "m/s", "10"),
        (1, "US gallon", "litres", "3.785411784"),
        (1, "imperial gallon", "litres", "4.54609"),
    ],
)
def test_converts_supported_units_and_aliases(value, source, target, expected):
    result = convert(value, source, target)

    assert result["status"] == "ok"
    assert result["value"] == expected


@pytest.mark.parametrize("unit", math_toolset_module._UNITS, ids=lambda unit: unit.key)
def test_every_registered_unit_and_alias_resolves(unit):
    for alias in (unit.key, unit.symbol, *unit.aliases):
        result = convert(1, alias, unit.key)
        assert result["status"] == "ok", alias
        assert result["value"] == "1", alias


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (32, "fahrenheit", "celsius", "0"),
        (212, "fahrenheit", "celsius", "100"),
        (0, "celsius", "fahrenheit", "32"),
        (273.15, "kelvin", "celsius", "0"),
    ],
)
def test_converts_affine_temperatures(value, source, target, expected):
    assert convert(value, source, target)["value"] == expected


def test_conversion_returns_canonical_structured_result():
    result = convert("98.6", "degrees fahrenheit", "°C")

    assert result == {
        "status": "ok",
        "value": "37",
        "unit": "celsius",
        "symbol": "°C",
        "dimension": "temperature",
        "input": {"value": "98.6", "unit": "fahrenheit"},
    }


@pytest.mark.parametrize("unit", ["gallon", "gal", "quart", "pint", "cup", "fluid ounce"])
def test_rejects_ambiguous_volume_units(unit):
    assert convert(1, unit, "l")["error"]["code"] == "ambiguous_unit"


def test_rejects_unknown_incompatible_and_impossible_conversions():
    assert convert(1, "furlong", "m")["error"]["code"] == "unknown_unit"
    assert convert(1, "m", "kg")["error"]["code"] == "incompatible_units"
    assert convert("-273.16", "celsius", "kelvin")["error"]["code"] == "below_absolute_zero"


def test_outputs_are_compact_ascii_json():
    success = MathToolset().execute("convert_units", ConvertUnitsArgs(value=0, from_unit="c", to_unit="f"))
    error = MathToolset().execute("calculate", CalculateArgs(expression="1/0"))

    assert "°" not in success
    assert ": " not in success
    assert ": " not in error


def test_rejects_unknown_tool_and_wrong_argument_models():
    toolset = MathToolset()

    with pytest.raises(ValueError, match="Unknown tool"):
        toolset.execute("math", CalculateArgs(expression="2+2"))
    with pytest.raises(TypeError, match="requires CalculateArgs"):
        toolset.execute("calculate", Mock())
    with pytest.raises(TypeError, match="requires ConvertUnitsArgs"):
        toolset.execute("convert_units", Mock())

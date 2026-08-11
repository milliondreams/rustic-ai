"""Deterministic arithmetic and unit-conversion tools for ReAct agents."""

from dataclasses import dataclass
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
    Rounded,
    localcontext,
)
import json
import re
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator

from rustic_ai.core.guild.agent_ext.depends.llm.tools_manager import ToolSpec
from rustic_ai.llm_agent.react.toolset import (
    ReActSkillSpec,
    ReActToolOutcome,
    ReActToolset,
    ToolOutcomeDisposition,
)


class CalculateArgs(BaseModel):
    """Arguments accepted by the deterministic calculator."""

    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        min_length=1,
        max_length=256,
        description=(
            "Arithmetic expression using numbers, parentheses, +, -, *, /, **, ^, "
            "pi, e, abs, sqrt, min, max, sum, mean, median, floor, ceil, and round. Use forms such as "
            "15 / 100 * 240, sqrt(144), mean(2, 4, 6), median(9, 2, 5), floor(3.8), "
            "ceil(-3.8), 2.5 ** 3, round(10 / 3, 2), and (18 - 3) * 4. "
            "Do not use natural-language operators, percent syntax, units, or prose."
        ),
    )

    @field_validator("expression")
    @classmethod
    def normalize_expression(cls, value: str) -> str:
        expression = value.strip()
        if not expression:
            raise ValueError("expression must not be blank")
        return expression


# Pydantic's Decimal regex uses lookaheads that llama.cpp's JSON grammar cannot parse.
DecimalInput = Annotated[
    Decimal,
    WithJsonSchema({"anyOf": [{"type": "number"}, {"type": "string"}]}),
]

_CANONICAL_UNIT_IDS = (
    "mm",
    "cm",
    "m",
    "km",
    "in",
    "ft",
    "yd",
    "mi",
    "mg",
    "g",
    "kg",
    "oz",
    "lb",
    "celsius",
    "fahrenheit",
    "kelvin",
    "ms",
    "s",
    "min",
    "h",
    "day",
    "ml",
    "l",
    "us_fl_oz",
    "us_cup",
    "us_pint",
    "us_quart",
    "us_gallon",
    "imperial_fl_oz",
    "imperial_cup",
    "imperial_pint",
    "imperial_quart",
    "imperial_gallon",
    "m2",
    "km2",
    "ft2",
    "yd2",
    "acre",
    "hectare",
    "m/s",
    "km/h",
    "mph",
)


class ConvertUnitsArgs(BaseModel):
    """Arguments accepted by the physical-unit converter."""

    model_config = ConfigDict(extra="forbid")

    value: DecimalInput = Field(description="Finite scalar value to convert.")
    from_unit: str = Field(json_schema_extra={"enum": list(_CANONICAL_UNIT_IDS)})
    to_unit: str = Field(json_schema_extra={"enum": list(_CANONICAL_UNIT_IDS)})

    @field_validator("value")
    @classmethod
    def require_finite_value(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("value must be finite")
        return value

    @field_validator("from_unit", "to_unit")
    @classmethod
    def normalize_unit_input(cls, value: str) -> str:
        unit = value.strip()
        if not unit:
            raise ValueError("unit must not be blank")
        if len(unit) > 40:
            raise ValueError("unit must not exceed 40 characters")
        return unit


class _MathToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NORMALIZE_EXPRESSION = str.maketrans({"×": "*", "÷": "/", "−": "-", "–": "-", "—": "-"})


class _ExpressionParser:
    MAX_TOKENS = 128
    MAX_NESTING = 24
    MAX_LITERAL_DIGITS = 100
    MAX_EXPONENT = 1_000
    MAX_RESULT_ADJUSTED_EXPONENT = 10_000
    MAX_FUNCTION_ARGS = 16

    CONSTANTS = {
        "pi": Decimal("3.1415926535897932384626433832795028841971693993751"),
        "e": Decimal("2.7182818284590452353602874713526624977572470937000"),
    }
    FUNCTIONS = frozenset({"abs", "sqrt", "min", "max", "sum", "mean", "median", "floor", "ceil", "round"})

    def __init__(self, expression: str):
        self.tokens = self._tokenize(expression.translate(_NORMALIZE_EXPRESSION))
        self.index = 0
        self.nesting = 0
        self.intrinsically_approximate = False
        self.normalized_expression = "".join(token.value for token in self.tokens if token.kind != "EOF")

    @classmethod
    def _tokenize(cls, expression: str) -> list[_Token]:
        tokens: list[_Token] = []
        position = 0
        while position < len(expression):
            character = expression[position]
            if character.isspace():
                position += 1
                continue

            number = _NUMBER_RE.match(expression, position)
            if number:
                value = number.group(0)
                if sum(character.isdigit() for character in value) > cls.MAX_LITERAL_DIGITS:
                    raise _MathToolError("numeric_limit_exceeded", "A numeric literal is too large.")
                tokens.append(_Token("NUMBER", value, position))
                position = number.end()
                continue

            identifier = _IDENTIFIER_RE.match(expression, position)
            if identifier:
                tokens.append(_Token("IDENTIFIER", identifier.group(0), position))
                position = identifier.end()
                continue

            if expression.startswith("**", position):
                tokens.append(_Token("OPERATOR", "**", position))
                position += 2
                continue
            if character == "^":
                tokens.append(_Token("OPERATOR", "**", position))
                position += 1
                continue
            if character in "+-*/":
                tokens.append(_Token("OPERATOR", character, position))
                position += 1
                continue
            if character == "(":
                tokens.append(_Token("LPAREN", character, position))
                position += 1
                continue
            if character == ")":
                tokens.append(_Token("RPAREN", character, position))
                position += 1
                continue
            if character == ",":
                tokens.append(_Token("COMMA", character, position))
                position += 1
                continue

            raise _MathToolError(
                "unsupported_syntax",
                f"Unsupported character at position {position}.",
            )

        if len(tokens) > cls.MAX_TOKENS:
            raise _MathToolError("numeric_limit_exceeded", "The expression contains too many tokens.")
        tokens.append(_Token("EOF", "", len(expression)))
        return tokens

    def parse(self) -> Decimal:
        value = self._parse_expression()
        if self._current.kind != "EOF":
            raise _MathToolError(
                "unsupported_syntax",
                f"Unexpected token at position {self._current.position}.",
            )
        return self._validate_result(value)

    @property
    def _current(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self._current
        self.index += 1
        return token

    def _match_operator(self, *values: str) -> str | None:
        if self._current.kind == "OPERATOR" and self._current.value in values:
            return self._advance().value
        return None

    def _parse_expression(self) -> Decimal:
        value = self._parse_term()
        while operator := self._match_operator("+", "-"):
            right = self._parse_term()
            value = self._apply_binary(operator, value, right)
        return value

    def _parse_term(self) -> Decimal:
        value = self._parse_factor()
        while operator := self._match_operator("*", "/"):
            right = self._parse_factor()
            value = self._apply_binary(operator, value, right)
        return value

    def _parse_factor(self) -> Decimal:
        operator = self._match_operator("+", "-")
        if operator:
            value = self._parse_factor()
            return value if operator == "+" else self._validate_result(-value)
        return self._parse_power()

    def _parse_power(self) -> Decimal:
        value = self._parse_primary()
        if self._match_operator("**"):
            exponent = self._parse_factor()
            value = self._power(value, exponent)
        return value

    def _parse_primary(self) -> Decimal:
        token = self._current
        if token.kind == "NUMBER":
            self._advance()
            try:
                return self._validate_result(Decimal(token.value))
            except InvalidOperation as exc:
                raise _MathToolError("invalid_expression", "Invalid numeric literal.") from exc

        if token.kind == "IDENTIFIER":
            name = self._advance().value
            if self._current.kind == "LPAREN":
                if name not in self.FUNCTIONS:
                    raise _MathToolError("unknown_function", f"Unsupported function '{name}'.")
                return self._parse_function(name)
            if name not in self.CONSTANTS:
                raise _MathToolError("unknown_identifier", f"Unsupported identifier '{name}'.")
            self.intrinsically_approximate = True
            return self.CONSTANTS[name]

        if token.kind == "LPAREN":
            self._enter_nesting()
            self._advance()
            value = self._parse_expression()
            if self._current.kind != "RPAREN":
                raise _MathToolError("invalid_expression", "Missing closing parenthesis.")
            self._advance()
            self.nesting -= 1
            return value

        raise _MathToolError("invalid_expression", f"Expected a number at position {token.position}.")

    def _parse_function(self, name: str) -> Decimal:
        self._enter_nesting()
        self._advance()  # Opening parenthesis.
        arguments: list[Decimal] = []
        if self._current.kind != "RPAREN":
            while True:
                arguments.append(self._parse_expression())
                if len(arguments) > self.MAX_FUNCTION_ARGS:
                    raise _MathToolError("numeric_limit_exceeded", "The function has too many arguments.")
                if self._current.kind != "COMMA":
                    break
                self._advance()
        if self._current.kind != "RPAREN":
            raise _MathToolError("invalid_expression", f"Missing closing parenthesis for '{name}'.")
        self._advance()
        self.nesting -= 1
        return self._call_function(name, arguments)

    def _enter_nesting(self) -> None:
        self.nesting += 1
        if self.nesting > self.MAX_NESTING:
            raise _MathToolError("numeric_limit_exceeded", "The expression is nested too deeply.")

    def _apply_binary(self, operator: str, left: Decimal, right: Decimal) -> Decimal:
        try:
            if operator == "+":
                result = left + right
            elif operator == "-":
                result = left - right
            elif operator == "*":
                result = left * right
            else:
                if right == 0:
                    raise _MathToolError("division_by_zero", "The expression divides by zero.")
                result = left / right
        except (InvalidOperation, DivisionByZero) as exc:
            raise _MathToolError("domain_error", "The operation is not defined for these values.") from exc
        except Overflow as exc:
            raise _MathToolError("numeric_limit_exceeded", "The result exceeds the numeric limits.") from exc
        return self._validate_result(result)

    def _power(self, base: Decimal, exponent: Decimal) -> Decimal:
        integral_exponent = exponent.to_integral_value()
        if exponent != integral_exponent:
            raise _MathToolError("domain_error", "Exponentiation requires an integer exponent.")
        exponent_value = int(integral_exponent)
        if abs(exponent_value) > self.MAX_EXPONENT:
            raise _MathToolError("numeric_limit_exceeded", "The exponent exceeds the supported limit.")
        if base == 0 and exponent_value < 0:
            raise _MathToolError("division_by_zero", "Zero cannot be raised to a negative exponent.")
        try:
            return self._validate_result(base**exponent_value)
        except (InvalidOperation, DivisionByZero) as exc:
            raise _MathToolError("domain_error", "The exponentiation is not defined.") from exc
        except Overflow as exc:
            raise _MathToolError("numeric_limit_exceeded", "The result exceeds the numeric limits.") from exc

    def _call_function(self, name: str, arguments: list[Decimal]) -> Decimal:
        if name in {"abs", "sqrt", "floor", "ceil"} and len(arguments) != 1:
            raise _MathToolError("invalid_expression", f"'{name}' requires exactly one argument.")
        if name in {"min", "max", "sum", "mean", "median"} and not arguments:
            raise _MathToolError("invalid_expression", f"'{name}' requires at least one argument.")
        if name == "round" and len(arguments) != 2:
            raise _MathToolError("invalid_expression", "'round' requires a value and decimal places.")

        if name == "abs":
            return self._validate_result(abs(arguments[0]))
        if name == "sqrt":
            if arguments[0] < 0:
                raise _MathToolError("domain_error", "Square root requires a non-negative value.")
            return self._validate_result(arguments[0].sqrt())
        if name == "min":
            return min(arguments)
        if name == "max":
            return max(arguments)
        if name == "sum":
            return self._validate_result(sum(arguments, Decimal(0)))
        if name == "mean":
            return self._validate_result(sum(arguments, Decimal(0)) / Decimal(len(arguments)))
        if name == "median":
            ordered = sorted(arguments)
            midpoint = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[midpoint]
            return self._validate_result((ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2))
        if name == "floor":
            return self._validate_result(arguments[0].to_integral_value(rounding=ROUND_FLOOR))
        if name == "ceil":
            return self._validate_result(arguments[0].to_integral_value(rounding=ROUND_CEILING))

        places = arguments[1]
        integral_places = places.to_integral_value()
        if places != integral_places or not 0 <= integral_places <= 12:
            raise _MathToolError("domain_error", "Round decimal places must be an integer from 0 to 12.")
        quantum = Decimal(1).scaleb(-int(integral_places))
        try:
            return self._validate_result(arguments[0].quantize(quantum))
        except InvalidOperation as exc:
            raise _MathToolError(
                "numeric_limit_exceeded",
                "The rounded result exceeds the numeric limits.",
            ) from exc

    def _validate_result(self, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise _MathToolError("numeric_limit_exceeded", "The result must be finite.")
        if value != 0 and abs(value.adjusted()) > self.MAX_RESULT_ADJUSTED_EXPONENT:
            raise _MathToolError("numeric_limit_exceeded", "The result exceeds the numeric limits.")
        return value


@dataclass(frozen=True)
class _Unit:
    key: str
    dimension: str
    symbol: str
    numerator: Decimal
    denominator: Decimal = Decimal(1)
    shift: Decimal = Decimal(0)
    aliases: tuple[str, ...] = ()


def _unit(
    key: str,
    dimension: str,
    symbol: str,
    numerator: str,
    *,
    denominator: str = "1",
    shift: str = "0",
    aliases: tuple[str, ...] = (),
) -> _Unit:
    return _Unit(
        key,
        dimension,
        symbol,
        Decimal(numerator),
        Decimal(denominator),
        Decimal(shift),
        aliases,
    )


_UNITS = (
    _unit(
        "mm",
        "length",
        "mm",
        "0.001",
        aliases=("millimeter", "millimeters", "millimetre", "millimetres"),
    ),
    _unit(
        "cm",
        "length",
        "cm",
        "0.01",
        aliases=("centimeter", "centimeters", "centimetre", "centimetres"),
    ),
    _unit("m", "length", "m", "1", aliases=("meter", "meters", "metre", "metres")),
    _unit(
        "km",
        "length",
        "km",
        "1000",
        aliases=("kilometer", "kilometers", "kilometre", "kilometres"),
    ),
    _unit("in", "length", "in", "0.0254", aliases=("inch", "inches")),
    _unit("ft", "length", "ft", "0.3048", aliases=("foot", "feet")),
    _unit("yd", "length", "yd", "0.9144", aliases=("yard", "yards")),
    _unit("mi", "length", "mi", "1609.344", aliases=("mile", "miles")),
    _unit("mg", "mass", "mg", "0.001", aliases=("milligram", "milligrams")),
    _unit("g", "mass", "g", "1", aliases=("gram", "grams")),
    _unit("kg", "mass", "kg", "1000", aliases=("kilogram", "kilograms")),
    _unit("oz", "mass", "oz", "28.349523125", aliases=("ounce", "ounces")),
    _unit("lb", "mass", "lb", "453.59237", aliases=("pound", "pounds", "lbs")),
    _unit(
        "celsius",
        "temperature",
        "°C",
        "1",
        aliases=("c", "°c", "degree celsius", "degrees celsius"),
    ),
    _unit(
        "fahrenheit",
        "temperature",
        "°F",
        "5",
        denominator="9",
        shift="-32",
        aliases=("f", "°f", "degree fahrenheit", "degrees fahrenheit"),
    ),
    _unit("kelvin", "temperature", "K", "1", shift="-273.15", aliases=("k",)),
    _unit("ms", "duration", "ms", "0.001", aliases=("millisecond", "milliseconds")),
    _unit("s", "duration", "s", "1", aliases=("second", "seconds", "sec", "secs")),
    _unit("min", "duration", "min", "60", aliases=("minute", "minutes", "mins")),
    _unit("h", "duration", "h", "3600", aliases=("hour", "hours", "hr", "hrs")),
    _unit("day", "duration", "d", "86400", aliases=("days", "d")),
    _unit(
        "ml",
        "volume",
        "mL",
        "0.001",
        aliases=("milliliter", "milliliters", "millilitre", "millilitres"),
    ),
    _unit("l", "volume", "L", "1", aliases=("liter", "liters", "litre", "litres")),
    _unit(
        "us_fl_oz",
        "volume",
        "US fl oz",
        "0.0295735295625",
        aliases=("us fluid ounce", "us fluid ounces", "us fl oz"),
    ),
    _unit("us_cup", "volume", "US cup", "0.2365882365", aliases=("us cup", "us cups")),
    _unit(
        "us_pint",
        "volume",
        "US pt",
        "0.473176473",
        aliases=("us pint", "us pints", "us pt"),
    ),
    _unit(
        "us_quart",
        "volume",
        "US qt",
        "0.946352946",
        aliases=("us quart", "us quarts", "us qt"),
    ),
    _unit(
        "us_gallon",
        "volume",
        "US gal",
        "3.785411784",
        aliases=("us gallon", "us gallons", "us gal"),
    ),
    _unit(
        "imperial_fl_oz",
        "volume",
        "imp fl oz",
        "0.0284130625",
        aliases=(
            "imperial fluid ounce",
            "imperial fluid ounces",
            "imperial fl oz",
            "imp fl oz",
        ),
    ),
    _unit(
        "imperial_cup",
        "volume",
        "imp cup",
        "0.284130625",
        aliases=("imperial cup", "imperial cups", "imp cup"),
    ),
    _unit(
        "imperial_pint",
        "volume",
        "imp pt",
        "0.56826125",
        aliases=("imperial pint", "imperial pints", "imp pt"),
    ),
    _unit(
        "imperial_quart",
        "volume",
        "imp qt",
        "1.1365225",
        aliases=("imperial quart", "imperial quarts", "imp qt"),
    ),
    _unit(
        "imperial_gallon",
        "volume",
        "imp gal",
        "4.54609",
        aliases=("imperial gallon", "imperial gallons", "imp gal"),
    ),
    _unit(
        "m2",
        "area",
        "m²",
        "1",
        aliases=("square meter", "square meters", "square metre", "square metres"),
    ),
    _unit(
        "km2",
        "area",
        "km²",
        "1000000",
        aliases=(
            "square kilometer",
            "square kilometers",
            "square kilometre",
            "square kilometres",
        ),
    ),
    _unit("ft2", "area", "ft²", "0.09290304", aliases=("square foot", "square feet")),
    _unit("yd2", "area", "yd²", "0.83612736", aliases=("square yard", "square yards")),
    _unit("acre", "area", "acre", "4046.8564224", aliases=("acres",)),
    _unit("hectare", "area", "ha", "10000", aliases=("hectares", "ha")),
    _unit(
        "m/s",
        "speed",
        "m/s",
        "1",
        aliases=(
            "meter per second",
            "meters per second",
            "metre per second",
            "metres per second",
            "mps",
        ),
    ),
    _unit(
        "km/h",
        "speed",
        "km/h",
        "5",
        denominator="18",
        aliases=(
            "kilometer per hour",
            "kilometers per hour",
            "kilometre per hour",
            "kilometres per hour",
            "kph",
        ),
    ),
    _unit("mph", "speed", "mph", "0.44704", aliases=("mile per hour", "miles per hour")),
)

_AMBIGUOUS_UNITS = frozenset(
    {
        "gallon",
        "gallons",
        "gal",
        "quart",
        "quarts",
        "qt",
        "pint",
        "pints",
        "pt",
        "cup",
        "cups",
        "fluid ounce",
        "fluid ounces",
        "fl oz",
        "floz",
    }
)


def _normalize_unit_name(value: str) -> str:
    normalized = value.strip().lower().replace("²", "2").replace("³", "3").replace("_", " ")
    return " ".join(normalized.split())


def _build_unit_aliases() -> dict[str, _Unit]:
    aliases: dict[str, _Unit] = {}
    for unit in _UNITS:
        for alias in (unit.key, unit.symbol, *unit.aliases):
            normalized = _normalize_unit_name(alias)
            existing = aliases.get(normalized)
            if existing and existing != unit:
                raise RuntimeError(f"Conflicting unit alias: {alias}")
            aliases[normalized] = unit
    return aliases


_UNIT_ALIASES = _build_unit_aliases()
_ABSOLUTE_ZERO_CELSIUS = Decimal("-273.15")


class MathToolset(ReActToolset):
    """Provide deterministic, offline arithmetic and physical-unit conversion."""

    DECIMAL_CONTEXT: ClassVar[Context] = Context(prec=50, rounding=ROUND_HALF_EVEN)

    def get_skill_specs(self) -> list[ReActSkillSpec]:
        return [
            ReActSkillSpec(
                name="math_and_units",
                description="Arithmetic, statistics, rounding, and physical-unit conversion.",
                tool_names=["calculate", "convert_units"],
                instructions=(
                    "Use expression syntax only. Do not put prose, units, or currency symbols in expressions. "
                    "Use this for physical conversions; qualify US and Imperial volume standards."
                ),
                examples=[
                    "15 / 100 * 240",
                    "round(86.4 * 1.18 / 4, 2)",
                    "5 mi to km",
                    "90 km/h to m/s",
                ],
                order=10,
            ),
        ]

    def get_toolspecs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="calculate",
                description=(
                    "Evaluate a deterministic decimal arithmetic expression. Pass expression syntax only, without "
                    "units, percent signs, or prose."
                ),
                parameter_class=CalculateArgs,
            ),
            ToolSpec(
                name="convert_units",
                description=(
                    "Convert one scalar value between compatible physical units. Volume standards must be qualified "
                    "as US or Imperial; currency is not supported."
                ),
                parameter_class=ConvertUnitsArgs,
            ),
        ]

    def execute(self, tool_name: str, args: BaseModel) -> str:
        if tool_name == "calculate":
            if not isinstance(args, CalculateArgs):
                raise TypeError("calculate requires CalculateArgs")
            result = self._calculate(args.expression)
        elif tool_name == "convert_units":
            if not isinstance(args, ConvertUnitsArgs):
                raise TypeError("convert_units requires ConvertUnitsArgs")
            result = self._convert_units(args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        return json.dumps(result, ensure_ascii=True, separators=(",", ":"))

    def interpret_result(self, tool_name: str, args: BaseModel, result: str) -> ReActToolOutcome | None:
        if tool_name not in {"calculate", "convert_units"}:
            return None
        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("status") == "ok":
            if tool_name == "calculate":
                summary = f"{payload.get('expression', '')} = {payload.get('value', '')}".strip()
            else:
                input_value = payload.get("input", {})
                summary = (
                    f"{input_value.get('value', '')} {input_value.get('unit', '')} = "
                    f"{payload.get('value', '')} {payload.get('unit', '')}"
                ).strip()
            return ReActToolOutcome(
                disposition=ToolOutcomeDisposition.SUCCESS,
                verified_summary=summary,
            )

        error = payload.get("error")
        if not isinstance(error, dict):
            return None
        code = error.get("code")
        message = error.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            return None
        if code == "ambiguous_unit":
            disposition = ToolOutcomeDisposition.CLARIFICATION_REQUIRED
        elif code in {
            "invalid_expression",
            "unsupported_syntax",
            "unknown_identifier",
            "unknown_function",
            "unknown_unit",
        }:
            disposition = ToolOutcomeDisposition.RETRYABLE_ERROR
        else:
            disposition = ToolOutcomeDisposition.TERMINAL_ERROR
        return ReActToolOutcome(disposition=disposition, code=code, message=message)

    def _calculate(self, expression: str) -> dict[str, object]:
        try:
            parser = _ExpressionParser(expression)
            with localcontext(self.DECIMAL_CONTEXT) as context:
                value = parser.parse()
                approximate = parser.intrinsically_approximate or context.flags[Inexact] or context.flags[Rounded]
            return {
                "status": "ok",
                "expression": parser.normalized_expression,
                "value": _decimal_text(value),
                "approximate": approximate,
            }
        except _MathToolError as exc:
            return _error_result(exc.code, exc.message)
        except (InvalidOperation, DivisionByZero):
            return _error_result("domain_error", "The operation is not defined for these values.")
        except Overflow:
            return _error_result("numeric_limit_exceeded", "The result exceeds the numeric limits.")

    def _convert_units(self, args: ConvertUnitsArgs) -> dict[str, object]:
        try:
            source = _resolve_unit(args.from_unit)
            target = _resolve_unit(args.to_unit)
            if source.dimension != target.dimension:
                raise _MathToolError(
                    "incompatible_units",
                    f"Cannot convert {source.dimension} to {target.dimension}.",
                )

            with localcontext(self.DECIMAL_CONTEXT):
                canonical = (args.value + source.shift) * source.numerator / source.denominator
                if source.dimension == "temperature" and canonical < _ABSOLUTE_ZERO_CELSIUS:
                    raise _MathToolError("below_absolute_zero", "The temperature is below absolute zero.")
                converted = canonical * target.denominator / target.numerator - target.shift
                if not converted.is_finite() or (converted != 0 and abs(converted.adjusted()) > 10_000):
                    raise _MathToolError(
                        "numeric_limit_exceeded",
                        "The result exceeds the numeric limits.",
                    )

            return {
                "status": "ok",
                "value": _decimal_text(converted),
                "unit": target.key,
                "symbol": target.symbol,
                "dimension": target.dimension,
                "input": {"value": _decimal_text(args.value), "unit": source.key},
            }
        except _MathToolError as exc:
            return _error_result(exc.code, exc.message)
        except (InvalidOperation, DivisionByZero):
            return _error_result("domain_error", "The conversion is not defined for this value.")
        except Overflow:
            return _error_result("numeric_limit_exceeded", "The result exceeds the numeric limits.")


def _resolve_unit(value: str) -> _Unit:
    normalized = _normalize_unit_name(value)
    if normalized in _AMBIGUOUS_UNITS:
        raise _MathToolError(
            "ambiguous_unit",
            f"Unit '{value}' is ambiguous; specify US or Imperial.",
        )
    unit = _UNIT_ALIASES.get(normalized)
    if unit is None:
        raise _MathToolError("unknown_unit", f"Unsupported unit '{value}'.")
    return unit


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    output_context = Context(prec=max(50, len(value.as_tuple().digits)), rounding=ROUND_HALF_EVEN)
    normalized = value.normalize(context=output_context)
    adjusted = normalized.adjusted()
    if -12 <= adjusted <= 50:
        return format(normalized, "f")
    mantissa, exponent = format(normalized, "E").split("E")
    return f"{mantissa}E{int(exponent):+d}"


def _error_result(code: str, message: str) -> dict[str, object]:
    return {"status": "error", "error": {"code": code, "message": message}}

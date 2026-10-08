"""Strict extraction of the last boxed answer and complete numeric answers."""

from decimal import Decimal, InvalidOperation
import math
import re

BOXED = "\\boxed{"
# Commas must be real thousands separators. A fraction/expression/unit suffix
# cannot be treated as a numeric prefix (for example, 12/2 must not become 12).
NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def _escaped(text: str, position: int) -> bool:
    n = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        n += 1
        position -= 1
    return n % 2 == 1


def last_boxed(text: str) -> str | None:
    """Return the final ``\\boxed{...}`` content, never an earlier fallback.

    Nested braces and escaped literal braces are supported. An incomplete final
    box invalidates the answer even when an earlier box was complete.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    start = text.rfind(BOXED)
    if start < 0:
        return None
    start += len(BOXED)
    depth = 1
    for pos in range(start, len(text)):
        if _escaped(text, pos):
            continue
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos]
    return None


def to_decimal(value: str | int | float | Decimal | None) -> Decimal | None:
    """Parse a full finite number, allowing currency and math dollar wrappers."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace("\\$", "$").replace("−", "-")
    # Accept $1,000, $1,000$, and -$1,000, with no arbitrary internal cleanup.
    if text.startswith("$") and text.endswith("$") and len(text) > 1:
        text = text[1:-1].strip()
    if text.startswith("$"):
        text = text[1:].strip()
    elif len(text) > 1 and text[0] in "+-" and text[1] == "$":
        text = text[0] + text[2:].strip()
    if NUMBER.fullmatch(text) is None:
        return None
    try:
        number = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def to_number(value: str | int | float | Decimal | None) -> float | None:
    number = to_decimal(value)
    if number is None:
        return None
    result = float(number)
    return result if math.isfinite(result) else None


def numeric_equal(prediction: str | int | float | Decimal | None,
                  gold: str | int | float | Decimal) -> bool:
    pred, target = to_decimal(prediction), to_decimal(gold)
    return pred is not None and target is not None and abs(pred - target) < Decimal("0.000001")


def gsm8k_gold(answer_field: str) -> float:
    if "####" not in answer_field:
        raise ValueError("GSM8K reference answer is missing ####")
    result = to_number(answer_field.rsplit("####", 1)[1])
    if result is None:
        raise ValueError("GSM8K reference answer is not a finite complete number")
    return result

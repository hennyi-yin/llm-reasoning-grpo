"""Conservative final-answer extraction for the follow-up, outside v1 scoring."""

import re

from .extract import last_boxed, to_decimal


MARKER = re.compile(
    r"(?im)(?:^\s*(?:\*\*)?(?:final\s+)?answer\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?"
    r"|\bfinal\s+answer\s*[:：]\s*(?:\*\*)?"
    r"|\b(?:the\s+)?(?:final\s+)?answer\s+(?:is|equals)\s+)"
)
DISPLAY = re.compile(r"(?s)(?:\\\[(.+?)\\\]|\$\$(.+?)\$\$)")
NUMERIC_SENTENCE = re.compile(
    r"^\s*([+\-−]?(?:\\?\$)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][+\-]?\d+)?)(?:\s+(?:[A-Za-z][A-Za-z'-]*\s*){1,6})?[.!]?\s*$"
)


def _clean(text: str) -> str:
    text = text.strip().removesuffix(".").strip().strip("*").strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            return text[len(left):-len(right)].strip()
    return text


def final_candidate(text: str, *, numeric: bool = False) -> tuple[str | None, str]:
    """Never search for a number in the working or use the reference answer."""
    markers = list(MARKER.finditer(text))
    marker = markers[-1] if markers else None
    box_start = text.rfind(r"\boxed{")
    displays = list(DISPLAY.finditer(text))
    display = displays[-1] if displays else None
    if display is not None and display.start() > box_start and not text[display.end():].strip(" .!\r\n\t"):
        display_text = _clean(next(x for x in display.groups() if x is not None))
    else:
        display_text = None
    if box_start >= 0 and (marker is None or box_start >= marker.start()):
        boxed = last_boxed(text)
        return (_clean(boxed), "boxed") if boxed and boxed.strip() else (None, "review")
    if marker is not None:
        tail = text[marker.end():].lstrip()
        marked_line = tail.splitlines()[0] if tail else ""
        if display_text and (not marked_line.strip() or marked_line.strip() in {r"\[", "$$"}):
            return display_text, "terminal_display"
        candidate = _clean(marked_line)
        if numeric and to_decimal(candidate) is None:
            match = NUMERIC_SENTENCE.fullmatch(candidate)
            if match and to_decimal(match[1]) is not None:
                return match[1], "explicit_final"
            return None, "review"
        if candidate:
            return candidate, "explicit_final"
    if display_text and (not numeric or to_decimal(display_text) is not None):
        return display_text, "terminal_display"
    last = _clean(next((line for line in reversed(text.splitlines()) if line.strip()), ""))
    if numeric and to_decimal(last) is not None:
        return last, "terminal_number"
    if not numeric and last and (last.startswith(r"\frac{") or to_decimal(last) is not None):
        return last, "terminal_expression"
    return None, "review"

"""English text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)

TEENS = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

SCALES = (
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
)

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class EnglishCardinalConfig:
    """Configuration surface for English cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = EnglishCardinalConfig()


def normalize_integer_token(
    token: str, config: EnglishCardinalConfig = DEFAULT_CONFIG
) -> str:
    value = parse_integer_token(token)
    return verbalize_integer(value, config=config)


def normalize_text(text: str) -> str:
    return INTEGER_TOKEN_RE.sub(_replace_integer_match, text)


def parse_integer_token(token: str) -> int:
    stripped = token.strip()
    if not stripped:
        raise ValueError("Empty token cannot be normalized.")
    if not stripped.isdigit():
        raise ValueError(f"Unsupported integer token: {token!r}")
    return int(stripped)


def verbalize_integer(
    value: int, config: EnglishCardinalConfig = DEFAULT_CONFIG
) -> str:
    if value == 0:
        return ONES[0]

    parts: list[str] = []
    scale_index = 0
    remaining = value

    while remaining > 0:
        triplet = remaining % 1000
        remaining //= 1000
        if scale_index >= config.max_scale_groups:
            raise ValueError(
                f"Value {value} exceeds configured English scale groups "
                f"({config.max_scale_groups})."
            )
        if triplet:
            parts.append(_verbalize_group(triplet, scale_index))
        scale_index += 1

    return " ".join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    words = _verbalize_triplet(group_value)
    scale_name = SCALES[scale_index]
    if not scale_name:
        return words
    return f"{words} {scale_name}"


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")
    if value < 10:
        return ONES[value]
    if value < 20:
        return TEENS[value]
    if value < 100:
        tens_value = (value // 10) * 10
        ones_digit = value % 10
        if ones_digit == 0:
            return TENS[tens_value]
        return f"{TENS[tens_value]} {ONES[ones_digit]}"

    hundreds_digit, remainder = divmod(value, 100)
    parts = [f"{ONES[hundreds_digit]} hundred"]
    if remainder:
        parts.append(_verbalize_triplet(remainder))
    return " ".join(parts)


def _verbalize_clock_minute(minute: int) -> str:
    if minute == 0:
        return "zero zero"
    if minute < 10:
        return f"zero {ONES[minute]}"
    return verbalize_integer(minute)


def _replace_integer_match(match: re.Match[str]) -> str:
    if _is_part_of_unsupported_grouped_number(match):
        return match.group("token")
    return normalize_integer_token(match.group("token"))


def _is_part_of_unsupported_grouped_number(match: re.Match[str]) -> bool:
    text = match.string
    start, end = match.span("token")
    if end + 1 < len(text) and text[end] in UNSUPPORTED_GROUP_SEPARATORS and text[end + 1].isdigit():
        return True
    if start > 1 and text[start - 1] in UNSUPPORTED_GROUP_SEPARATORS and text[start - 2].isdigit():
        return True
    return False


__all__ = [
    "DEFAULT_CONFIG",
    "EnglishCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

"""Polish text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "zero",
    "jeden",
    "dwa",
    "trzy",
    "cztery",
    "pięć",
    "sześć",
    "siedem",
    "osiem",
    "dziewięć",
)

TEENS = {
    10: "dziesięć",
    11: "jedenaście",
    12: "dwanaście",
    13: "trzynaście",
    14: "czternaście",
    15: "piętnaście",
    16: "szesnaście",
    17: "siedemnaście",
    18: "osiemnaście",
    19: "dziewiętnaście",
}

TENS = {
    20: "dwadzieścia",
    30: "trzydzieści",
    40: "czterdzieści",
    50: "pięćdziesiąt",
    60: "sześćdziesiąt",
    70: "siedemdziesiąt",
    80: "osiemdziesiąt",
    90: "dziewięćdziesiąt",
}

HUNDREDS = {
    100: "sto",
    200: "dwieście",
    300: "trzysta",
    400: "czterysta",
    500: "pięćset",
    600: "sześćset",
    700: "siedemset",
    800: "osiemset",
    900: "dziewięćset",
}

SCALES = (
    ("", "", ""),
    ("tysiąc", "tysiące", "tysięcy"),
    ("milion", "miliony", "milionów"),
    ("miliard", "miliardy", "miliardów"),
    ("bilion", "biliony", "bilionów"),
    ("biliard", "biliardy", "biliardów"),
    ("trylion", "tryliony", "trylionów"),
)

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class PolishCardinalConfig:
    """Configuration surface for Polish cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = PolishCardinalConfig()


def normalize_integer_token(
    token: str, config: PolishCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: PolishCardinalConfig = DEFAULT_CONFIG
) -> str:
    if value == 0:
        return ONES[0]

    parts: list[str] = []
    scale_index = 0
    remaining = value

    while remaining > 0:
        group = remaining % 1000
        remaining //= 1000
        if scale_index >= config.max_scale_groups:
            raise ValueError(
                f"Value {value} exceeds configured Polish scale groups "
                f"({config.max_scale_groups})."
            )
        if group:
            parts.append(_verbalize_group(group, scale_index))
        scale_index += 1

    return " ".join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    singular, paucal, plural = SCALES[scale_index]
    group_words = _verbalize_triplet(group_value)

    if scale_index == 0:
        return group_words
    if group_value == 1:
        return singular
    scale_word = _select_plural_form(group_value, singular, paucal, plural)
    return f"{group_words} {scale_word}"


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")
    parts: list[str] = []
    hundreds_value = (value // 100) * 100
    remainder = value % 100
    if hundreds_value:
        parts.append(HUNDREDS[hundreds_value])
    if remainder:
        if remainder < 10:
            parts.append(ONES[remainder])
        elif remainder < 20:
            parts.append(TEENS[remainder])
        else:
            tens_value = (remainder // 10) * 10
            ones_digit = remainder % 10
            parts.append(TENS[tens_value])
            if ones_digit:
                parts.append(ONES[ones_digit])
    return " ".join(parts)


def _select_plural_form(value: int, singular: str, paucal: str, plural: str) -> str:
    last_two = value % 100
    last_digit = value % 10
    if 12 <= last_two <= 14:
        return plural
    if last_digit == 1:
        return singular
    if 2 <= last_digit <= 4 and last_two not in (12, 13, 14):
        return paucal
    return plural


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
    "PolishCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

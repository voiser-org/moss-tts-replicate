"""Czech text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "nula",
    "jedna",
    "dva",
    "tři",
    "čtyři",
    "pět",
    "šest",
    "sedm",
    "osm",
    "devět",
)

TEENS = {
    10: "deset",
    11: "jedenáct",
    12: "dvanáct",
    13: "třináct",
    14: "čtrnáct",
    15: "patnáct",
    16: "šestnáct",
    17: "sedmnáct",
    18: "osmnáct",
    19: "devatenáct",
}

TENS = {
    20: "dvacet",
    30: "třicet",
    40: "čtyřicet",
    50: "padesát",
    60: "šedesát",
    70: "sedmdesát",
    80: "osmdesát",
    90: "devadesát",
}

HUNDREDS = {
    100: "sto",
    200: "dvě stě",
    300: "tři sta",
    400: "čtyři sta",
    500: "pět set",
    600: "šest set",
    700: "sedm set",
    800: "osm set",
    900: "devět set",
}

SCALES = (
    ("", "", ""),
    ("tisíc", "tisíce", "tisíc"),
    ("milion", "miliony", "milionů"),
    ("miliarda", "miliardy", "miliard"),
    ("bilion", "biliony", "bilionů"),
    ("biliarda", "biliardy", "biliard"),
    ("trilion", "triliony", "trilionů"),
)

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class CzechCardinalConfig:
    """Configuration surface for Czech cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = CzechCardinalConfig()


def normalize_integer_token(
    token: str, config: CzechCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: CzechCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured Czech scale groups "
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
    if 11 <= last_two <= 14:
        return plural
    if last_digit == 1:
        return singular
    if 2 <= last_digit <= 4:
        return paucal
    return plural


def _verbalize_clock_minute(minute: int) -> str:
    if minute == 0:
        return "nula nula"
    if minute < 10:
        return f"nula {ONES[minute]}"
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
    "CzechCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

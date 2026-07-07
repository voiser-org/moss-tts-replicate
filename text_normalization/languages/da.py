"""Danish text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "nul",
    "en",
    "to",
    "tre",
    "fire",
    "fem",
    "seks",
    "syv",
    "otte",
    "ni",
)

TEENS = {
    10: "ti",
    11: "elleve",
    12: "tolv",
    13: "tretten",
    14: "fjorten",
    15: "femten",
    16: "seksten",
    17: "sytten",
    18: "atten",
    19: "nitten",
}

TENS = {
    20: "tyve",
    30: "tredive",
    40: "fyrre",
    50: "halvtreds",
    60: "tres",
    70: "halvfjerds",
    80: "firs",
    90: "halvfems",
}

SCALES = (
    ("", ""),
    ("tusind", "tusind"),
    ("million", "millioner"),
    ("milliard", "milliarder"),
    ("billion", "billioner"),
    ("billiard", "billiarder"),
    ("trillion", "trillioner"),
)

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class DanishCardinalConfig:
    """Configuration surface for Danish cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = DanishCardinalConfig()


def normalize_integer_token(
    token: str, config: DanishCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: DanishCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured Danish scale groups "
                f"({config.max_scale_groups})."
            )
        if group:
            parts.append(_verbalize_group(group, scale_index))
        scale_index += 1

    return " ".join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    singular, plural = SCALES[scale_index]
    group_words = _verbalize_triplet(group_value)

    if scale_index == 0:
        return group_words
    if scale_index == 1:
        return singular if group_value == 1 else f"{group_words} {singular}"
    if group_value == 1:
        return f"en {singular}"
    return f"{group_words} {plural}"


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")
    if value < 100:
        return _verbalize_under_hundred(value)

    hundreds_digit, remainder = divmod(value, 100)
    if hundreds_digit == 1:
        prefix = "hundrede"
    else:
        prefix = f"{ONES[hundreds_digit]} hundrede"
    if remainder == 0:
        return prefix
    return f"{prefix} og {_verbalize_under_hundred(remainder)}"


def _verbalize_under_hundred(value: int) -> str:
    if value < 10:
        return ONES[value]
    if value < 20:
        return TEENS[value]
    tens_value = (value // 10) * 10
    ones_digit = value % 10
    if ones_digit == 0:
        return TENS[tens_value]
    return f"{ONES[ones_digit]}og{TENS[tens_value]}"


def _verbalize_clock_minute(minute: int) -> str:
    if minute == 0:
        return "nul nul"
    if minute < 10:
        return f"nul {ONES[minute]}"
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
    "DanishCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

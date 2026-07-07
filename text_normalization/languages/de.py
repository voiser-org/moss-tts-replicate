"""German text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = {
    0: "null",
    1: "eins",
    2: "zwei",
    3: "drei",
    4: "vier",
    5: "fünf",
    6: "sechs",
    7: "sieben",
    8: "acht",
    9: "neun",
}

TEENS = {
    10: "zehn",
    11: "elf",
    12: "zwölf",
    13: "dreizehn",
    14: "vierzehn",
    15: "fünfzehn",
    16: "sechzehn",
    17: "siebzehn",
    18: "achtzehn",
    19: "neunzehn",
}

TENS = {
    20: "zwanzig",
    30: "dreißig",
    40: "vierzig",
    50: "fünfzig",
    60: "sechzig",
    70: "siebzig",
    80: "achtzig",
    90: "neunzig",
}

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class GermanCardinalConfig:
    """Configuration surface for German cardinal normalization."""

    max_scale_groups: int = 7


DEFAULT_CONFIG = GermanCardinalConfig()


def normalize_integer_token(
    token: str, config: GermanCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: GermanCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured German scale groups "
                f"({config.max_scale_groups})."
            )

        if triplet:
            parts.append(_verbalize_group(triplet, scale_index))

        scale_index += 1

    return " ".join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    words = _verbalize_triplet(group_value)

    if scale_index == 0:
        return words
    if scale_index == 1:
        return "tausend" if group_value == 1 else f"{words}tausend"
    if scale_index == 2:
        return "eine Million" if group_value == 1 else f"{words} Millionen"
    if scale_index == 3:
        return "eine Milliarde" if group_value == 1 else f"{words} Milliarden"
    if scale_index == 4:
        return "eine Billion" if group_value == 1 else f"{words} Billionen"
    if scale_index == 5:
        return "eine Billiarde" if group_value == 1 else f"{words} Billiarden"
    if scale_index == 6:
        return "eine Trillion" if group_value == 1 else f"{words} Trillionen"
    raise ValueError(f"Unsupported scale index: {scale_index}")


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")
    if value < 100:
        return _verbalize_under_hundred(value, standalone=True)

    hundreds_digit, remainder = divmod(value, 100)
    if hundreds_digit == 1:
        prefix = "einhundert"
    else:
        prefix = f"{_one_word(hundreds_digit, standalone=False)}hundert"

    if remainder == 0:
        return prefix
    return f"{prefix}{_verbalize_under_hundred(remainder, standalone=True)}"


def _verbalize_under_hundred(value: int, standalone: bool) -> str:
    if value < 10:
        return _one_word(value, standalone=standalone)
    if value < 20:
        return TEENS[value]
    tens_value = (value // 10) * 10
    ones_digit = value % 10
    if ones_digit == 0:
        return TENS[tens_value]
    return f"{_one_word(ones_digit, standalone=False)}und{TENS[tens_value]}"


def _one_word(value: int, standalone: bool) -> str:
    if value == 1:
        return "eins" if standalone else "ein"
    return ONES[value]


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
    "GermanCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

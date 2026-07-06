"""Hungarian text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = {
    0: "nulla",
    1: "egy",
    2: "kettő",
    3: "három",
    4: "négy",
    5: "öt",
    6: "hat",
    7: "hét",
    8: "nyolc",
    9: "kilenc",
}

COMPOUND_ONES = {
    1: "egy",
    2: "két",
    3: "három",
    4: "négy",
    5: "öt",
    6: "hat",
    7: "hét",
    8: "nyolc",
    9: "kilenc",
}

TEENS = {
    10: "tíz",
    11: "tizenegy",
    12: "tizenkettő",
    13: "tizenhárom",
    14: "tizennégy",
    15: "tizenöt",
    16: "tizenhat",
    17: "tizenhét",
    18: "tizennyolc",
    19: "tizenkilenc",
}

TENS = {
    20: "húsz",
    30: "harminc",
    40: "negyven",
    50: "ötven",
    60: "hatvan",
    70: "hetven",
    80: "nyolcvan",
    90: "kilencven",
}

HUNDREDS = {
    1: "száz",
    2: "kétszáz",
    3: "háromszáz",
    4: "négyszáz",
    5: "ötszáz",
    6: "hatszáz",
    7: "hétszáz",
    8: "nyolcszáz",
    9: "kilencszáz",
}

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class HungarianCardinalConfig:
    """Configuration surface for Hungarian cardinal normalization."""

    max_scale_groups: int = 7


DEFAULT_CONFIG = HungarianCardinalConfig()


def normalize_integer_token(
    token: str, config: HungarianCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: HungarianCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured Hungarian scale groups "
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
        return "ezer" if group_value == 1 else f"{_verbalize_triplet(group_value, compound=True)}ezer"
    if scale_index == 2:
        return "egymillió" if group_value == 1 else f"{words} millió"
    if scale_index == 3:
        return "egymilliárd" if group_value == 1 else f"{words} milliárd"
    if scale_index == 4:
        return "egybillió" if group_value == 1 else f"{words} billió"
    if scale_index == 5:
        return "egybilliárd" if group_value == 1 else f"{words} billiárd"
    if scale_index == 6:
        return "egytrillió" if group_value == 1 else f"{words} trillió"
    raise ValueError(f"Unsupported scale index: {scale_index}")


def _verbalize_triplet(value: int, compound: bool = False) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")
    if value < 10:
        return COMPOUND_ONES[value] if compound and value in COMPOUND_ONES else ONES[value]
    if value < 20:
        return TEENS[value]
    if value < 100:
        tens_value = (value // 10) * 10
        ones_digit = value % 10
        if tens_value == 20 and ones_digit > 0:
            return f"huszon{COMPOUND_ONES[ones_digit]}"
        tens_word = TENS[tens_value]
        if ones_digit == 0:
            return tens_word
        return f"{tens_word}{COMPOUND_ONES[ones_digit]}"

    hundreds_digit, remainder = divmod(value, 100)
    hundreds_word = HUNDREDS[hundreds_digit]
    if remainder == 0:
        return hundreds_word
    return f"{hundreds_word}{_verbalize_triplet(remainder, compound=True)}"


def _verbalize_clock_minute(minute: int) -> str:
    if minute == 0:
        return "nulla nulla"
    if minute < 10:
        return f"nulla {ONES[minute]}"
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
    "HungarianCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

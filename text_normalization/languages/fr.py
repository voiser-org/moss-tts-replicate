"""French text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "zéro",
    "un",
    "deux",
    "trois",
    "quatre",
    "cinq",
    "six",
    "sept",
    "huit",
    "neuf",
)

TEENS = {
    10: "dix",
    11: "onze",
    12: "douze",
    13: "treize",
    14: "quatorze",
    15: "quinze",
    16: "seize",
    17: "dix-sept",
    18: "dix-huit",
    19: "dix-neuf",
}

TENS = {
    20: "vingt",
    30: "trente",
    40: "quarante",
    50: "cinquante",
    60: "soixante",
}

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class FrenchCardinalConfig:
    """Configuration surface for French cardinal normalization."""

    max_scale_groups: int = 7


DEFAULT_CONFIG = FrenchCardinalConfig()


def normalize_integer_token(
    token: str, config: FrenchCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: FrenchCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured French scale groups "
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
        return "mille" if group_value == 1 else f"{words} mille"
    if scale_index == 2:
        return "un million" if group_value == 1 else f"{words} millions"
    if scale_index == 3:
        return "un milliard" if group_value == 1 else f"{words} milliards"
    if scale_index == 4:
        return "un billion" if group_value == 1 else f"{words} billions"
    if scale_index == 5:
        return "un billiard" if group_value == 1 else f"{words} billiards"
    if scale_index == 6:
        return "un trillion" if group_value == 1 else f"{words} trillions"
    raise ValueError(f"Unsupported scale index: {scale_index}")


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")
    if value < 100:
        return _verbalize_under_hundred(value)
    if value == 100:
        return "cent"

    hundreds_digit, remainder = divmod(value, 100)
    prefix = "cent" if hundreds_digit == 1 else f"{ONES[hundreds_digit]} cent"
    if remainder:
        return f"{prefix} {_verbalize_under_hundred(remainder)}"
    return prefix


def _verbalize_under_hundred(value: int) -> str:
    if value < 10:
        return ONES[value]
    if value < 20:
        return TEENS[value]
    if value < 70:
        tens_value = (value // 10) * 10
        ones_digit = value % 10
        if ones_digit == 0:
            return TENS[tens_value]
        if ones_digit == 1:
            return f"{TENS[tens_value]} et un"
        return f"{TENS[tens_value]} {ONES[ones_digit]}"
    if value < 80:
        if value == 71:
            return "soixante et onze"
        return f"soixante {_verbalize_under_hundred(value - 60)}"
    if value == 80:
        return "quatre-vingts"
    if value == 81:
        return "quatre-vingt-un"
    return f"quatre-vingt {_verbalize_under_hundred(value - 80)}"


def _verbalize_clock_minute(minute: int) -> str:
    if minute == 0:
        return "zéro zéro"
    if minute < 10:
        return f"zéro {ONES[minute]}"
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
    "FrenchCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

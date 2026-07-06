"""Spanish text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "cero",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
)

SPECIALS = {
    10: "diez",
    11: "once",
    12: "doce",
    13: "trece",
    14: "catorce",
    15: "quince",
    16: "dieciséis",
    17: "diecisiete",
    18: "dieciocho",
    19: "diecinueve",
    20: "veinte",
    21: "veintiuno",
    22: "veintidós",
    23: "veintitrés",
    24: "veinticuatro",
    25: "veinticinco",
    26: "veintiséis",
    27: "veintisiete",
    28: "veintiocho",
    29: "veintinueve",
}

TENS = (
    "",
    "",
    "",
    "treinta",
    "cuarenta",
    "cincuenta",
    "sesenta",
    "setenta",
    "ochenta",
    "noventa",
)

HUNDREDS = {
    1: "ciento",
    2: "doscientos",
    3: "trescientos",
    4: "cuatrocientos",
    5: "quinientos",
    6: "seiscientos",
    7: "setecientos",
    8: "ochocientos",
    9: "novecientos",
}

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class SpanishCardinalConfig:
    """Configuration surface for Spanish cardinal normalization."""

    max_scale_groups: int = 7


DEFAULT_CONFIG = SpanishCardinalConfig()


def normalize_integer_token(
    token: str, config: SpanishCardinalConfig = DEFAULT_CONFIG
) -> str:
    """Normalize a standalone integer token like '2024' into Spanish text."""
    value = parse_integer_token(token)
    return verbalize_integer(value, config=config)


def normalize_text(text: str) -> str:
    """Normalize supported Spanish patterns inside arbitrary text."""
    return INTEGER_TOKEN_RE.sub(_replace_integer_match, text)


def parse_integer_token(token: str) -> int:
    stripped = token.strip()
    if not stripped:
        raise ValueError("Empty token cannot be normalized.")
    if not stripped.isdigit():
        raise ValueError(f"Unsupported integer token: {token!r}")
    return int(stripped)


def verbalize_integer(
    value: int, config: SpanishCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured Spanish scale groups "
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
        return "mil" if group_value == 1 else f"{words} mil"
    if scale_index == 2:
        return "un millón" if group_value == 1 else f"{words} millones"
    if scale_index == 3:
        return "mil millones" if group_value == 1 else f"{words} mil millones"
    if scale_index == 4:
        return "un billón" if group_value == 1 else f"{words} billones"
    if scale_index == 5:
        return "mil billones" if group_value == 1 else f"{words} mil billones"
    if scale_index == 6:
        return "un trillón" if group_value == 1 else f"{words} trillones"
    raise ValueError(f"Unsupported scale index: {scale_index}")


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")
    if value < 30:
        if value in SPECIALS:
            return SPECIALS[value]
        return ONES[value]
    if value < 100:
        tens_digit, ones_digit = divmod(value, 10)
        if ones_digit == 0:
            return TENS[tens_digit]
        return f"{TENS[tens_digit]} y {ONES[ones_digit]}"
    if value == 100:
        return "cien"

    hundreds_digit, remainder = divmod(value, 100)
    parts = [HUNDREDS[hundreds_digit]]
    if remainder:
        parts.append(_verbalize_triplet(remainder))
    return " ".join(parts)


def _verbalize_clock_minute(minute: int) -> str:
    if minute == 0:
        return "cero cero"
    if minute < 10:
        return f"cero {ONES[minute]}"
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
    "ONES",
    "SPECIALS",
    "TENS",
    "SpanishCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

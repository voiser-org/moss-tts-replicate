"""European Portuguese text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "zero",
    "um",
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
)

TEENS = {
    10: "dez",
    11: "onze",
    12: "doze",
    13: "treze",
    14: "catorze",
    15: "quinze",
    16: "dezasseis",
    17: "dezassete",
    18: "dezoito",
    19: "dezanove",
}

TENS = {
    20: "vinte",
    30: "trinta",
    40: "quarenta",
    50: "cinquenta",
    60: "sessenta",
    70: "setenta",
    80: "oitenta",
    90: "noventa",
}

HUNDREDS = {
    1: "cento",
    2: "duzentos",
    3: "trezentos",
    4: "quatrocentos",
    5: "quinhentos",
    6: "seiscentos",
    7: "setecentos",
    8: "oitocentos",
    9: "novecentos",
}

JOINER = " e "
INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class PortugueseCardinalConfig:
    """Configuration surface for Portuguese cardinal normalization."""

    max_scale_groups: int = 7


DEFAULT_CONFIG = PortugueseCardinalConfig()


def normalize_integer_token(
    token: str, config: PortugueseCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: PortugueseCardinalConfig = DEFAULT_CONFIG
) -> str:
    if value == 0:
        return ONES[0]

    group_pairs: list[tuple[int, str]] = []
    scale_index = 0
    remaining = value

    while remaining > 0:
        triplet = remaining % 1000
        remaining //= 1000

        if scale_index >= config.max_scale_groups:
            raise ValueError(
                f"Value {value} exceeds configured Portuguese scale groups "
                f"({config.max_scale_groups})."
            )

        if triplet:
            group_pairs.append((triplet, _verbalize_group(triplet, scale_index)))

        scale_index += 1

    ordered = list(reversed(group_pairs))
    result = ordered[0][1]
    for lower_value, lower_words in ordered[1:]:
        joiner = JOINER if lower_value < 100 else " "
        result = f"{result}{joiner}{lower_words}"
    return result


def _verbalize_group(group_value: int, scale_index: int) -> str:
    words = _verbalize_triplet(group_value)

    if scale_index == 0:
        return words
    if scale_index == 1:
        return "mil" if group_value == 1 else f"{words} mil"
    if scale_index == 2:
        return "um milhão" if group_value == 1 else f"{words} milhões"
    if scale_index == 3:
        return "mil milhões" if group_value == 1 else f"{words} mil milhões"
    if scale_index == 4:
        return "um bilião" if group_value == 1 else f"{words} biliões"
    if scale_index == 5:
        return "mil biliões" if group_value == 1 else f"{words} mil biliões"
    if scale_index == 6:
        return "um trilião" if group_value == 1 else f"{words} triliões"
    raise ValueError(f"Unsupported scale index: {scale_index}")


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
        return f"{TENS[tens_value]}{JOINER}{ONES[ones_digit]}"
    if value == 100:
        return "cem"

    hundreds_digit, remainder = divmod(value, 100)
    parts = [HUNDREDS[hundreds_digit]]
    if remainder:
        parts.append(_verbalize_triplet(remainder))
    return JOINER.join(parts)


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
    "ONES",
    "TEENS",
    "TENS",
    "PortugueseCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

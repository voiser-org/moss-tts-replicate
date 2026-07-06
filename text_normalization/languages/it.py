"""Italian text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "zero",
    "uno",
    "due",
    "tre",
    "quattro",
    "cinque",
    "sei",
    "sette",
    "otto",
    "nove",
)

TEENS = {
    10: "dieci",
    11: "undici",
    12: "dodici",
    13: "tredici",
    14: "quattordici",
    15: "quindici",
    16: "sedici",
    17: "diciassette",
    18: "diciotto",
    19: "diciannove",
}

TENS = {
    20: "venti",
    30: "trenta",
    40: "quaranta",
    50: "cinquanta",
    60: "sessanta",
    70: "settanta",
    80: "ottanta",
    90: "novanta",
}

HUNDREDS = {
    1: "cento",
    2: "duecento",
    3: "trecento",
    4: "quattrocento",
    5: "cinquecento",
    6: "seicento",
    7: "settecento",
    8: "ottocento",
    9: "novecento",
}

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class ItalianCardinalConfig:
    """Configuration surface for Italian cardinal normalization."""

    max_scale_groups: int = 7


DEFAULT_CONFIG = ItalianCardinalConfig()


def normalize_integer_token(
    token: str, config: ItalianCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: ItalianCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured Italian scale groups "
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
        return "mille" if group_value == 1 else f"{words}mila"
    if scale_index == 2:
        return "un milione" if group_value == 1 else f"{words} milioni"
    if scale_index == 3:
        return "un miliardo" if group_value == 1 else f"{words} miliardi"
    if scale_index == 4:
        return "un bilione" if group_value == 1 else f"{words} bilioni"
    if scale_index == 5:
        return "un biliardo" if group_value == 1 else f"{words} biliardi"
    if scale_index == 6:
        return "un trilione" if group_value == 1 else f"{words} trilioni"
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
        tens_word = TENS[tens_value]
        if ones_digit in {1, 8}:
            tens_word = tens_word[:-1]
        return tens_word if ones_digit == 0 else f"{tens_word}{ONES[ones_digit]}"

    hundreds_digit, remainder = divmod(value, 100)
    hundreds_word = HUNDREDS[hundreds_digit]
    if remainder and 80 <= remainder < 90 and hundreds_word.endswith("o"):
        hundreds_word = hundreds_word[:-1]
    if remainder == 0:
        return hundreds_word
    return f"{hundreds_word}{_verbalize_triplet(remainder)}"


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
    "ItalianCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

"""Russian text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
)

FEMININE_ONES = {
    1: "одна",
    2: "две",
}

TEENS = {
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    16: "шестнадцать",
    17: "семнадцать",
    18: "восемнадцать",
    19: "девятнадцать",
}

TENS = {
    20: "двадцать",
    30: "тридцать",
    40: "сорок",
    50: "пятьдесят",
    60: "шестьдесят",
    70: "семьдесят",
    80: "восемьдесят",
    90: "девяносто",
}

HUNDREDS = {
    100: "сто",
    200: "двести",
    300: "триста",
    400: "четыреста",
    500: "пятьсот",
    600: "шестьсот",
    700: "семьсот",
    800: "восемьсот",
    900: "девятьсот",
}

SCALES = (
    ("", "", "", False),
    ("тысяча", "тысячи", "тысяч", True),
    ("миллион", "миллиона", "миллионов", False),
    ("миллиард", "миллиарда", "миллиардов", False),
    ("триллион", "триллиона", "триллионов", False),
    ("квадриллион", "квадриллиона", "квадриллионов", False),
    ("квинтиллион", "квинтиллиона", "квинтиллионов", False),
)

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class RussianCardinalConfig:
    """Configuration surface for Russian cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = RussianCardinalConfig()


def normalize_integer_token(
    token: str, config: RussianCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: RussianCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured Russian scale groups "
                f"({config.max_scale_groups})."
            )
        if group:
            parts.append(_verbalize_group(group, scale_index))
        scale_index += 1

    return " ".join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    singular, paucal, plural, feminine = SCALES[scale_index]
    group_words = _verbalize_triplet(group_value, feminine=feminine)

    if scale_index == 0:
        return group_words

    scale_word = _select_plural_form(group_value, singular, paucal, plural)
    return f"{group_words} {scale_word}"


def _verbalize_triplet(value: int, feminine: bool = False) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")

    parts: list[str] = []
    hundreds_value = (value // 100) * 100
    remainder = value % 100

    if hundreds_value:
        parts.append(HUNDREDS[hundreds_value])

    if remainder:
        if remainder < 10:
            parts.append(_ones_word(remainder, feminine=feminine))
        elif remainder < 20:
            parts.append(TEENS[remainder])
        else:
            tens_value = (remainder // 10) * 10
            ones_digit = remainder % 10
            parts.append(TENS[tens_value])
            if ones_digit:
                parts.append(_ones_word(ones_digit, feminine=feminine))

    return " ".join(parts)


def _ones_word(value: int, feminine: bool = False) -> str:
    if feminine and value in FEMININE_ONES:
        return FEMININE_ONES[value]
    return ONES[value]


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
        return "ноль ноль"
    if minute < 10:
        return f"ноль {ONES[minute]}"
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
    "RussianCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

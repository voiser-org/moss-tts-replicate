"""Arabic text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "صفر",
    "واحد",
    "اثنان",
    "ثلاثة",
    "أربعة",
    "خمسة",
    "ستة",
    "سبعة",
    "ثمانية",
    "تسعة",
)

TEENS = {
    10: "عشرة",
    11: "أحد عشر",
    12: "اثنا عشر",
    13: "ثلاثة عشر",
    14: "أربعة عشر",
    15: "خمسة عشر",
    16: "ستة عشر",
    17: "سبعة عشر",
    18: "ثمانية عشر",
    19: "تسعة عشر",
}

TENS = {
    20: "عشرون",
    30: "ثلاثون",
    40: "أربعون",
    50: "خمسون",
    60: "ستون",
    70: "سبعون",
    80: "ثمانون",
    90: "تسعون",
}

HUNDREDS = {
    100: "مائة",
    200: "مائتان",
    300: "ثلاثمائة",
    400: "أربعمائة",
    500: "خمسمائة",
    600: "ستمائة",
    700: "سبعمائة",
    800: "ثمانمائة",
    900: "تسعمائة",
}

SCALES = (
    ("", "", "", ""),
    ("ألف", "ألفان", "آلاف", "ألف"),
    ("مليون", "مليونان", "ملايين", "مليون"),
    ("مليار", "ملياران", "مليارات", "مليار"),
    ("تريليون", "تريليونان", "تريليونات", "تريليون"),
    ("كوادريليون", "كوادريليونان", "كوادريليونات", "كوادريليون"),
    ("كوينتليون", "كوينتليونان", "كوينتليونات", "كوينتليون"),
)

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", "،", "٬", ".", " ", "_"}
JOINER = " و "


@dataclass(frozen=True)
class ArabicCardinalConfig:
    """Configuration surface for Arabic cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = ArabicCardinalConfig()


def normalize_integer_token(
    token: str, config: ArabicCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: ArabicCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured Arabic scale groups "
                f"({config.max_scale_groups})."
            )
        if group:
            parts.append(_verbalize_group(group, scale_index))
        scale_index += 1

    return JOINER.join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    singular, dual, plural, singular_after_number = SCALES[scale_index]
    group_words = _verbalize_triplet(group_value)

    if scale_index == 0:
        return group_words
    if group_value == 1:
        return singular
    if group_value == 2:
        return dual

    last_two = group_value % 100
    if 3 <= last_two <= 10:
        return f"{group_words} {plural}"
    return f"{group_words} {singular_after_number}"


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
        return f"{ONES[ones_digit]} و{TENS[tens_value]}"

    hundreds_value = (value // 100) * 100
    remainder = value % 100
    if remainder == 0:
        return HUNDREDS[hundreds_value]
    return f"{HUNDREDS[hundreds_value]} و{_verbalize_triplet(remainder)}"


def _verbalize_clock_minute(minute: int) -> str:
    if minute == 0:
        return "صفر صفر"
    if minute < 10:
        return f"صفر {ONES[minute]}"
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
    "ArabicCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

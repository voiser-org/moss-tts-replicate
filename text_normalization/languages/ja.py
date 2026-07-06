"""Japanese text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

KANJI_DIGITS = (
    "零",
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
)

SMALL_UNITS = (
    (1000, "千"),
    (100, "百"),
    (10, "十"),
)

LARGE_UNITS = (
    "",
    "万",
    "億",
    "兆",
    "京",
    "垓",
)

DIGIT_CLASS = "0-9０-９"
INTEGER_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z{DIGIT_CLASS}:+-])(?P<token>[{DIGIT_CLASS}]+)(?![A-Za-z{DIGIT_CLASS}:：])"
)
UNSUPPORTED_GROUP_SEPARATORS = {",", "，", ".", " ", "_"}

_DIGIT_TRANSLATION = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
    }
)


@dataclass(frozen=True)
class JapaneseCardinalConfig:
    """Configuration surface for Japanese cardinal normalization."""

    max_scale_groups: int = len(LARGE_UNITS)


DEFAULT_CONFIG = JapaneseCardinalConfig()


def normalize_integer_token(
    token: str, config: JapaneseCardinalConfig = DEFAULT_CONFIG
) -> str:
    value = parse_integer_token(token)
    return verbalize_integer(value, config=config)


def normalize_text(text: str) -> str:
    return INTEGER_TOKEN_RE.sub(_replace_integer_match, text)


def parse_integer_token(token: str) -> int:
    stripped = _normalize_digits(token.strip())
    if not stripped:
        raise ValueError("Empty token cannot be normalized.")
    if not stripped.isdigit():
        raise ValueError(f"Unsupported integer token: {token!r}")
    return int(stripped)


def verbalize_integer(
    value: int, config: JapaneseCardinalConfig = DEFAULT_CONFIG
) -> str:
    if value == 0:
        return KANJI_DIGITS[0]

    groups: list[str] = []
    scale_index = 0
    remaining = value

    while remaining > 0:
        group = remaining % 10000
        remaining //= 10000

        if scale_index >= config.max_scale_groups:
            raise ValueError(
                f"Value {value} exceeds configured Japanese scale groups "
                f"({config.max_scale_groups})."
            )

        if group:
            groups.append(f"{_verbalize_group(group)}{LARGE_UNITS[scale_index]}")

        scale_index += 1

    return "".join(reversed(groups))


def _verbalize_group(value: int) -> str:
    if not 0 < value < 10000:
        raise ValueError(f"Group out of range: {value}")

    parts: list[str] = []
    remaining = value

    for unit_value, unit_word in SMALL_UNITS:
        digit, remaining = divmod(remaining, unit_value)
        if digit == 0:
            continue
        if digit == 1:
            parts.append(unit_word)
        else:
            parts.append(f"{KANJI_DIGITS[digit]}{unit_word}")

    if remaining:
        parts.append(KANJI_DIGITS[remaining])

    return "".join(parts)


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


def _normalize_digits(value: str) -> str:
    return value.translate(_DIGIT_TRANSLATION)


__all__ = [
    "DEFAULT_CONFIG",
    "JapaneseCardinalConfig",
    "KANJI_DIGITS",
    "UNSUPPORTED_GROUP_SEPARATORS",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

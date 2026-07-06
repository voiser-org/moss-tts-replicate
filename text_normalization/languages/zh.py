"""Chinese text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

DIGITS = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")
SMALL_UNITS = ((1000, "千"), (100, "百"), (10, "十"))
LARGE_UNITS = ("", "万", "亿", "兆", "京", "垓")

DIGIT_CLASS = "0-9０-９"
INTEGER_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z{DIGIT_CLASS}:+-])(?P<token>[{DIGIT_CLASS}]+)(?![A-Za-z{DIGIT_CLASS}:：])"
)
UNSUPPORTED_GROUP_SEPARATORS = {",", "，", ".", " ", "_"}

_DIGIT_TRANSLATION = str.maketrans(
    {"０": "0", "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6", "７": "7", "８": "8", "９": "9"}
)


@dataclass(frozen=True)
class ChineseCardinalConfig:
    """Configuration surface for Chinese cardinal normalization."""

    max_scale_groups: int = len(LARGE_UNITS)


DEFAULT_CONFIG = ChineseCardinalConfig()


def normalize_integer_token(
    token: str, config: ChineseCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: ChineseCardinalConfig = DEFAULT_CONFIG
) -> str:
    if value == 0:
        return DIGITS[0]

    groups: list[int] = []
    remaining = value
    while remaining > 0:
        groups.append(remaining % 10000)
        remaining //= 10000

    if len(groups) > config.max_scale_groups:
        raise ValueError(
            f"Value {value} exceeds configured Chinese scale groups "
            f"({config.max_scale_groups})."
        )

    ordered = list(reversed(groups))
    result: list[str] = []
    for idx, group in enumerate(ordered):
        scale_index = len(ordered) - idx - 1
        if group == 0:
            continue

        lower_groups = ordered[idx + 1 :]
        if result and (group < 1000 or any(next_group == 0 for next_group in lower_groups[:-1])):
            if result[-1] != DIGITS[0]:
                result.append(DIGITS[0])

        omit_one_ten = idx == 0 and len(ordered) == 1
        result.append(f"{_verbalize_group(group, omit_leading_one_ten=omit_one_ten)}{LARGE_UNITS[scale_index]}")

    spoken = "".join(result)
    return spoken.rstrip(DIGITS[0])


def _verbalize_group(value: int, omit_leading_one_ten: bool) -> str:
    if not 0 < value < 10000:
        raise ValueError(f"Group out of range: {value}")

    parts: list[str] = []
    remaining = value
    zero_pending = False

    for unit_value, unit_word in SMALL_UNITS:
        digit, remaining = divmod(remaining, unit_value)
        if digit == 0:
            if parts and remaining > 0:
                zero_pending = True
            continue

        if zero_pending:
            parts.append(DIGITS[0])
            zero_pending = False

        if unit_value == 10 and digit == 1 and not parts and omit_leading_one_ten:
            parts.append(unit_word)
        else:
            parts.append(f"{DIGITS[digit]}{unit_word}")

    if remaining:
        if zero_pending:
            parts.append(DIGITS[0])
        parts.append(DIGITS[remaining])

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
    "ChineseCardinalConfig",
    "DIGITS",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

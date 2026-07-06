"""Persian (Farsi) text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "صفر",
    "یک",
    "دو",
    "سه",
    "چهار",
    "پنج",
    "شش",
    "هفت",
    "هشت",
    "نه",
)

TEENS = (
    "ده",
    "یازده",
    "دوازده",
    "سیزده",
    "چهارده",
    "پانزده",
    "شانزده",
    "هفده",
    "هجده",
    "نوزده",
)

TENS = (
    "",
    "",
    "بیست",
    "سی",
    "چهل",
    "پنجاه",
    "شصت",
    "هفتاد",
    "هشتاد",
    "نود",
)

HUNDREDS = (
    "",
    "صد",
    "دویست",
    "سیصد",
    "چهارصد",
    "پانصد",
    "ششصد",
    "هفتصد",
    "هشتصد",
    "نهصد",
)

SCALES = (
    "",
    "هزار",
    "میلیون",
    "میلیارد",
    "تریلیون",
    "کوادریلیون",
    "کوینتیلیون",
)

JOINER = " و "
DIGIT_CLASS = "0-9۰-۹٠-٩"
INTEGER_TOKEN_RE = re.compile(
    rf"(?<![\w:+-])(?P<token>[{DIGIT_CLASS}]+)(?![\w:])"
)
UNSUPPORTED_GROUP_SEPARATORS = {",", "٬", ".", " ", "_"}

_DIGIT_TRANSLATION = str.maketrans(
    {
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


@dataclass(frozen=True)
class PersianCardinalConfig:
    """Configuration surface for Persian cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = PersianCardinalConfig()


def normalize_integer_token(
    token: str, config: PersianCardinalConfig = DEFAULT_CONFIG
) -> str:
    """Normalize a standalone integer token like '۲۰۲۴' into Persian text."""
    value = parse_integer_token(token)
    return verbalize_integer(value, config=config)


def normalize_text(text: str) -> str:
    """Normalize supported Persian patterns inside arbitrary text."""
    return INTEGER_TOKEN_RE.sub(_replace_integer_match, text)


def parse_integer_token(token: str) -> int:
    """Parse a Persian cardinal integer token."""
    stripped = token.strip()
    if not stripped:
        raise ValueError("Empty token cannot be normalized.")

    stripped = _normalize_digits(stripped)

    if "٫" in stripped or "." in stripped:
        raise ValueError("Decimal numbers are out of scope for Persian cardinal rules.")

    if not stripped.isdigit():
        raise ValueError(f"Unsupported integer token: {token!r}")

    return int(stripped)


def verbalize_integer(
    value: int, config: PersianCardinalConfig = DEFAULT_CONFIG
) -> str:
    """Convert an integer value into Persian cardinal text."""
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
                f"Value {value} exceeds configured Persian scale groups "
                f"({config.max_scale_groups})."
            )

        if triplet:
            parts.append(_verbalize_group(triplet, scale_index))

        scale_index += 1

    return JOINER.join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    words = _verbalize_triplet(group_value)
    scale_name = SCALES[scale_index]

    if not scale_name:
        return words

    if scale_name == "هزار" and group_value == 1:
        return scale_name

    return f"{words} {scale_name}"


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")

    if value < 10:
        return ONES[value]
    if value < 20:
        return TEENS[value - 10]
    if value < 100:
        tens_digit, ones_digit = divmod(value, 10)
        parts = [TENS[tens_digit]]
        if ones_digit:
            parts.append(ONES[ones_digit])
        return JOINER.join(parts)

    hundreds_digit, remainder = divmod(value, 100)
    parts = [HUNDREDS[hundreds_digit]]
    if remainder:
        parts.append(_verbalize_triplet(remainder))
    return JOINER.join(parts)


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
    "HUNDREDS",
    "JOINER",
    "ONES",
    "SCALES",
    "TENS",
    "TEENS",
    "UNSUPPORTED_GROUP_SEPARATORS",
    "PersianCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

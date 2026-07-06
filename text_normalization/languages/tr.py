"""Turkish text normalization rules for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "sıfır",
    "bir",
    "iki",
    "üç",
    "dört",
    "beş",
    "altı",
    "yedi",
    "sekiz",
    "dokuz",
)

TENS = (
    "",
    "on",
    "yirmi",
    "otuz",
    "kırk",
    "elli",
    "altmış",
    "yetmiş",
    "seksen",
    "doksan",
)

SCALES = (
    "",
    "bin",
    "milyon",
    "milyar",
    "trilyon",
    "katrilyon",
    "kentilyon",
)

INTEGER_TOKEN_RE = re.compile(
    r"(?<![\w:+-])(?P<token>\d+)(?![\w:])"
)
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class TurkishCardinalConfig:
    """Configuration surface for Turkish cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = TurkishCardinalConfig()


def normalize_integer_token(
    token: str, config: TurkishCardinalConfig = DEFAULT_CONFIG
) -> str:
    """Normalize a standalone integer token like '2.024' into Turkish text."""
    value = parse_integer_token(token)
    return verbalize_integer(value, config=config)


def normalize_text(text: str) -> str:
    """Normalize supported Turkish patterns inside arbitrary text.

    Current scope:
    - standalone integer tokens such as 2024
    """
    return INTEGER_TOKEN_RE.sub(_replace_integer_match, text)


def parse_integer_token(token: str) -> int:
    """Parse a cardinal integer token used by the normalizer layer.

    Supported examples:
    - 2024

    Out of scope examples:
    - 3,14
    - 2.024
    - 12/05
    - A320
    """

    stripped = token.strip()
    if not stripped:
        raise ValueError("Empty token cannot be normalized.")

    if "," in stripped or "." in stripped:
        raise ValueError("Decimal numbers are out of scope for Turkish cardinal rules.")

    if not stripped.isdigit():
        raise ValueError(f"Unsupported integer token: {token!r}")

    return int(stripped)


def verbalize_integer(
    value: int, config: TurkishCardinalConfig = DEFAULT_CONFIG
) -> str:
    """Convert an integer value into Turkish cardinal text."""

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
                f"Value {value} exceeds configured Turkish scale groups "
                f"({config.max_scale_groups})."
            )

        if triplet:
            parts.append(_verbalize_group(triplet, scale_index))

        scale_index += 1

    return " ".join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    words = _verbalize_triplet(group_value)
    scale_name = SCALES[scale_index]

    if not scale_name:
        return words

    if scale_name == "bin" and group_value == 1:
        return scale_name

    return f"{words} {scale_name}"


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")

    hundreds, remainder = divmod(value, 100)
    tens_digit, ones_digit = divmod(remainder, 10)

    parts: list[str] = []

    if hundreds == 1:
        parts.append("yüz")
    elif hundreds > 1:
        parts.append(ONES[hundreds])
        parts.append("yüz")

    if tens_digit:
        parts.append(TENS[tens_digit])

    if ones_digit:
        parts.append(ONES[ones_digit])

    return " ".join(parts)


def _verbalize_clock_minute(minute: int) -> str:
    if not 0 <= minute <= 59:
        raise ValueError(f"Minute out of range: {minute}")

    if minute == 0:
        return "sıfır sıfır"

    if minute < 10:
        return f"sıfır {ONES[minute]}"

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
    "SCALES",
    "TENS",
    "UNSUPPORTED_GROUP_SEPARATORS",
    "TurkishCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

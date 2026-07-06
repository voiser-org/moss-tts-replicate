"""Swedish text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "noll",
    "ett",
    "två",
    "tre",
    "fyra",
    "fem",
    "sex",
    "sju",
    "åtta",
    "nio",
)

TEENS = (
    "tio",
    "elva",
    "tolv",
    "tretton",
    "fjorton",
    "femton",
    "sexton",
    "sjutton",
    "arton",
    "nitton",
)

TENS = (
    "",
    "",
    "tjugo",
    "trettio",
    "fyrtio",
    "femtio",
    "sextio",
    "sjuttio",
    "åttio",
    "nittio",
)

SCALES = (
    ("", ""),
    ("tusen", "tusen"),
    ("miljon", "miljoner"),
    ("miljard", "miljarder"),
    ("biljon", "biljoner"),
    ("biljard", "biljarder"),
    ("triljon", "triljoner"),
)

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class SwedishCardinalConfig:
    """Configuration surface for Swedish cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = SwedishCardinalConfig()


def normalize_integer_token(
    token: str, config: SwedishCardinalConfig = DEFAULT_CONFIG
) -> str:
    """Normalize a standalone integer token like '2024' into Swedish text."""
    value = parse_integer_token(token)
    return verbalize_integer(value, config=config)


def normalize_text(text: str) -> str:
    """Normalize supported Swedish patterns inside arbitrary text."""
    return INTEGER_TOKEN_RE.sub(_replace_integer_match, text)


def parse_integer_token(token: str) -> int:
    """Parse a Swedish cardinal integer token."""
    stripped = token.strip()
    if not stripped:
        raise ValueError("Empty token cannot be normalized.")

    if not stripped.isdigit():
        raise ValueError(f"Unsupported integer token: {token!r}")

    return int(stripped)


def verbalize_integer(
    value: int, config: SwedishCardinalConfig = DEFAULT_CONFIG
) -> str:
    """Convert an integer value into Swedish cardinal text."""
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
                f"Value {value} exceeds configured Swedish scale groups "
                f"({config.max_scale_groups})."
            )

        if triplet:
            parts.append(_verbalize_group(triplet, scale_index))

        scale_index += 1

    return " ".join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    words = _verbalize_triplet(group_value)
    singular, plural = SCALES[scale_index]

    if not singular:
        return words

    if scale_index == 1:
        if group_value == 1:
            return singular
        return f"{words} {singular}"

    if group_value == 1:
        return f"en {singular}"

    return f"{words} {plural}"


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")

    if value < 10:
        return ONES[value]

    if value < 20:
        return TEENS[value - 10]

    if value < 100:
        tens_digit, ones_digit = divmod(value, 10)
        if ones_digit == 0:
            return TENS[tens_digit]
        return f"{TENS[tens_digit]}{ONES[ones_digit]}"

    hundreds_digit, remainder = divmod(value, 100)
    parts: list[str] = []

    if hundreds_digit == 1:
        parts.append("hundra")
    else:
        parts.append(ONES[hundreds_digit])
        parts.append("hundra")

    if remainder:
        parts.append(_verbalize_triplet(remainder))

    return " ".join(parts)


def _verbalize_clock_minute(minute: int) -> str:
    if not 0 <= minute <= 59:
        raise ValueError(f"Minute out of range: {minute}")

    if minute == 0:
        return "noll noll"

    if minute < 10:
        return f"noll {ONES[minute]}"

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
    "TEENS",
    "SwedishCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

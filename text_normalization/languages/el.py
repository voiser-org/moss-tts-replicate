"""Greek text normalization for standalone integer tokens."""

from __future__ import annotations

from dataclasses import dataclass
import re

ONES = (
    "μηδέν",
    "ένα",
    "δύο",
    "τρία",
    "τέσσερα",
    "πέντε",
    "έξι",
    "επτά",
    "οκτώ",
    "εννέα",
)

TEENS = {
    10: "δέκα",
    11: "έντεκα",
    12: "δώδεκα",
    13: "δεκατρία",
    14: "δεκατέσσερα",
    15: "δεκαπέντε",
    16: "δεκαέξι",
    17: "δεκαεπτά",
    18: "δεκαοκτώ",
    19: "δεκαεννέα",
}

TENS = {
    20: "είκοσι",
    30: "τριάντα",
    40: "σαράντα",
    50: "πενήντα",
    60: "εξήντα",
    70: "εβδομήντα",
    80: "ογδόντα",
    90: "ενενήντα",
}

HUNDREDS = {
    100: "εκατό",
    200: "διακόσια",
    300: "τριακόσια",
    400: "τετρακόσια",
    500: "πεντακόσια",
    600: "εξακόσια",
    700: "επτακόσια",
    800: "οκτακόσια",
    900: "εννιακόσια",
}

SCALES = (
    ("", ""),
    ("χίλια", "χιλιάδες"),
    ("εκατομμύριο", "εκατομμύρια"),
    ("δισεκατομμύριο", "δισεκατομμύρια"),
    ("τρισεκατομμύριο", "τρισεκατομμύρια"),
    ("τετρασεκατομμύριο", "τετρασεκατομμύρια"),
    ("πεντασεκατομμύριο", "πεντασεκατομμύρια"),
)

INTEGER_TOKEN_RE = re.compile(r"(?<![\w:+-])(?P<token>\d+)(?![\w:])")
UNSUPPORTED_GROUP_SEPARATORS = {",", ".", " ", "_"}


@dataclass(frozen=True)
class GreekCardinalConfig:
    """Configuration surface for Greek cardinal normalization."""

    max_scale_groups: int = len(SCALES)


DEFAULT_CONFIG = GreekCardinalConfig()


def normalize_integer_token(
    token: str, config: GreekCardinalConfig = DEFAULT_CONFIG
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
    value: int, config: GreekCardinalConfig = DEFAULT_CONFIG
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
                f"Value {value} exceeds configured Greek scale groups "
                f"({config.max_scale_groups})."
            )
        if group:
            parts.append(_verbalize_group(group, scale_index))
        scale_index += 1

    return " ".join(reversed(parts))


def _verbalize_group(group_value: int, scale_index: int) -> str:
    singular, plural = SCALES[scale_index]
    group_words = _verbalize_triplet(group_value)

    if scale_index == 0:
        return group_words
    if scale_index == 1:
        return singular if group_value == 1 else f"{group_words} {plural}"
    if group_value == 1:
        return f"ένα {singular}"
    return f"{group_words} {plural}"


def _verbalize_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError(f"Triplet out of range: {value}")
    parts: list[str] = []
    hundreds_value = (value // 100) * 100
    remainder = value % 100
    if hundreds_value:
        parts.append(HUNDREDS[hundreds_value])
    if remainder:
        if remainder < 10:
            parts.append(ONES[remainder])
        elif remainder < 20:
            parts.append(TEENS[remainder])
        else:
            tens_value = (remainder // 10) * 10
            ones_digit = remainder % 10
            parts.append(TENS[tens_value])
            if ones_digit:
                parts.append(ONES[ones_digit])
    return " ".join(parts)


def _verbalize_clock_minute(minute: int) -> str:
    if minute == 0:
        return "μηδέν μηδέν"
    if minute < 10:
        return f"μηδέν {ONES[minute]}"
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
    "GreekCardinalConfig",
    "normalize_integer_token",
    "normalize_text",
    "parse_integer_token",
    "verbalize_integer",
]

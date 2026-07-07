"""Per-language text normalization modules."""

from collections.abc import Callable

from .ar import (
    normalize_integer_token as normalize_arabic_integer_token,
    normalize_text as normalize_arabic_text,
)
from .cs import (
    normalize_integer_token as normalize_czech_integer_token,
    normalize_text as normalize_czech_text,
)
from .da import (
    normalize_integer_token as normalize_danish_integer_token,
    normalize_text as normalize_danish_text,
)
from .de import (
    normalize_integer_token as normalize_german_integer_token,
    normalize_text as normalize_german_text,
)
from .el import (
    normalize_integer_token as normalize_greek_integer_token,
    normalize_text as normalize_greek_text,
)
from .en import (
    normalize_integer_token as normalize_english_integer_token,
    normalize_text as normalize_english_text,
)
from .es import (
    normalize_integer_token as normalize_spanish_integer_token,
    normalize_text as normalize_spanish_text,
)
from .fa import (
    normalize_integer_token as normalize_persian_integer_token,
    normalize_text as normalize_persian_text,
)
from .fr import (
    normalize_integer_token as normalize_french_integer_token,
    normalize_text as normalize_french_text,
)
from .hu import (
    normalize_integer_token as normalize_hungarian_integer_token,
    normalize_text as normalize_hungarian_text,
)
from .it import (
    normalize_integer_token as normalize_italian_integer_token,
    normalize_text as normalize_italian_text,
)
from .ja import (
    normalize_integer_token as normalize_japanese_integer_token,
    normalize_text as normalize_japanese_text,
)
from .ko import (
    normalize_integer_token as normalize_korean_integer_token,
    normalize_text as normalize_korean_text,
)
from .pl import (
    normalize_integer_token as normalize_polish_integer_token,
    normalize_text as normalize_polish_text,
)
from .pt import (
    normalize_integer_token as normalize_portuguese_integer_token,
    normalize_text as normalize_portuguese_text,
)
from .ru import (
    normalize_integer_token as normalize_russian_integer_token,
    normalize_text as normalize_russian_text,
)
from .sv import (
    normalize_integer_token as normalize_swedish_integer_token,
    normalize_text as normalize_swedish_text,
)
from .tr import (
    normalize_integer_token as normalize_turkish_integer_token,
    normalize_text as normalize_turkish_text,
)
from .zh import (
    normalize_integer_token as normalize_chinese_integer_token,
    normalize_text as normalize_chinese_text,
)

IntegerTokenNormalizer = Callable[[str], str]
TextNormalizer = Callable[[str], str]

_INTEGER_TOKEN_NORMALIZERS: dict[str, IntegerTokenNormalizer] = {
    "ar": normalize_arabic_integer_token,
    "cs": normalize_czech_integer_token,
    "da": normalize_danish_integer_token,
    "de": normalize_german_integer_token,
    "el": normalize_greek_integer_token,
    "en": normalize_english_integer_token,
    "es": normalize_spanish_integer_token,
    "fa": normalize_persian_integer_token,
    "fr": normalize_french_integer_token,
    "hu": normalize_hungarian_integer_token,
    "it": normalize_italian_integer_token,
    "ja": normalize_japanese_integer_token,
    "ko": normalize_korean_integer_token,
    "pl": normalize_polish_integer_token,
    "pt": normalize_portuguese_integer_token,
    "ru": normalize_russian_integer_token,
    "sv": normalize_swedish_integer_token,
    "tr": normalize_turkish_integer_token,
    "zh": normalize_chinese_integer_token,
}

_TEXT_NORMALIZERS: dict[str, TextNormalizer] = {
    "ar": normalize_arabic_text,
    "cs": normalize_czech_text,
    "da": normalize_danish_text,
    "de": normalize_german_text,
    "el": normalize_greek_text,
    "en": normalize_english_text,
    "es": normalize_spanish_text,
    "fa": normalize_persian_text,
    "fr": normalize_french_text,
    "hu": normalize_hungarian_text,
    "it": normalize_italian_text,
    "ja": normalize_japanese_text,
    "ko": normalize_korean_text,
    "pl": normalize_polish_text,
    "pt": normalize_portuguese_text,
    "ru": normalize_russian_text,
    "sv": normalize_swedish_text,
    "tr": normalize_turkish_text,
    "zh": normalize_chinese_text,
}


def get_integer_token_normalizer(language: str) -> IntegerTokenNormalizer:
    try:
        return _INTEGER_TOKEN_NORMALIZERS[language]
    except KeyError as exc:
        supported = ", ".join(sorted(_INTEGER_TOKEN_NORMALIZERS))
        raise ValueError(
            f"Unsupported language '{language}'. Supported languages: {supported}"
        ) from exc


def get_text_normalizer(language: str) -> TextNormalizer:
    try:
        return _TEXT_NORMALIZERS[language]
    except KeyError as exc:
        supported = ", ".join(sorted(_TEXT_NORMALIZERS))
        raise ValueError(
            f"Unsupported language '{language}'. Supported languages: {supported}"
        ) from exc


__all__ = [
    "get_integer_token_normalizer",
    "get_text_normalizer",
    "IntegerTokenNormalizer",
    "TextNormalizer",
]

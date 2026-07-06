"""Language-aware text normalization entry points."""

from .languages import get_integer_token_normalizer, get_text_normalizer


def normalize_integer_token(token: str, language: str = "tr") -> str:
    """Convert a standalone integer token into spoken-form text."""
    normalizer = get_integer_token_normalizer(language)
    return normalizer(token)


def normalize_text(text: str, language: str = "tr") -> str:
    """Normalize supported text patterns into spoken-form text."""
    normalizer = get_text_normalizer(language)
    return normalizer(text)

"""Text normalisation shared by the keyword index and the hashing embedder."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Final

import snowballstemmer

_TOKEN: Final = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Small English stop-word list (function words only): keeps negations and numbers.
STOP_WORDS: Final = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does", "for",
        "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my",
        "of", "on", "or", "our", "so", "such", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    )
)  # fmt: skip

_stemmer = snowballstemmer.stemmer("english")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return str(_stemmer.stemWord(word))


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stop words, Snowball-stem."""
    words = (token.replace("'", "") for token in _TOKEN.findall(text.lower()))
    return [_stem(word) for word in words if word not in STOP_WORDS]

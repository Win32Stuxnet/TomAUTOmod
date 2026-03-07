from __future__ import annotations

import re
import unicodedata

_EMOJI_PATTERN = re.compile(
    r"[\U0001f600-\U0001f64f"
    r"\U0001f300-\U0001f5ff"
    r"\U0001f680-\U0001f6ff"
    r"\U0001f1e0-\U0001f1ff"
    r"\U00002702-\U000027b0"
    r"\U0001f900-\U0001f9ff"
    r"\U0001fa00-\U0001fa6f"
    r"\U0001fa70-\U0001faff"
    r"\U00002600-\U000026ff"
    r"]+",
    flags=re.UNICODE,
)

_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_MENTION_PATTERN = re.compile(r"<@!?\d+>")
_REPEATED_CHARS = re.compile(r"(.)\1{2,}")


def extract_features(content: str) -> dict:
    length = len(content)
    if length == 0:
        return _empty_features()

    words = content.split()
    word_count = len(words)

    alpha_chars = [c for c in content if c.isalpha()]
    caps_count = sum(1 for c in alpha_chars if c.isupper())
    caps_ratio = caps_count / len(alpha_chars) if alpha_chars else 0.0

    emoji_chars = _EMOJI_PATTERN.findall(content)
    emoji_count = sum(len(e) for e in emoji_chars)

    special = sum(1 for c in content if not c.isalnum() and not c.isspace())
    special_char_ratio = special / length

    repeated_spans = _REPEATED_CHARS.findall(content)
    repeated_char_count = sum(len(m) for m in _REPEATED_CHARS.finditer(content))
    repeated_chars_ratio = repeated_char_count / length

    unique_words = set(w.lower() for w in words)
    unique_word_ratio = len(unique_words) / word_count if word_count else 1.0

    mention_count = len(_MENTION_PATTERN.findall(content))
    link_count = len(_URL_PATTERN.findall(content))
    newline_count = content.count("\n")

    return {
        "length": length,
        "word_count": word_count,
        "caps_ratio": round(caps_ratio, 4),
        "emoji_count": emoji_count,
        "emoji_density": round(emoji_count / length, 4),
        "special_char_ratio": round(special_char_ratio, 4),
        "repeated_chars_ratio": round(repeated_chars_ratio, 4),
        "unique_word_ratio": round(unique_word_ratio, 4),
        "mention_count": mention_count,
        "link_count": link_count,
        "newline_count": newline_count,
    }


def _empty_features() -> dict:
    return {
        "length": 0,
        "word_count": 0,
        "caps_ratio": 0.0,
        "emoji_count": 0,
        "emoji_density": 0.0,
        "special_char_ratio": 0.0,
        "repeated_chars_ratio": 0.0,
        "unique_word_ratio": 1.0,
        "mention_count": 0,
        "link_count": 0,
        "newline_count": 0,
    }

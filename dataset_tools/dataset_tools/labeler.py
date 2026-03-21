from __future__ import annotations

import logging
import re
from pathlib import Path

from .models import Sample

log = logging.getLogger(__name__)


def _load_wordlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip().lower() for line in lines if line.strip() and not line.startswith("#")]


def _normalize(text: str, leet_map: dict[str, str]) -> str:
    result = text.lower()
    for char, replacement in leet_map.items():
        result = result.replace(char, replacement)
    # collapse repeated characters (e.g. "fuuuck" -> "fuck")
    result = re.sub(r"(.)\1{2,}", r"\1\1", result)
    # remove spaces/dots/dashes between single chars (e.g. "f u c k" -> "fuck")
    result = re.sub(r"(?<=\b\w)[.\s\-_]+(?=\w\b)", "", result)
    return result


def _matches_wordlist(text: str, words: list[str]) -> bool:
    for word in words:
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def label_samples(
    samples: list[Sample],
    wordlists_dir: Path,
    leet_map: dict[str, str] | None = None,
) -> list[Sample]:
    leet = leet_map or {}

    slurs = _load_wordlist(wordlists_dir / "slurs.txt")
    aggressive = _load_wordlist(wordlists_dir / "aggressive.txt")

    log.info("Loaded %d slurs, %d aggressive phrases", len(slurs), len(aggressive))

    stats = {"toxic": 0, "flagged": 0, "skipped": 0}

    for sample in samples:
        if sample.label is not None:
            stats["skipped"] += 1
            continue

        normalized = _normalize(sample.text, leet)

        if _matches_wordlist(normalized, slurs):
            sample.label = "toxic"
            stats["toxic"] += 1
        elif _matches_wordlist(normalized, aggressive):
            sample.label = "flagged"
            stats["flagged"] += 1

    log.info(
        "Labeling complete: %d toxic, %d flagged, %d skipped (already labeled)",
        stats["toxic"], stats["flagged"], stats["skipped"],
    )
    return samples

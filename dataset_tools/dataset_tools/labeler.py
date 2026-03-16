from __future__ import annotations

import logging
import re
from pathlib import Path

from .models import Sample

log = logging.getLogger(__name__)


def _load_wordlist(path: Path) -> list[str]:
    """
    Load a wordlist file and return its non-empty, lowercased entries while skipping comment lines.
    
    Parameters:
        path (Path): Filesystem path to the wordlist file.
    
    Returns:
        list[str]: List of lines from the file, stripped and lowercased; lines that are empty or start with `#` are omitted. If the path does not exist, returns an empty list.
    """
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip().lower() for line in lines if line.strip() and not line.startswith("#")]


def _normalize(text: str, leet_map: dict[str, str]) -> str:
    """
    Normalize and sanitize text for wordlist matching.
    
    Transforms `text` to lowercase, applies character substitutions from `leet_map`,
    collapses runs of the same character of length three or more into two characters
    (e.g., "fuuuck" -> "fuuck"), and removes spaces, dots, underscores or dashes
    placed between single characters to join separated letters (e.g., "f u c k" -> "fuck").
    
    Parameters:
        text (str): Input string to normalize.
        leet_map (dict[str, str]): Mapping of characters or substrings to replace
            (commonly used for leetspeak substitutions); treated as empty if None.
    
    Returns:
        str: The normalized string suitable for wordlist matching.
    """
    result = text.lower()
    for char, replacement in leet_map.items():
        result = result.replace(char, replacement)
    # collapse repeated characters (e.g. "fuuuck" -> "fuck")
    result = re.sub(r"(.)\1{2,}", r"\1\1", result)
    # remove spaces/dots/dashes between single chars (e.g. "f u c k" -> "fuck")
    result = re.sub(r"(?<=\b\w)[.\s\-_]+(?=\w\b)", "", result)
    return result


def _matches_wordlist(text: str, words: list[str]) -> bool:
    """
    Check whether any word from the provided list appears as a whole word in the given text.
    
    Parameters:
        text (str): Input text to search.
        words (list[str]): Words to test for whole-word matches.
    
    Returns:
        bool: True if any word matches as a whole word in the text, False otherwise.
    """
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
    """
    Assigns toxicity-related labels to unlabeled Sample objects by matching their normalized text against wordlists.
    
    Parameters:
        samples (list[Sample]): List of Sample objects to label; samples with a pre-existing `label` are left unchanged.
        wordlists_dir (Path): Directory containing wordlist files `slurs.txt` and `aggressive.txt`; missing files yield no matches.
        leet_map (dict[str, str] | None): Optional mapping of characters for leetspeak substitution before matching; treated as empty when `None`.
    
    Returns:
        list[Sample]: The same list of Sample objects, with `label` set to `"toxic"` for slur matches, `"flagged"` for aggressive-phrase matches, or left unchanged if already labeled or no match is found.
    """
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

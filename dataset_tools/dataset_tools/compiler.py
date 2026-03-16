from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

from .models import Sample
from .storage import load_samples

log = logging.getLogger(__name__)


def compile_dataset(
    input_dir: Path,
    output_dir: Path,
    output_format: str = "both",
) -> None:
    samples = load_samples(input_dir)
    if not samples:
        log.error("No samples found in %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Print stats
    labels = Counter(s.label or "unlabeled" for s in samples)
    sources = Counter(s.source for s in samples)
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {len(samples)}")
    print(f"\nBy label:")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")
    print(f"\nBy source:")
    for source, count in sources.most_common():
        print(f"  {source}: {count}")

    if output_format in ("csv", "both"):
        _export_csv(samples, output_dir / "dataset.csv")

    if output_format in ("huggingface", "both"):
        _export_huggingface(samples, output_dir / "hf_dataset")

    print(f"\nCompiled to {output_dir}/")


def _export_csv(samples: list[Sample], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        writer.writeheader()
        for s in samples:
            writer.writerow({"text": s.text, "label": s.label or "safe", "source": s.source})
    log.info("Exported CSV to %s", path)


def _export_huggingface(samples: list[Sample], path: Path) -> None:
    from datasets import Dataset

    records = [
        {"text": s.text, "label": s.label or "safe", "source": s.source}
        for s in samples
    ]
    ds = Dataset.from_list(records)
    ds.save_to_disk(str(path))
    log.info("Exported HuggingFace dataset to %s", path)

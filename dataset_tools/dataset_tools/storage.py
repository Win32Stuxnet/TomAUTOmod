from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from .models import Sample

log = logging.getLogger(__name__)


def save_samples(samples: list[Sample], output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    path = output_dir / filename

    data = [s.to_dict() for s in samples]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d samples to %s", len(samples), path)
    return path


def load_samples(input_path: Path) -> list[Sample]:
    samples: list[Sample] = []

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
    else:
        return samples

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        samples.extend(Sample.from_dict(d) for d in data)

    log.info("Loaded %d samples from %s", len(samples), input_path)
    return samples

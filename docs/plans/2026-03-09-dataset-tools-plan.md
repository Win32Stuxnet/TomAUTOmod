# Dataset Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone CLI toolkit that scrapes Reddit, exports from the bot's MongoDB, labels samples using word lists, and compiles everything into HuggingFace-compatible datasets.

**Architecture:** Subcommand-based CLI (`python -m dataset_tools <cmd>`). All sources normalize to a shared `Sample` dataclass, stored as JSON in `data/`. A labeler applies word-list rules, then a compiler merges and exports to CSV + HuggingFace parquet.

**Tech Stack:** Python 3.11+, PRAW (Reddit), pymongo (MongoDB), datasets (HuggingFace), PyYAML, python-dotenv

---

### Task 1: Scaffold project structure and dependencies

**Files:**
- Create: `dataset_tools/dataset_tools/__init__.py`
- Create: `dataset_tools/dataset_tools/scrapers/__init__.py`
- Create: `dataset_tools/dataset_tools/exporters/__init__.py`
- Create: `dataset_tools/requirements.txt`
- Create: `dataset_tools/.gitignore`
- Create: `dataset_tools/wordlists/.gitkeep`

**Step 1: Create directory structure**

```bash
mkdir -p dataset_tools/dataset_tools/scrapers
mkdir -p dataset_tools/dataset_tools/exporters
mkdir -p dataset_tools/data/raw
mkdir -p dataset_tools/data/labeled
mkdir -p dataset_tools/data/compiled
mkdir -p dataset_tools/wordlists
```

**Step 2: Create init files**

`dataset_tools/dataset_tools/__init__.py` — empty file

`dataset_tools/dataset_tools/scrapers/__init__.py` — empty file

`dataset_tools/dataset_tools/exporters/__init__.py` — empty file

**Step 3: Create requirements.txt**

```
praw>=7.7.0
pymongo>=4.12.0
datasets>=3.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

**Step 4: Create .gitignore**

```
data/
.env
__pycache__/
*.pyc
```

**Step 5: Create wordlists/.gitkeep**

Empty file so git tracks the directory.

**Step 6: Commit**

```bash
git add dataset_tools/
git commit -m "scaffold: dataset_tools project structure and dependencies"
```

---

### Task 2: Data model and storage layer

**Files:**
- Create: `dataset_tools/dataset_tools/models.py`
- Create: `dataset_tools/dataset_tools/storage.py`

**Step 1: Write models.py**

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Sample:
    text: str
    label: str | None  # "toxic", "flagged", "safe", or None
    source: str  # "reddit", "mongodb", "manual"
    metadata: dict = field(default_factory=dict)
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Sample:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
```

**Step 2: Write storage.py**

```python
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
```

**Step 3: Commit**

```bash
git add dataset_tools/dataset_tools/models.py dataset_tools/dataset_tools/storage.py
git commit -m "feat: add Sample data model and JSON storage layer"
```

---

### Task 3: Config loader

**Files:**
- Create: `dataset_tools/dataset_tools/config.py`
- Create: `dataset_tools/config.yaml`

**Step 1: Write config.yaml**

```yaml
reddit:
  sort: controversial  # hot, new, controversial, top
  limit: 1000          # comments per subreddit
  subreddits: []       # user fills these in

mongodb:
  db_name: discord     # matches bot's MONGODB_DB_NAME

labeler:
  # leet speak substitutions for evasion detection
  leet_map:
    "0": "o"
    "1": "i"
    "3": "e"
    "4": "a"
    "5": "s"
    "@": "a"
    "$": "s"

data:
  raw_dir: data/raw
  labeled_dir: data/labeled
  compiled_dir: data/compiled
```

**Step 2: Write config.py**

```python
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    load_dotenv(_PROJECT_ROOT / ".env")

    config_path = _PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        cfg = {}

    # Inject env vars
    cfg.setdefault("reddit", {})
    cfg["reddit"]["client_id"] = os.getenv("REDDIT_CLIENT_ID", "")
    cfg["reddit"]["client_secret"] = os.getenv("REDDIT_CLIENT_SECRET", "")
    cfg["reddit"]["user_agent"] = os.getenv(
        "REDDIT_USER_AGENT", "dataset_tools/1.0"
    )

    cfg.setdefault("mongodb", {})
    cfg["mongodb"]["uri"] = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    return cfg
```

**Step 3: Commit**

```bash
git add dataset_tools/dataset_tools/config.py dataset_tools/config.yaml
git commit -m "feat: add YAML config loader with env var support"
```

---

### Task 4: Reddit scraper

**Files:**
- Create: `dataset_tools/dataset_tools/scrapers/reddit.py`

**Step 1: Write reddit.py**

```python
from __future__ import annotations

import logging

import praw

from ..models import Sample

log = logging.getLogger(__name__)


def scrape(config: dict, subreddits: list[str] | None = None, limit: int | None = None, sort: str | None = None) -> list[Sample]:
    reddit_cfg = config.get("reddit", {})

    reddit = praw.Reddit(
        client_id=reddit_cfg["client_id"],
        client_secret=reddit_cfg["client_secret"],
        user_agent=reddit_cfg["user_agent"],
    )

    subs = subreddits or reddit_cfg.get("subreddits", [])
    max_comments = limit or reddit_cfg.get("limit", 1000)
    sort_mode = sort or reddit_cfg.get("sort", "controversial")

    if not subs:
        log.error("No subreddits specified. Use --subreddits or set in config.yaml")
        return []

    samples: list[Sample] = []

    for sub_name in subs:
        log.info("Scraping r/%s (%s, limit=%d)", sub_name, sort_mode, max_comments)
        subreddit = reddit.subreddit(sub_name)

        getter = getattr(subreddit, sort_mode, subreddit.controversial)
        submissions = getter(limit=None)

        count = 0
        for submission in submissions:
            if count >= max_comments:
                break

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if count >= max_comments:
                    break

                body = comment.body.strip()
                if not body or body in ("[deleted]", "[removed]"):
                    continue

                is_removed = body == "[removed]" or getattr(comment, "removed", False)

                samples.append(Sample(
                    text=body,
                    label="toxic" if is_removed else None,
                    source="reddit",
                    metadata={
                        "subreddit": sub_name,
                        "score": comment.score,
                        "author": str(comment.author) if comment.author else None,
                        "created_utc": comment.created_utc,
                        "submission_title": submission.title,
                        "is_removed": is_removed,
                    },
                ))
                count += 1

        log.info("Scraped %d comments from r/%s", count, sub_name)

    log.info("Total scraped: %d samples", len(samples))
    return samples
```

**Step 2: Commit**

```bash
git add dataset_tools/dataset_tools/scrapers/reddit.py
git commit -m "feat: add Reddit comment scraper using PRAW"
```

---

### Task 5: MongoDB exporter

**Files:**
- Create: `dataset_tools/dataset_tools/exporters/mongodb.py`

**Step 1: Write mongodb.py**

Reference: Bot stores samples in `ml_training_data` collection with fields: `guild_id`, `message_id`, `user_id`, `features` (dict), `label` (str|None), `created_at`.

```python
from __future__ import annotations

import logging

from pymongo import MongoClient

from ..models import Sample

log = logging.getLogger(__name__)


def export(config: dict, labeled_only: bool = False) -> list[Sample]:
    mongo_cfg = config.get("mongodb", {})
    uri = mongo_cfg.get("uri", "mongodb://localhost:27017")
    db_name = mongo_cfg.get("db_name", "discord")

    client = MongoClient(uri)
    db = client[db_name]
    collection = db["ml_training_data"]

    query = {}
    if labeled_only:
        query["label"] = {"$ne": None}

    samples: list[Sample] = []
    for doc in collection.find(query):
        text = doc.get("text", "")
        # Bot samples may not store raw text — store features as metadata
        features = doc.get("features", {})

        samples.append(Sample(
            text=text,
            label=doc.get("label"),
            source="mongodb",
            metadata={
                "guild_id": doc.get("guild_id"),
                "user_id": doc.get("user_id"),
                "message_id": doc.get("message_id"),
                "features": features,
            },
        ))

    client.close()
    log.info("Exported %d samples from MongoDB (%s)", len(samples), "labeled only" if labeled_only else "all")
    return samples
```

**Step 2: Commit**

```bash
git add dataset_tools/dataset_tools/exporters/mongodb.py
git commit -m "feat: add MongoDB exporter for bot training data"
```

---

### Task 6: Labeler with word-list matching

**Files:**
- Create: `dataset_tools/dataset_tools/labeler.py`
- Create: `dataset_tools/wordlists/slurs.txt` (placeholder)
- Create: `dataset_tools/wordlists/aggressive.txt` (placeholder)

**Step 1: Write labeler.py**

```python
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
```

**Step 2: Create placeholder word lists**

`dataset_tools/wordlists/slurs.txt`:
```
# Add slurs here, one per line
# Lines starting with # are comments
```

`dataset_tools/wordlists/aggressive.txt`:
```
# Add aggressive phrases here, one per line
# These get labeled as "flagged" (less severe than slurs)
# Lines starting with # are comments
```

**USER INPUT NEEDED:** The user fills in `slurs.txt` and `aggressive.txt` with their own curated words. These are the core of the labeling system.

**Step 3: Commit**

```bash
git add dataset_tools/dataset_tools/labeler.py dataset_tools/wordlists/
git commit -m "feat: add rule-based labeler with word-list matching and evasion detection"
```

---

### Task 7: Compiler (JSON → CSV + HuggingFace)

**Files:**
- Create: `dataset_tools/dataset_tools/compiler.py`

**Step 1: Write compiler.py**

```python
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
```

**Step 2: Commit**

```bash
git add dataset_tools/dataset_tools/compiler.py
git commit -m "feat: add dataset compiler with CSV and HuggingFace export"
```

---

### Task 8: CLI entry point

**Files:**
- Create: `dataset_tools/dataset_tools/__main__.py`

**Step 1: Write __main__.py**

```python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .storage import save_samples, load_samples

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cmd_scrape_reddit(args: argparse.Namespace, config: dict) -> None:
    from .scrapers.reddit import scrape

    samples = scrape(
        config,
        subreddits=args.subreddits,
        limit=args.limit,
        sort=args.sort,
    )
    if samples:
        raw_dir = _PROJECT_ROOT / config.get("data", {}).get("raw_dir", "data/raw")
        save_samples(samples, raw_dir, prefix="reddit")


def cmd_export_mongodb(args: argparse.Namespace, config: dict) -> None:
    from .exporters.mongodb import export

    samples = export(config, labeled_only=args.labeled_only)
    if samples:
        raw_dir = _PROJECT_ROOT / config.get("data", {}).get("raw_dir", "data/raw")
        save_samples(samples, raw_dir, prefix="mongodb")


def cmd_label(args: argparse.Namespace, config: dict) -> None:
    from .labeler import label_samples

    input_dir = Path(args.input) if args.input else _PROJECT_ROOT / config.get("data", {}).get("raw_dir", "data/raw")
    output_dir = Path(args.output) if args.output else _PROJECT_ROOT / config.get("data", {}).get("labeled_dir", "data/labeled")
    wordlists_dir = Path(args.wordlists) if args.wordlists else _PROJECT_ROOT / "wordlists"

    samples = load_samples(input_dir)
    if not samples:
        print("No samples found.")
        return

    leet_map = config.get("labeler", {}).get("leet_map", {})
    labeled = label_samples(samples, wordlists_dir, leet_map=leet_map)
    save_samples(labeled, output_dir, prefix="labeled")


def cmd_compile(args: argparse.Namespace, config: dict) -> None:
    from .compiler import compile_dataset

    input_dir = Path(args.input) if args.input else _PROJECT_ROOT / config.get("data", {}).get("labeled_dir", "data/labeled")
    output_dir = Path(args.output) if args.output else _PROJECT_ROOT / config.get("data", {}).get("compiled_dir", "data/compiled")
    compile_dataset(input_dir, output_dir, output_format=args.format)


def cmd_stats(args: argparse.Namespace, config: dict) -> None:
    from collections import Counter

    for stage in ("raw", "labeled", "compiled"):
        stage_dir = _PROJECT_ROOT / config.get("data", {}).get(f"{stage}_dir", f"data/{stage}")
        samples = load_samples(stage_dir)
        if not samples:
            continue
        labels = Counter(s.label or "unlabeled" for s in samples)
        print(f"\n=== {stage.upper()} ({len(samples)} samples) ===")
        for label, count in labels.most_common():
            print(f"  {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataset_tools",
        description="Build toxicity classification datasets",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    # scrape reddit
    scrape_p = sub.add_parser("scrape", help="Scrape data from sources")
    scrape_sub = scrape_p.add_subparsers(dest="source")
    reddit_p = scrape_sub.add_parser("reddit", help="Scrape Reddit comments")
    reddit_p.add_argument("--subreddits", nargs="+", help="Subreddits to scrape")
    reddit_p.add_argument("--limit", type=int, help="Max comments per subreddit")
    reddit_p.add_argument("--sort", choices=["hot", "new", "controversial", "top"], help="Sort mode")

    # export mongodb
    export_p = sub.add_parser("export", help="Export data from sources")
    export_sub = export_p.add_subparsers(dest="source")
    mongo_p = export_sub.add_parser("mongodb", help="Export from bot MongoDB")
    mongo_p.add_argument("--labeled-only", action="store_true", help="Only export labeled samples")

    # label
    label_p = sub.add_parser("label", help="Apply rule-based labels")
    label_p.add_argument("--input", help="Input directory (default: data/raw)")
    label_p.add_argument("--output", help="Output directory (default: data/labeled)")
    label_p.add_argument("--wordlists", help="Wordlists directory (default: wordlists/)")

    # compile
    compile_p = sub.add_parser("compile", help="Compile labeled data into dataset")
    compile_p.add_argument("--input", help="Input directory (default: data/labeled)")
    compile_p.add_argument("--output", help="Output directory (default: data/compiled)")
    compile_p.add_argument("--format", choices=["csv", "huggingface", "both"], default="both", help="Output format")

    # stats
    sub.add_parser("stats", help="Show dataset statistics")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config()

    commands = {
        "scrape": lambda: cmd_scrape_reddit(args, config) if getattr(args, "source", None) == "reddit" else print("Unknown source. Use: scrape reddit"),
        "export": lambda: cmd_export_mongodb(args, config) if getattr(args, "source", None) == "mongodb" else print("Unknown source. Use: export mongodb"),
        "label": lambda: cmd_label(args, config),
        "compile": lambda: cmd_compile(args, config),
        "stats": lambda: cmd_stats(args, config),
    }

    handler = commands.get(args.command)
    if handler:
        handler()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add dataset_tools/dataset_tools/__main__.py
git commit -m "feat: add CLI entry point with all subcommands"
```

---

### Task 9: Smoke test the full pipeline

**Step 1: Install dependencies**

```bash
cd dataset_tools
pip install -r requirements.txt
```

**Step 2: Verify CLI loads**

```bash
python -m dataset_tools --help
python -m dataset_tools stats
```

Expected: help output shows all subcommands, stats shows nothing yet (no data).

**Step 3: Test label + compile with mock data**

Create a test file `data/raw/test.json` manually with a few samples, run label and compile to verify the full pipeline works end to end.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address smoke test issues"
```

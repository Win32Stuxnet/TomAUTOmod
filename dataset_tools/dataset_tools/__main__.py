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

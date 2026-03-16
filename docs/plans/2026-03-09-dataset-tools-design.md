# Dataset Tools — Design Document

## Purpose

Standalone CLI toolkit for building toxicity classification datasets. Scrapes Reddit, exports from the bot's MongoDB, applies rule-based labeling, and compiles into HuggingFace-compatible datasets for training a content moderation model.

## Project Structure

```
dataset_tools/
  ├── dataset_tools/
  │   ├── __init__.py
  │   ├── __main__.py           # argparse CLI with subcommands
  │   ├── config.py             # loads config.yaml + .env
  │   ├── models.py             # Sample dataclass
  │   ├── storage.py            # read/write JSON to data/ folder
  │   ├── scrapers/
  │   │   ├── __init__.py
  │   │   └── reddit.py         # PRAW scraper
  │   ├── exporters/
  │   │   ├── __init__.py
  │   │   └── mongodb.py        # bot's ml_training_data exporter
  │   ├── labeler.py            # rule-based auto-labeling
  │   └── compiler.py           # JSON → HuggingFace Dataset + CSV
  ├── data/
  │   ├── raw/                  # scraped/exported JSON before labeling
  │   ├── labeled/              # after labeling rules applied
  │   └── compiled/             # final HuggingFace dataset + CSV
  ├── wordlists/                # curated slur/toxicity lists (text files)
  ├── config.yaml               # subreddits, label rules, thresholds
  ├── .env                      # Reddit API creds, MongoDB URI
  └── requirements.txt
```

## Data Model

```python
@dataclass
class Sample:
    text: str                    # message/comment content
    label: str | None            # "toxic", "flagged", "safe", or None
    source: str                  # "reddit", "mongodb", "manual"
    metadata: dict               # subreddit, author, score, timestamp, etc.
```

Stored as JSON files in `data/raw/`, one file per scrape/export run (e.g. `reddit_2026-03-09_askreddit.json`).

## CLI Commands

```bash
# Scrape Reddit comments
python -m dataset_tools scrape reddit \
    --subreddits askreddit science gaming \
    --limit 1000 \
    --sort controversial

# Export from bot's MongoDB
python -m dataset_tools export mongodb \
    --uri $MONGODB_URI \
    --labeled-only               # or --all

# Apply rule-based labels
python -m dataset_tools label \
    --input data/raw/ \
    --wordlists wordlists/ \
    --output data/labeled/

# Compile to final dataset
python -m dataset_tools compile \
    --input data/labeled/ \
    --format both                # json + huggingface
    --output data/compiled/

# Show dataset statistics
python -m dataset_tools stats
```

## Component Details

### Reddit Scraper (`scrapers/reddit.py`)
- Uses PRAW (Python Reddit API Wrapper)
- Configurable via config.yaml: subreddit list, sort mode (hot/new/controversial), limit per sub
- Saves per comment: text, score, subreddit, author, created_utc, is_removed
- Removed/deleted comments auto-labeled "toxic" (moderator-removed = natural label)
- Rate limited by PRAW (60 req/min)

### MongoDB Exporter (`exporters/mongodb.py`)
- Connects to bot's `ml_training_data` collection
- Pulls samples with existing labels from moderator actions
- Converts to shared Sample format
- Preserves original 13 features as metadata

### Labeler (`labeler.py`)
- Loads word lists from `wordlists/` (one file per category, one word/phrase per line)
- Regex matching with common evasion handling (spaces between letters, leet speak)
- Slur match → "toxic", aggressive pattern → "flagged", no match → left None
- Never overwrites existing labels

### Compiler (`compiler.py`)
- Merges all labeled JSON into unified dataset
- Exports as CSV (quick inspection) and HuggingFace Dataset (.parquet)
- Prints summary: total samples, label distribution, source breakdown

## Dependencies (requirements.txt)
- praw (Reddit API)
- pymongo (MongoDB export)
- datasets (HuggingFace format)
- pyyaml (config)
- python-dotenv (env vars)

## Future Plans
- Discord button-based labeling UI in the bot
- Web interface for browsing and labeling samples

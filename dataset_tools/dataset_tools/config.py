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

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FEATURE_KEYS = [
    "length",
    "word_count",
    "caps_ratio",
    "emoji_count",
    "emoji_density",
    "special_char_ratio",
    "repeated_chars_ratio",
    "unique_word_ratio",
    "mention_count",
    "link_count",
    "newline_count",
    "messages_last_60s",
    "avg_interval_seconds",
]


async def export_data(uri: str, db_name: str) -> tuple[np.ndarray, np.ndarray]:
    from pymongo import AsyncMongoClient

    client = AsyncMongoClient(uri)
    db = client[db_name]
    collection = db["ml_training_data"]

    X_rows: list[list[float]] = []
    y_labels: list[str] = []

    async for doc in collection.find({"label": {"$ne": None}}):
        features = doc.get("features", {})
        row = [float(features.get(k, 0)) for k in FEATURE_KEYS]
        X_rows.append(row)
        y_labels.append(doc["label"])

    labeled_count = len(X_rows)
    log.info("Found %d labeled samples", labeled_count)

    if labeled_count < 50:
        log.error("Not enough labeled data to train. Need at least 50, have %d.", labeled_count)
        sys.exit(1)

    safe_limit = labeled_count * 2
    pipeline = [
        {"$match": {"label": None}},
        {"$sample": {"size": safe_limit}},
    ]
    async for doc in collection.aggregate(pipeline):
        features = doc.get("features", {})
        row = [float(features.get(k, 0)) for k in FEATURE_KEYS]
        X_rows.append(row)
        y_labels.append("safe")

    log.info("Total samples: %d (%d labeled + %d safe)", len(X_rows), labeled_count, len(X_rows) - labeled_count)

    client.close()
    return np.array(X_rows), np.array(y_labels)


def train(X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\n=== Classification Report ===")
    print(report)

    importances = sorted(
        zip(FEATURE_KEYS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n=== Feature Importance ===")
    for name, score in importances:
        bar = "#" * int(score * 50)
        print(f"  {name:.<25s} {score:.4f} {bar}")

    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_KEYS}, output_path)
    log.info("Model saved to %s", output_path)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Train moderation classifier")
    parser.add_argument("--db", default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument("--db-name", default="discord_mod_bot", help="Database name")
    parser.add_argument("--out", default="model.joblib", help="Output model path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    X, y = await export_data(args.db, args.db_name)
    train(X, y, Path(args.out))


if __name__ == "__main__":
    asyncio.run(main())

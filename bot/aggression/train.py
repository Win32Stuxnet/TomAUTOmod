from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

log = logging.getLogger(__name__)


async def export_data(uri: str, db_name: str) -> tuple[np.ndarray, np.ndarray]:
    from pymongo import AsyncMongoClient

    client = AsyncMongoClient(uri)
    db = client[db_name]
    collection = db["aggression_training_data"]

    X_rows: list[list[float]] = []
    y_labels: list[str] = []

    try:
        async for doc in collection.find({"label": {"$ne": None}}):
            embedding = doc.get("embedding") or []
            if not embedding:
                continue
            X_rows.append([float(value) for value in embedding])
            y_labels.append(doc["label"])
    finally:
        client.close()

    labeled_count = len(X_rows)
    if labeled_count < 50:
        raise ValueError(f"Need at least 50 labeled samples to train, found {labeled_count}.")

    class_counts = Counter(y_labels)
    if len(class_counts) < 2:
        raise ValueError("Need at least two aggression labels to train a classifier.")
    if min(class_counts.values()) < 2:
        raise ValueError("Need at least two samples for each aggression label to train.")

    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_labels)


def train(X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Aggression Classification Report ===")
    print(classification_report(y_test, y_pred))

    joblib.dump({"model": model, "features": "embedding"}, output_path)
    log.info("Aggression model saved to %s", output_path)


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train aggression classifier")
    parser.add_argument("--db", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"), help="MongoDB URI")
    parser.add_argument("--db-name", default=os.getenv("MONGODB_DB_NAME", "discord_mod_bot"), help="Database name")
    parser.add_argument("--out", default="aggression_model.joblib", help="Output model path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        X, y = await export_data(args.db, args.db_name)
        train(X, y, Path(args.out))
    except ValueError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())

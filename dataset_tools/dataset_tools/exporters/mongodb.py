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

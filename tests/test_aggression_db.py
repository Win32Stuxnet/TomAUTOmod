from __future__ import annotations

from unittest.mock import patch

from bot.database import Database


def test_aggression_strikes_property():
    with patch("bot.database.AsyncMongoClient"):
        db = Database("mongodb://localhost", "test")
    assert db.aggression_strikes is not None


def test_aggression_training_data_property():
    with patch("bot.database.AsyncMongoClient"):
        db = Database("mongodb://localhost", "test")
    assert db.aggression_training_data is not None

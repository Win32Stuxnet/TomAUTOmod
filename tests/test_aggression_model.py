from __future__ import annotations

from datetime import datetime

from bot.models.aggression_data import AggressionTrainingSample


def test_to_doc_contains_all_fields():
    sample = AggressionTrainingSample(
        guild_id=123,
        message_id=456,
        user_id=789,
        channel_id=101,
        content="you are so annoying",
        embedding=[0.1] * 384,
    )
    doc = sample.to_doc()

    assert doc["guild_id"] == 123
    assert doc["message_id"] == 456
    assert doc["user_id"] == 789
    assert doc["channel_id"] == 101
    assert doc["content"] == "you are so annoying"
    assert len(doc["embedding"]) == 384
    assert doc["label"] is None
    assert doc["score"] == 0.0
    assert doc["reviewed_by"] is None
    assert doc["reviewed_at"] is None
    assert isinstance(doc["created_at"], datetime)


def test_from_doc_round_trips():
    sample = AggressionTrainingSample(
        guild_id=123,
        message_id=456,
        user_id=789,
        channel_id=101,
        content="test",
        embedding=[0.5] * 384,
        label="aggressive",
        score=0.85,
    )
    doc = sample.to_doc()
    doc["_id"] = "mongo_id"
    restored = AggressionTrainingSample.from_doc(doc)

    assert restored.guild_id == 123
    assert restored.label == "aggressive"
    assert restored.score == 0.85
    assert len(restored.embedding) == 384

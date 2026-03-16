from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_record_strike_inserts_documents():
    from bot.aggression.tracker import StrikeTracker

    db = MagicMock()
    db.aggression_strikes.update_one = AsyncMock()
    db.aggression_training_data.update_one = AsyncMock()

    tracker = StrikeTracker(db)
    await tracker.record_strike(
        guild_id=1,
        user_id=2,
        message_id=3,
        channel_id=4,
        score=0.91,
        content="you are being hostile",
        embedding=[0.1] * 384,
    )

    db.aggression_strikes.update_one.assert_called_once()
    db.aggression_training_data.update_one.assert_called_once()


@pytest.mark.asyncio
async def test_check_threshold_counts_recent_strikes():
    from bot.aggression.tracker import StrikeTracker

    db = MagicMock()
    db.aggression_strikes.count_documents = AsyncMock(return_value=3)

    tracker = StrikeTracker(db)
    crossed = await tracker.check_threshold(1, 2, 3, 2)

    assert crossed is True


@pytest.mark.asyncio
async def test_has_alert_in_window_returns_true_when_prior_alert_exists():
    from bot.aggression.tracker import StrikeTracker

    db = MagicMock()
    db.aggression_strikes.find_one = AsyncMock(return_value={"_id": "x"})

    tracker = StrikeTracker(db)
    assert await tracker.has_alert_in_window(1, 2, 2) is True


@pytest.mark.asyncio
async def test_mark_alerted_updates_recent_strikes():
    from bot.aggression.tracker import StrikeTracker

    db = MagicMock()
    db.aggression_strikes.update_many = AsyncMock(return_value=SimpleNamespace(modified_count=2))

    tracker = StrikeTracker(db)
    await tracker.mark_alerted(1, 2, 2)

    db.aggression_strikes.update_many.assert_called_once()

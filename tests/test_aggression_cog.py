from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import numpy as np
import pytest

from bot.ml.predictor import Prediction


def _make_bot():
    bot = MagicMock()
    bot.db = MagicMock()
    bot.db.aggression_strikes.count_documents = AsyncMock(return_value=0)
    bot.db.aggression_training_data.count_documents = AsyncMock(return_value=0)
    bot.owner_ids = {111}
    bot.collector = MagicMock()
    bot.collector.process_message = AsyncMock(return_value=None)
    return bot


def _make_message(*, guild_id=123, author_id=456, content="you are so annoying"):
    message = MagicMock(spec=discord.Message)
    message.guild = MagicMock()
    message.guild.id = guild_id
    message.author = MagicMock(spec=discord.Member)
    message.author.id = author_id
    message.author.bot = False
    message.author.mention = f"<@{author_id}>"
    message.channel = MagicMock()
    message.channel.id = 999
    message.channel.mention = "<#999>"
    message.id = 789
    message.content = content
    return message


@pytest.mark.asyncio
async def test_on_message_skips_bot_messages():
    from bot.cogs.aggression import Aggression

    bot = _make_bot()
    cog = Aggression(bot)
    cog._ready = True
    cog.predictor = MagicMock()

    message = _make_message()
    message.author.bot = True

    await cog.on_message(message)
    bot.collector.process_message.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_skips_when_automod_flags():
    from bot.cogs.aggression import Aggression

    bot = _make_bot()
    bot.collector.process_message = AsyncMock(
        return_value=Prediction(label="toxic", confidence=0.9, features={})
    )
    cog = Aggression(bot)
    cog._ready = True
    cog.predictor = MagicMock()
    cog.embedder = MagicMock()
    cog.embedder.embed = AsyncMock()

    mock_config = MagicMock()
    mock_config.ml_consent = True
    with patch.object(cog.config_svc, "get", AsyncMock(return_value=mock_config)):
        message = _make_message()
        await cog.on_message(message)

    cog.embedder.embed.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_records_strike_and_alerts():
    from bot.cogs.aggression import Aggression

    bot = _make_bot()
    cog = Aggression(bot)
    cog._ready = True
    cog.embedder = MagicMock()
    cog.embedder.embed = AsyncMock(return_value=np.ones(384, dtype=np.float32))
    cog.predictor = MagicMock()
    cog.predictor.predict = AsyncMock(return_value=Prediction(label="aggressive", confidence=0.82, features={}))
    cog.tracker = MagicMock()
    cog.tracker.record_strike = AsyncMock()
    cog.tracker.check_threshold = AsyncMock(return_value=True)
    cog.tracker.has_alert_in_window = AsyncMock(return_value=False)
    cog.tracker.mark_alerted = AsyncMock()
    cog._send_alert = AsyncMock(return_value=True)

    mock_config = MagicMock()
    mock_config.ml_consent = True
    mock_config.aggression_strike_count = 3
    mock_config.aggression_window_hours = 2

    with patch.object(cog.config_svc, "get", AsyncMock(return_value=mock_config)):
        message = _make_message()
        await cog.on_message(message)

    cog.tracker.record_strike.assert_called_once()
    cog._send_alert.assert_called_once()
    cog.tracker.mark_alerted.assert_called_once()

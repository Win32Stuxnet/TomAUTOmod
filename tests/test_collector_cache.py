from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_message() -> MagicMock:
    message = MagicMock()
    message.guild = MagicMock()
    message.guild.id = 123
    message.author = MagicMock()
    message.author.bot = False
    message.author.id = 456
    message.channel = MagicMock()
    message.channel.id = 789
    message.id = 999
    message.content = "hello world"
    return message


@pytest.mark.asyncio
async def test_process_message_caches_result():
    from bot.ml.collector import Collector

    bot = MagicMock()
    bot.db = MagicMock()
    bot.db.ml_training_data.insert_one = AsyncMock()

    with patch("bot.ml.collector.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        collector = Collector(bot)

    await collector.setup()

    mock_config = MagicMock()
    mock_config.ml_consent = True

    with patch("bot.ml.collector.ConfigService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get = AsyncMock(return_value=mock_config)
        mock_svc_cls.return_value = mock_svc

        message = _make_message()
        result1 = await collector.process_message(message)
        result2 = await collector.process_message(message)

    assert bot.db.ml_training_data.insert_one.call_count == 1
    assert result1 is result2


@pytest.mark.asyncio
async def test_process_message_deduplicates_concurrent_calls():
    from bot.ml.collector import Collector

    bot = MagicMock()
    bot.db = MagicMock()
    bot.db.ml_training_data.insert_one = AsyncMock()

    with patch("bot.ml.collector.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        collector = Collector(bot)

    await collector.setup()

    mock_config = MagicMock()
    mock_config.ml_consent = True

    with patch("bot.ml.collector.ConfigService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get = AsyncMock(return_value=mock_config)
        mock_svc_cls.return_value = mock_svc

        message = _make_message()
        result1, result2 = await asyncio.gather(
            collector.process_message(message),
            collector.process_message(message),
        )

    assert bot.db.ml_training_data.insert_one.call_count == 1
    assert result1 is result2

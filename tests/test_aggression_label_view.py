from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest


@pytest.mark.asyncio
async def test_aggressive_label_updates_document():
    from bot.views.aggression_label import AggressionLabelView

    bot = MagicMock()
    bot.db.aggression_training_data.update_one = AsyncMock()

    view = AggressionLabelView(bot, guild_id=123, message_id=456)
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock()
    interaction.user.id = 789
    interaction.user.mention = "<@789>"
    interaction.response = AsyncMock()

    with patch("bot.views.aggression_label.is_mod", return_value=True):
        await view._label(interaction, "aggressive")

    update = bot.db.aggression_training_data.update_one.call_args[0][1]
    assert update["$set"]["label"] == "aggressive"


@pytest.mark.asyncio
async def test_not_aggressive_label_updates_document():
    from bot.views.aggression_label import AggressionLabelView

    bot = MagicMock()
    bot.db.aggression_training_data.update_one = AsyncMock()

    view = AggressionLabelView(bot, guild_id=123, message_id=456)
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock()
    interaction.user.id = 789
    interaction.user.mention = "<@789>"
    interaction.response = AsyncMock()

    with patch("bot.views.aggression_label.is_mod", return_value=True):
        await view._label(interaction, "not_aggressive")

    update = bot.db.aggression_training_data.update_one.call_args[0][1]
    assert update["$set"]["label"] == "not_aggressive"

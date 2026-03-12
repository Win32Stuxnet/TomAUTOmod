from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from bot.ml.predictor import HeuristicPredictor, TrainedPredictor


def _make_cog():
    from bot.cogs.ml import ML

    bot = MagicMock()
    bot.db = MagicMock()
    bot.db.ml_training_data = MagicMock()
    bot.collector = MagicMock()
    bot.collector.predictor = HeuristicPredictor()

    cog = ML(bot)
    return cog, bot


def _make_interaction(guild_id: int = 12345):
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.guild = MagicMock()
    interaction.guild.id = guild_id
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    return interaction


@pytest.mark.asyncio
async def test_ml_stats_no_data_returns_error():
    cog, bot = _make_cog()
    interaction = _make_interaction()

    bot.db.ml_training_data.count_documents = AsyncMock(return_value=0)

    await cog.ml_stats.callback(cog, interaction)

    interaction.response.defer.assert_called_once_with(ephemeral=True)
    interaction.followup.send.assert_called_once()

    call_kwargs = interaction.followup.send.call_args
    embed = call_kwargs.kwargs.get("embed") or call_kwargs[1].get("embed")
    assert "No ML data collected yet" in embed.description


@pytest.mark.asyncio
async def test_ml_stats_with_data_shows_counts():
    cog, bot = _make_cog()
    interaction = _make_interaction()

    bot.db.ml_training_data.count_documents = AsyncMock(
        side_effect=[100, 70, 15, 15]
    )

    await cog.ml_stats.callback(cog, interaction)

    interaction.followup.send.assert_called_once()
    call_kwargs = interaction.followup.send.call_args
    embed = call_kwargs.kwargs.get("embed") or call_kwargs[1].get("embed")

    field_values = {f.name: f.value for f in embed.fields}
    assert field_values["Total Samples"] == "100"
    assert field_values["Unlabeled (safe)"] == "70"
    assert field_values["Toxic"] == "15"
    assert field_values["Flagged"] == "15"
    assert field_values["Labeled"] == "30"


@pytest.mark.asyncio
async def test_ml_stats_footer_shows_needed_samples():
    cog, bot = _make_cog()
    interaction = _make_interaction()

    # total=50, unlabeled=40, toxic=5, flagged=5 → labeled=10
    bot.db.ml_training_data.count_documents = AsyncMock(
        side_effect=[50, 40, 5, 5]
    )

    await cog.ml_stats.callback(cog, interaction)

    call_kwargs = interaction.followup.send.call_args
    embed = call_kwargs.kwargs.get("embed") or call_kwargs[1].get("embed")
    assert "Need 40 more labeled samples" in embed.footer.text


@pytest.mark.asyncio
async def test_ml_stats_footer_shows_ready_when_enough_labels():
    #When labeled >= 50, footer should show ready to train.
    cog, bot = _make_cog()
    interaction = _make_interaction()

    # total=200, unlabeled=100, toxic=60, flagged=40 → labeled=100
    bot.db.ml_training_data.count_documents = AsyncMock(
        side_effect=[200, 100, 60, 40]
    )

    await cog.ml_stats.callback(cog, interaction)

    call_kwargs = interaction.followup.send.call_args
    embed = call_kwargs.kwargs.get("embed") or call_kwargs[1].get("embed")
    assert "Ready to train!" in embed.footer.text


@pytest.mark.asyncio
async def test_ml_stats_shows_heuristic_predictor():
    cog, bot = _make_cog()
    bot.collector.predictor = HeuristicPredictor()
    interaction = _make_interaction()

    bot.db.ml_training_data.count_documents = AsyncMock(
        side_effect=[10, 10, 0, 0]
    )

    await cog.ml_stats.callback(cog, interaction)

    call_kwargs = interaction.followup.send.call_args
    embed = call_kwargs.kwargs.get("embed") or call_kwargs[1].get("embed")
    field_values = {f.name: f.value for f in embed.fields}
    assert field_values["Predictor"] == "Heuristic (no model trained yet)"


@pytest.mark.asyncio
async def test_ml_stats_shows_trained_predictor():
    cog, bot = _make_cog()
    bot.collector.predictor = TrainedPredictor("fake_model.joblib")
    interaction = _make_interaction()

    bot.db.ml_training_data.count_documents = AsyncMock(
        side_effect=[10, 10, 0, 0]
    )

    await cog.ml_stats.callback(cog, interaction)

    call_kwargs = interaction.followup.send.call_args
    embed = call_kwargs.kwargs.get("embed") or call_kwargs[1].get("embed")
    field_values = {f.name: f.value for f in embed.fields}
    assert field_values["Predictor"] == "Trained Model"

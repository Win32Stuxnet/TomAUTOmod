from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from bot.bot import ModBot
from bot.ml.predictor import HeuristicPredictor, TrainedPredictor
from bot.utils.embeds import error_embed, success_embed

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "model.joblib"


class ML(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot

    ml_group = app_commands.Group(
        name="ml", description="Machine learning stats and tools",
        default_permissions=discord.Permissions(administrator=True),
    )

    @ml_group.command(name="stats", description="Show ML training data statistics")
    async def ml_stats(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)

        db = self.bot.db
        guild_id = interaction.guild.id

        total = await db.ml_training_data.count_documents({"guild_id": guild_id})
        if total == 0:
            return await interaction.followup.send(
                embed=error_embed(
                    "No ML data collected yet. Enable collection with `/config mlconsent enabled:True`."
                )
            )

        unlabeled = await db.ml_training_data.count_documents({"guild_id": guild_id, "label": None})
        safe = await db.ml_training_data.count_documents({"guild_id": guild_id, "label": "safe"})
        toxic = await db.ml_training_data.count_documents({"guild_id": guild_id, "label": "toxic"})
        flagged = await db.ml_training_data.count_documents({"guild_id": guild_id, "label": "flagged"})
        labeled = safe + toxic + flagged

        predictor = type(self.bot.collector.predictor).__name__
        if isinstance(self.bot.collector.predictor, TrainedPredictor):
            predictor_status = "Trained Model"
        elif isinstance(self.bot.collector.predictor, HeuristicPredictor):
            predictor_status = "Heuristic (no model trained yet)"
        else:
            predictor_status = predictor

        embed = discord.Embed(
            title="ML Training Data",
            color=discord.Color.blurple(),
        )
        embed.add_field(name="Total Samples", value=f"{total:,}", inline=True)
        embed.add_field(name="Unlabeled", value=f"{unlabeled:,}", inline=True)
        embed.add_field(name="Labeled", value=f"{labeled:,}", inline=True)
        embed.add_field(name="Safe", value=f"{safe:,}", inline=True)
        embed.add_field(name="Flagged", value=f"{flagged:,}", inline=True)
        embed.add_field(name="Toxic", value=f"{toxic:,}", inline=True)
        embed.add_field(name="Predictor", value=predictor_status, inline=False)

        if labeled < 50:
            embed.set_footer(text=f"Need {50 - labeled} more labeled samples before training is possible.")
        else:
            embed.set_footer(text=f"Ready to train! Use /ml train")

        await interaction.followup.send(embed=embed)

    @ml_group.command(name="train", description="Train the ML model from labeled data")
    async def ml_train(self, interaction: discord.Interaction) -> None:
        if interaction.user.id not in self.bot.owner_ids:
            return await interaction.response.send_message(
                embed=error_embed("Only bot owners can trigger training."),
                ephemeral=True,
            )

        await interaction.response.defer(ephemeral=True)

        labeled = await self.bot.db.ml_training_data.count_documents({"label": {"$ne": None}})
        if labeled < 50:
            return await interaction.followup.send(
                embed=error_embed(f"Need at least 50 labeled samples to train. Currently have {labeled}.")
            )

        try:
            from bot.ml.train import export_data, train as train_model

            X, y = await export_data(
                self.bot._settings.mongodb_uri,
                self.bot._settings.mongodb_db_name,
            )

            await asyncio.get_event_loop().run_in_executor(
                None, partial(train_model, X, y, MODEL_PATH),
            )

            self.bot.collector.predictor = TrainedPredictor(MODEL_PATH)
            await self.bot.collector.predictor.load()

            await interaction.followup.send(
                embed=success_embed(
                    f"Model trained on **{len(y)}** samples and loaded.\n"
                    f"Predictor switched to **TrainedPredictor**."
                )
            )
        except SystemExit:
            await interaction.followup.send(
                embed=error_embed("Training failed: not enough labeled data.")
            )
        except Exception as e:
            log.exception("Training failed")
            await interaction.followup.send(
                embed=error_embed(f"Training failed: {e}")
            )


async def setup(bot: ModBot) -> None:
    await bot.add_cog(ML(bot))

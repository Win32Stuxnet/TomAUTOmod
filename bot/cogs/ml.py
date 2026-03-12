from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from bot.bot import ModBot
from bot.ml.predictor import HeuristicPredictor, TrainedPredictor
from bot.utils.embeds import error_embed


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
        toxic = await db.ml_training_data.count_documents({"guild_id": guild_id, "label": "toxic"})
        flagged = await db.ml_training_data.count_documents({"guild_id": guild_id, "label": "flagged"})
        labeled = toxic + flagged

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
        embed.add_field(name="Unlabeled (safe)", value=f"{unlabeled:,}", inline=True)
        embed.add_field(name="Labeled", value=f"{labeled:,}", inline=True)
        embed.add_field(name="Toxic", value=f"{toxic:,}", inline=True)
        embed.add_field(name="Flagged", value=f"{flagged:,}", inline=True)
        embed.add_field(name="Predictor", value=predictor_status, inline=True)

        if labeled < 50:
            embed.set_footer(text=f"Need {50 - labeled} more labeled samples before training is possible.")
        else:
            embed.set_footer(text="Ready to train! Run: python -m bot.ml.train")

        await interaction.followup.send(embed=embed)


async def setup(bot: ModBot) -> None:
    await bot.add_cog(ML(bot))

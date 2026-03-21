from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands, tasks

from bot.aggression.embedder import Embedder
from bot.aggression.predictor import AggressionPredictor, SeedPredictor, TrainedAggressionPredictor
from bot.aggression.seeds import DEFAULT_SEEDS
from bot.aggression.tracker import StrikeTracker
from bot.bot import ModBot
from bot.models.guild_config import GuildConfig
from bot.services.config_service import ConfigService
from bot.utils.embeds import audit_embed, error_embed, success_embed
from bot.views.aggression_label import AggressionLabelView

log = logging.getLogger(__name__)

AGGRESSION_MODEL_PATH = Path(__file__).parent.parent.parent / "aggression_model.joblib"


class Aggression(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.embedder = Embedder()
        self.predictor: AggressionPredictor | None = None
        self.tracker = StrikeTracker(bot.db)
        self.config_svc = ConfigService(bot.db)
        self._ready = False

    async def cog_load(self) -> None:
        await self.embedder.setup()

        if AGGRESSION_MODEL_PATH.exists():
            self.predictor = TrainedAggressionPredictor(AGGRESSION_MODEL_PATH)
        else:
            self.predictor = SeedPredictor(self.embedder, seeds=DEFAULT_SEEDS)

        await self.predictor.load()
        self.cleanup_old_strikes.start()
        self._ready = True
        log.info("Aggression cog ready with %s", type(self.predictor).__name__)

    async def cog_unload(self) -> None:
        self._ready = False
        self.cleanup_old_strikes.cancel()

    aggression_group = app_commands.Group(
        name="aggression",
        description="Aggression detection stats and tools",
        default_permissions=discord.Permissions(administrator=True),
    )
    seeds_group = app_commands.Group(
        name="seeds",
        description="Manage aggression seed phrases",
        parent=aggression_group,
    )

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot or not message.content.strip():
            return
        if not self._ready or self.predictor is None:
            return

        config = await self.config_svc.get(message.guild.id)
        if not config.ml_consent:
            return

        automod_result = await self.bot.collector.process_message(message)
        if automod_result is not None:
            return

        embedding = await self.embedder.embed(message.content)
        result = await self.predictor.predict(embedding)
        if result.label != "aggressive":
            return

        await self.tracker.record_strike(
            guild_id=message.guild.id,
            user_id=message.author.id,
            message_id=message.id,
            channel_id=message.channel.id,
            score=result.confidence,
            content=message.content,
            embedding=embedding.tolist(),
        )

        crossed_threshold = await self.tracker.check_threshold(
            message.guild.id,
            message.author.id,
            config.aggression_strike_count,
            config.aggression_window_hours,
        )
        if not crossed_threshold:
            return

        if await self.tracker.has_alert_in_window(
            message.guild.id,
            message.author.id,
            config.aggression_window_hours,
        ):
            return

        sent = await self._send_alert(message, config, result.confidence)
        if sent:
            await self.tracker.mark_alerted(
                message.guild.id,
                message.author.id,
                config.aggression_window_hours,
            )

    async def _send_alert(self, message: discord.Message, config: GuildConfig, score: float) -> bool:
        channel = self._get_alert_channel(message.guild, config)
        if channel is None:
            return False

        strikes = await self.tracker.get_recent_strikes(
            message.guild.id,
            message.author.id,
            config.aggression_window_hours,
            limit=max(config.aggression_strike_count, 5),
        )
        strike_lines = []
        for strike in strikes:
            link = self._jump_url(message.guild.id, strike["channel_id"], strike["message_id"])
            content = (strike.get("content") or "*content unavailable*").replace("\n", " ")
            strike_lines.append(f"[{strike['score']:.0%}] {content[:90]} {link}")

        embed = audit_embed(
            title="Aggression Alert",
            description=(
                f"**User:** {message.author.mention}\n"
                f"**Channel:** {message.channel.mention}\n"
                f"**Score:** {score:.0%}\n"
                f"**Threshold:** {config.aggression_strike_count} strikes in {config.aggression_window_hours}h\n"
                f"**Latest Message:** {message.content[:500]}"
            ),
            user=message.author,
        )
        if strike_lines:
            embed.add_field(
                name="Recent Strikes",
                value="\n".join(strike_lines)[:1024],
                inline=False,
            )
        embed.set_footer(text=f"Message ID: {message.id}")

        view = AggressionLabelView(self.bot, message.guild.id, message.id)
        await channel.send(embed=embed, view=view)
        return True

    def _get_alert_channel(self, guild: discord.Guild, config: GuildConfig) -> discord.TextChannel | None:
        channel_id = (
            config.aggression_channel_id
            or config.review_channel_id
            or config.audit_log_channel_id
        )
        if not channel_id:
            return None

        channel = guild.get_channel(channel_id)
        return channel if isinstance(channel, discord.TextChannel) else None

    @staticmethod
    def _jump_url(guild_id: int, channel_id: int, message_id: int) -> str:
        return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

    @aggression_group.command(name="stats", description="Show aggression detection statistics")
    async def aggression_stats(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)

        db = self.bot.db
        guild_id = interaction.guild.id
        total_strikes = await db.aggression_strikes.count_documents({"guild_id": guild_id})
        total_samples = await db.aggression_training_data.count_documents({"guild_id": guild_id})
        labeled = await db.aggression_training_data.count_documents(
            {"guild_id": guild_id, "label": {"$ne": None}}
        )

        if isinstance(self.predictor, TrainedAggressionPredictor):
            predictor_status = "Trained Model"
        elif isinstance(self.predictor, SeedPredictor):
            predictor_status = "Seed Phrases (no model trained yet)"
        else:
            predictor_status = type(self.predictor).__name__

        embed = discord.Embed(title="Aggression Detection", color=discord.Color.orange())
        embed.add_field(name="Total Strikes", value=f"{total_strikes:,}", inline=True)
        embed.add_field(name="Training Samples", value=f"{total_samples:,}", inline=True)
        embed.add_field(name="Labeled", value=f"{labeled:,}", inline=True)
        embed.add_field(name="Predictor", value=predictor_status, inline=False)

        if labeled < 50:
            embed.set_footer(text=f"Need {50 - labeled} more labeled samples before training.")
        else:
            embed.set_footer(text="Ready to train! Use /aggression train")

        await interaction.followup.send(embed=embed)

    @aggression_group.command(name="train", description="Train the aggression model from labeled data")
    async def aggression_train(self, interaction: discord.Interaction) -> None:
        if interaction.user.id not in self.bot.owner_ids:
            return await interaction.response.send_message(
                embed=error_embed("Only bot owners can trigger training."),
                ephemeral=True,
            )

        await interaction.response.defer(ephemeral=True)

        labeled = await self.bot.db.aggression_training_data.count_documents(
            {"label": {"$ne": None}}
        )
        if labeled < 50:
            return await interaction.followup.send(
                embed=error_embed(f"Need at least 50 labeled samples to train. Currently have {labeled}.")
            )

        try:
            from bot.aggression.train import export_data, train as train_model

            X, y = await export_data(
                self.bot._settings.mongodb_uri,
                self.bot._settings.mongodb_db_name,
            )

            await asyncio.get_running_loop().run_in_executor(
                None,
                partial(train_model, X, y, AGGRESSION_MODEL_PATH),
            )

            self.predictor = TrainedAggressionPredictor(AGGRESSION_MODEL_PATH)
            await self.predictor.load()

            await interaction.followup.send(
                embed=success_embed(
                    f"Aggression model trained on **{len(y)}** samples and loaded.\n"
                    "Predictor switched to **TrainedAggressionPredictor**."
                )
            )
        except Exception as exc:
            log.exception("Aggression training failed")
            await interaction.followup.send(embed=error_embed(f"Training failed: {exc}"))

    @seeds_group.command(name="list", description="List the current aggression seed phrases")
    async def list_seeds(self, interaction: discord.Interaction) -> None:
        if not isinstance(self.predictor, SeedPredictor):
            return await interaction.response.send_message(
                embed=error_embed("A trained aggression model is active, so seed phrases are not in use."),
                ephemeral=True,
            )

        phrases = list(self.predictor.seeds)
        embed = discord.Embed(
            title=f"Seed Phrases ({len(phrases)})",
            description="\n".join(f"- {phrase}" for phrase in phrases[:40]) or "No seed phrases configured.",
            color=discord.Color.orange(),
        )
        if len(phrases) > 40:
            embed.set_footer(text=f"Showing 40 of {len(phrases)} seeds.")
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @seeds_group.command(name="add", description="Add an aggression seed phrase")
    @app_commands.describe(phrase="A phrase that should count as aggressive")
    async def add_seed(self, interaction: discord.Interaction, phrase: str) -> None:
        if not isinstance(self.predictor, SeedPredictor):
            return await interaction.response.send_message(
                embed=error_embed("A trained aggression model is active, so seed phrases are not in use."),
                ephemeral=True,
            )

        await interaction.response.defer(ephemeral=True)
        self.predictor.add_seed(phrase.strip())
        await self.predictor.load()
        await interaction.followup.send(
            embed=success_embed(f"Added seed phrase. Total seeds: {len(self.predictor.seeds)}")
        )

    @seeds_group.command(name="remove", description="Remove an aggression seed phrase")
    @app_commands.describe(phrase="The seed phrase to remove")
    async def remove_seed(self, interaction: discord.Interaction, phrase: str) -> None:
        if not isinstance(self.predictor, SeedPredictor):
            return await interaction.response.send_message(
                embed=error_embed("A trained aggression model is active, so seed phrases are not in use."),
                ephemeral=True,
            )

        removed = self.predictor.remove_seed(phrase.strip())
        if not removed:
            return await interaction.response.send_message(
                embed=error_embed("That seed phrase was not found."),
                ephemeral=True,
            )

        await interaction.response.defer(ephemeral=True)
        await self.predictor.load()
        await interaction.followup.send(
            embed=success_embed(f"Removed seed phrase. Total seeds: {len(self.predictor.seeds)}")
        )

    @tasks.loop(hours=1)
    async def cleanup_old_strikes(self) -> None:
        now = datetime.now(timezone.utc)
        for guild in self.bot.guilds:
            try:
                config = await self.config_svc.get(guild.id)
                cutoff = now - timedelta(days=config.log_retention_days)
                result = await self.bot.db.aggression_strikes.delete_many(
                    {"guild_id": guild.id, "created_at": {"$lt": cutoff}}
                )
                if result.deleted_count:
                    log.info(
                        "Cleaned %d expired aggression strikes for guild %d",
                        result.deleted_count,
                        guild.id,
                    )
            except Exception:
                log.exception("Failed cleaning aggression strikes for guild %d", guild.id)

    @cleanup_old_strikes.before_loop
    async def before_cleanup_old_strikes(self) -> None:
        await self.bot.wait_until_ready()


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Aggression(bot))

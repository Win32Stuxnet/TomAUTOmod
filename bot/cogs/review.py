from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import discord
from discord import app_commands
from discord.ext import commands, tasks

from bot.bot import ModBot
from bot.ml.features import extract_features
from bot.models.ml_data import MLTrainingSample
from bot.services.config_service import ConfigService
from bot.utils.embeds import success_embed, error_embed
from bot.utils.permissions import is_mod

log = logging.getLogger(__name__)

BATCH_SIZE = 20
BACKFILL_HOURS = 48
BACKFILL_CHAN_LIMIT = 200  # max messages to scan per channel



class ReviewView(discord.ui.View):
    def __init__(self, bot: ModBot, samples: list[dict], *, author_id: int) -> None:
        super().__init__(timeout=300)
        self.bot = bot
        self.samples = samples
        self.author_id = author_id
        self.index = 0
        self.labeled = 0

    def current_embed(self) -> discord.Embed:
        sample = self.samples[self.index]
        prediction = sample.get("prediction") or "unknown"
        confidence = sample.get("confidence", 0.0)

        color = {
            "toxic": discord.Color.red(),
            "flagged": discord.Color.gold(),
        }.get(prediction, discord.Color.greyple())

        embed = discord.Embed(
            title=f"Review Queue ({self.index + 1}/{len(self.samples)})",
            description=sample.get("content", "*no content*")[:1500],
            color=color,
        )
        embed.add_field(name="Author", value=f"<@{sample['user_id']}>", inline=True)
        channel_id = sample.get("channel_id")
        if channel_id:
            embed.add_field(name="Channel", value=f"<#{channel_id}>", inline=True)
        embed.add_field(
            name="Prediction",
            value=f"{prediction} ({confidence:.0%})",
            inline=True,
        )
        created = sample.get("created_at")
        if isinstance(created, datetime):
            embed.add_field(name="Sent", value=discord.utils.format_dt(created, "R"), inline=True)
        embed.set_footer(text=f"Labeled this session: {self.labeled}")
        return embed

    async def _label(self, interaction: discord.Interaction, label: str) -> None:
        if interaction.user.id != self.author_id:
            return await interaction.response.send_message("Not your review session.", ephemeral=True)

        sample = self.samples[self.index]
        await self.bot.db.ml_training_data.update_one(
            {"guild_id": sample["guild_id"], "message_id": sample["message_id"]},
            {"$set": {
                "label": label,
                "reviewed_by": interaction.user.id,
                "reviewed_at": datetime.now(timezone.utc),
            }},
        )
        self.labeled += 1
        self.index += 1

        if self.index >= len(self.samples):
            for child in self.children:
                if isinstance(child, discord.ui.Button):
                    child.disabled = True
            await interaction.response.edit_message(
                embed=success_embed(
                    f"Batch complete! Labeled **{self.labeled}** messages.\n"
                    f"Run `/review queue` for more."
                ),
                view=self,
            )
        else:
            await interaction.response.edit_message(embed=self.current_embed(), view=self)

    @discord.ui.button(label="Safe", style=discord.ButtonStyle.success)
    async def safe_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "safe")

    @discord.ui.button(label="Flag", style=discord.ButtonStyle.secondary)
    async def flag_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "flagged")

    @discord.ui.button(label="Toxic", style=discord.ButtonStyle.danger)
    async def toxic_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "toxic")

    @discord.ui.button(label="Skip", style=discord.ButtonStyle.secondary)
    async def skip_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if interaction.user.id != self.author_id:
            return await interaction.response.send_message("Not your review session.", ephemeral=True)
        self.index += 1
        if self.index >= len(self.samples):
            for child in self.children:
                if isinstance(child, discord.ui.Button):
                    child.disabled = True
            await interaction.response.edit_message(
                embed=success_embed(
                    f"End of batch. Labeled **{self.labeled}** messages.\n"
                    f"Run `/review queue` for more."
                ),
                view=self,
            )
        else:
            await interaction.response.edit_message(embed=self.current_embed(), view=self)

    async def on_timeout(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True



class SingleLabelView(discord.ui.View):
    def __init__(self, bot: ModBot, query: dict, *, author_id: int) -> None:
        super().__init__(timeout=120)
        self.bot = bot
        self.query = query
        self.author_id = author_id

    async def _label(self, interaction: discord.Interaction, label: str) -> None:
        if interaction.user.id != self.author_id:
            return await interaction.response.send_message("Not yours.", ephemeral=True)
        await self.bot.db.ml_training_data.update_one(
            self.query,
            {"$set": {
                "label": label,
                "reviewed_by": interaction.user.id,
                "reviewed_at": datetime.now(timezone.utc),
            }},
        )
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        await interaction.response.edit_message(
            embed=success_embed(f"Message labeled as **{label}**."),
            view=self,
        )

    @discord.ui.button(label="Safe", style=discord.ButtonStyle.success)
    async def safe_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "safe")

    @discord.ui.button(label="Flag", style=discord.ButtonStyle.secondary)
    async def flag_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "flagged")

    @discord.ui.button(label="Toxic", style=discord.ButtonStyle.danger)
    async def toxic_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "toxic")



class Review(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.config_svc = ConfigService(bot.db)
        self.ctx_menu = app_commands.ContextMenu(
            name="Label for ML",
            callback=self.label_message_ctx,
        )
        self.bot.tree.add_command(self.ctx_menu)

    async def cog_load(self) -> None:
        self.cleanup_task.start()

    async def cog_unload(self) -> None:
        self.bot.tree.remove_command(self.ctx_menu.name, type=self.ctx_menu.type)
        self.cleanup_task.cancel()

    # ---- slash commands ---------------------------------------------------

    review_group = app_commands.Group(
        name="review",
        description="Review and label messages for ML training",
        default_permissions=discord.Permissions(moderate_members=True),
    )

    async def _backfill_recent(self, guild: discord.Guild) -> int:
        """Queue messages from the last 48h that aren't already in the DB."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=BACKFILL_HOURS)
        inserted = 0

        for channel in guild.text_channels:
            perms = channel.permissions_for(guild.me)
            if not perms.read_message_history:
                continue
            try:
                async for message in channel.history(
                    after=cutoff, limit=BACKFILL_CHAN_LIMIT, oldest_first=False,
                ):
                    if message.author.bot or not message.content:
                        continue
                    exists = await self.bot.db.ml_training_data.find_one(
                        {"guild_id": guild.id, "message_id": message.id},
                        projection={"_id": 1},
                    )
                    if exists:
                        continue
                    features = extract_features(message.content)
                    sample = MLTrainingSample(
                        guild_id=guild.id,
                        message_id=message.id,
                        channel_id=channel.id,
                        user_id=message.author.id,
                        content=message.content[:300],
                        features=features,
                        created_at=message.created_at,
                    )
                    await self.bot.db.ml_training_data.insert_one(sample.to_doc())
                    inserted += 1
            except discord.Forbidden:
                continue
            except Exception:
                log.exception("Backfill failed for channel %d", channel.id)
        return inserted

    @review_group.command(name="queue", description="Review unreviewed messages for ML training")
    @app_commands.describe(sort="Sort order for the queue")
    @app_commands.choices(sort=[
        app_commands.Choice(name="Highest confidence first", value="confidence"),
        app_commands.Choice(name="Most recent first", value="recent"),
    ])
    async def review_queue(
        self,
        interaction: discord.Interaction,
        sort: app_commands.Choice[str] | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)

        # Backfill recent channel history (last 48h) before querying
        config = await self.config_svc.get(interaction.guild.id)
        if config.ml_consent:
            backfilled = await self._backfill_recent(interaction.guild)
            if backfilled:
                log.info("Backfilled %d messages for guild %d", backfilled, interaction.guild.id)

        sort_key = sort.value if sort else "confidence"
        sort_order = (
            [("confidence", -1), ("created_at", -1)]
            if sort_key == "confidence"
            else [("created_at", -1)]
        )

        cursor = self.bot.db.ml_training_data.find(
            {
                "guild_id": interaction.guild.id,
                "label": None,
                "content": {"$nin": [None, ""]},
            },
            sort=sort_order,
            limit=BATCH_SIZE,
        )
        samples = await cursor.to_list(length=BATCH_SIZE)

        if not samples:
            return await interaction.followup.send(
                embed=error_embed("No unreviewed messages in the queue.")
            )

        view = ReviewView(self.bot, samples, author_id=interaction.user.id)
        await interaction.followup.send(embed=view.current_embed(), view=view)

    @review_group.command(name="stats", description="Show review and labeling statistics")
    async def review_stats(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)

        db = self.bot.db
        gid = interaction.guild.id

        total = await db.ml_training_data.count_documents({"guild_id": gid})
        if total == 0:
            return await interaction.followup.send(
                embed=error_embed("No ML data collected yet. Enable with `/config mlconsent enabled:True`.")
            )

        unreviewed = await db.ml_training_data.count_documents({"guild_id": gid, "label": None})
        safe = await db.ml_training_data.count_documents({"guild_id": gid, "label": "safe"})
        flagged = await db.ml_training_data.count_documents({"guild_id": gid, "label": "flagged"})
        toxic = await db.ml_training_data.count_documents({"guild_id": gid, "label": "toxic"})
        reviewable = await db.ml_training_data.count_documents({
            "guild_id": gid, "label": None, "content": {"$nin": [None, ""]},
        })

        labeled = safe + flagged + toxic

        embed = discord.Embed(title="Review Statistics", color=discord.Color.blurple())
        embed.add_field(name="Total Samples", value=f"{total:,}", inline=True)
        embed.add_field(name="Unreviewed", value=f"{unreviewed:,}", inline=True)
        embed.add_field(name="Reviewable", value=f"{reviewable:,}", inline=True)
        embed.add_field(name="Safe", value=f"{safe:,}", inline=True)
        embed.add_field(name="Flagged", value=f"{flagged:,}", inline=True)
        embed.add_field(name="Toxic", value=f"{toxic:,}", inline=True)

        if labeled < 50:
            embed.set_footer(text=f"Need {50 - labeled} more labeled samples before training.")
        else:
            embed.set_footer(text=f"{labeled} labeled samples - ready to train with /ml train")

        await interaction.followup.send(embed=embed)

    # ---- context menu -----------------------------------------------------

    async def label_message_ctx(
        self, interaction: discord.Interaction, message: discord.Message,
    ) -> None:
        if not is_mod(interaction.user):
            return await interaction.response.send_message(
                "You need Moderate Members permission.", ephemeral=True,
            )

        doc = await self.bot.db.ml_training_data.find_one({
            "guild_id": interaction.guild.id,
            "message_id": message.id,
        })

        if not doc:
            features = extract_features(message.content)
            sample = MLTrainingSample(
                guild_id=interaction.guild.id,
                message_id=message.id,
                channel_id=message.channel.id,
                user_id=message.author.id,
                features=features,
                content=message.content[:300],
            )
            await self.bot.db.ml_training_data.insert_one(sample.to_doc())
            doc = sample.to_doc()

        if doc.get("label"):
            return await interaction.response.send_message(
                f"Already labeled as **{doc['label']}**.", ephemeral=True,
            )

        content = doc.get("content") or message.content[:300]
        prediction = doc.get("prediction") or "unknown"
        confidence = doc.get("confidence", 0.0)

        embed = discord.Embed(
            title="Label Message",
            description=content[:1500],
            color=discord.Color.blurple(),
        )
        embed.add_field(name="Author", value=f"<@{message.author.id}>", inline=True)
        embed.add_field(name="Channel", value=f"<#{message.channel.id}>", inline=True)
        if prediction != "unknown":
            embed.add_field(name="ML Prediction", value=f"{prediction} ({confidence:.0%})", inline=True)

        query = {"guild_id": interaction.guild.id, "message_id": message.id}
        view = SingleLabelView(self.bot, query, author_id=interaction.user.id)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    # cleanup -----------------------------------------------

    @tasks.loop(hours=24)
    async def cleanup_task(self) -> None:
        """Strip content from old unreviewed samples to save storage."""
        for guild in self.bot.guilds:
            try:
                config = await self.config_svc.get(guild.id)
                cutoff = datetime.now(timezone.utc) - timedelta(days=config.log_retention_days)

                result = await self.bot.db.ml_training_data.update_many(
                    {
                        "guild_id": guild.id,
                        "label": None,
                        "created_at": {"$lt": cutoff},
                        "content": {"$nin": [None, ""]},
                    },
                    {"$set": {"content": ""}},
                )
                if result.modified_count:
                    log.info(
                        "Cleaned content from %d old entries in guild %d",
                        result.modified_count, guild.id,
                    )
            except Exception:
                log.exception("Cleanup failed for guild %d", guild.id)

    @cleanup_task.before_loop
    async def before_cleanup(self) -> None:
        await self.bot.wait_until_ready()


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Review(bot))

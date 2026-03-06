from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import traceback

import discord
from discord import app_commands
from discord.ext import commands

from bot.database import Database
from bot.ml.collector import Collector

if TYPE_CHECKING:
    from config import Settings

log = logging.getLogger(__name__)

COGS_DIR = Path(__file__).parent / "cogs"


class ModBot(commands.Bot):
    db: Database
    collector: Collector

    def __init__(self, *, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            owner_ids=settings.owner_ids,
        )
        self._settings = settings

    async def setup_hook(self) -> None:
        self.db = Database(self._settings.mongodb_uri, self._settings.mongodb_db_name)
        await self.db.create_indexes()
        log.info("Database connected and indexes ensured.")

        self.collector = Collector(self)
        await self.collector.setup()
        await self.setup_hook_tree_error()

        for cog_file in sorted(COGS_DIR.glob("*.py")):
            if cog_file.name.startswith("_"):
                continue
            ext = f"bot.cogs.{cog_file.stem}"
            try:
                await self.load_extension(ext)
                log.info("Loaded extension %s", ext)
            except Exception:
                log.exception("Failed to load extension %s", ext)

    async def on_ready(self) -> None:
        log.info("Logged in as %s (ID: %s)", self.user, self.user.id)

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.NotOwner):
            return
        if isinstance(error, commands.CommandNotFound):
            return
        log.error("Command error in %s: %s", ctx.command, error)

    async def setup_hook_tree_error(self) -> None:

        @self.tree.error
        async def on_app_command_error(
            interaction: discord.Interaction, error: app_commands.AppCommandError
        ) -> None:
            if isinstance(error, app_commands.MissingPermissions):
                msg = "You don't have permission to use this command."
            elif isinstance(error, app_commands.CommandOnCooldown):
                msg = f"Cooldown: try again in {error.retry_after:.0f}s."
            else:
                msg = "Something went wrong."
                log.error("Slash command error: %s\n%s", error, traceback.format_exc())

            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)

    async def close(self) -> None:
        if hasattr(self, "db"):
            self.db.close()
        await super().close()

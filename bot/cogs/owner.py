from __future__ import annotations

import logging
from typing import Literal

import discord
from discord.ext import commands

from bot.bot import ModBot

log = logging.getLogger(__name__)


class Owner(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot

    @commands.command(name="sync")
    @commands.is_owner()
    async def sync_commands(
        self,
        ctx: commands.Context,
        scope: Literal["global", "guild"] = "global",
    ) -> None:
        if scope == "guild" and ctx.guild:
            self.bot.tree.copy_global_to(guild=ctx.guild)
            synced = await self.bot.tree.sync(guild=ctx.guild)
            await ctx.send(f"Synced {len(synced)} commands to this guild.")
        else:
            synced = await self.bot.tree.sync()
            await ctx.send(f"Synced {len(synced)} commands globally.")
        log.info("Synced %d commands (%s)", len(synced), scope)

    @commands.command(name="reload")
    @commands.is_owner()
    async def reload_cog(self, ctx: commands.Context, extension: str) -> None:
        ext = f"bot.cogs.{extension}"
        try:
            await self.bot.reload_extension(ext)
            await ctx.send(f"Reloaded `{ext}`.")
        except Exception as e:
            await ctx.send(f"Failed to reload `{ext}`: {e}")

    @commands.command(name="shutdown")
    @commands.is_owner()
    async def shutdown(self, ctx: commands.Context) -> None:
        await ctx.send("Shutting down...")
        await self.bot.close()


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Owner(bot))

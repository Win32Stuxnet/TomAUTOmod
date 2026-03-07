from __future__ import annotations

import logging
from datetime import timedelta

import discord
from discord.ext import commands

from bot.bot import ModBot
from bot.services.config_service import ConfigService
from bot.services.message_cache_service import MessageCacheService
from bot.utils.embeds import audit_embed

log = logging.getLogger(__name__)


class AuditLog(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.config = ConfigService(bot.db)
        self.cache = MessageCacheService(bot.db)

    async def _get_audit_channel(self, guild: discord.Guild) -> discord.TextChannel | None:
        cfg = await self.config.get(guild.id)
        if not cfg.audit_log_channel_id:
            return None
        ch = guild.get_channel(cfg.audit_log_channel_id)
        return ch if isinstance(ch, discord.TextChannel) else None

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot:
            return

        await self.cache.cache_message(message)

        prediction = await self.bot.collector.process_message(message)

        if prediction and prediction.label in ("toxic", "flagged"):
            await self._handle_ml_flag(message, prediction)

    async def _handle_ml_flag(self, message: discord.Message, prediction) -> None:
        channel = await self._get_audit_channel(message.guild)
        if not channel:
            return

        embed = audit_embed(
            title=f"ML Flag: {prediction.label}",
            description=(
                f"**Author:** {message.author.mention}\n"
                f"**Channel:** {message.channel.mention}\n"
                f"**Confidence:** {prediction.confidence:.0%}\n"
                f"**Content:** {message.content[:500]}"
            ),
            user=message.author,
        )
        await channel.send(embed=embed)

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        if not after.guild or after.author.bot:
            return
        if before.content == after.content:
            return

        channel = await self._get_audit_channel(after.guild)
        if not channel:
            return

        embed = audit_embed(
            title="Message Edited",
            description=(
                f"**Author:** {after.author.mention}\n"
                f"**Channel:** {after.channel.mention}\n\n"
                f"**Before:**\n{before.content[:1000]}\n\n"
                f"**After:**\n{after.content[:1000]}"
            ),
            user=after.author,
        )
        embed.set_footer(text=f"Message ID: {after.id}")
        await channel.send(embed=embed)

    @commands.Cog.listener()
    async def on_message_delete(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot:
            return

        channel = await self._get_audit_channel(message.guild)
        if not channel:
            return

        cached = await self.cache.get_cached(message.id)
        content = cached.content if cached else (message.content or "*content unavailable*")

        embed = audit_embed(
            title="Message Deleted",
            description=(
                f"**Author:** {message.author.mention}\n"
                f"**Channel:** {message.channel.mention}\n\n"
                f"**Content:**\n{content[:1500]}"
            ),
            user=message.author,
        )
        if cached and cached.attachments:
            embed.add_field(
                name="Attachments",
                value="\n".join(cached.attachments[:5]),
                inline=False,
            )
        embed.set_footer(text=f"Message ID: {message.id}")
        await channel.send(embed=embed)

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member) -> None:
        channel = await self._get_audit_channel(member.guild)
        if not channel:
            return

        age = discord.utils.utcnow() - member.created_at
        embed = audit_embed(
            title="Member Joined",
            description=(
                f"**User:** {member.mention} ({member})\n"
                f"**Account Age:** {age.days} days\n"
                f"**ID:** {member.id}"
            ),
            user=member,
        )
        await channel.send(embed=embed)

    @commands.Cog.listener()
    async def on_member_remove(self, member: discord.Member) -> None:
        channel = await self._get_audit_channel(member.guild)
        if not channel:
            return

        roles = ", ".join(r.mention for r in member.roles[1:]) or "None"
        embed = audit_embed(
            title="Member Left",
            description=(
                f"**User:** {member} ({member.id})\n"
                f"**Roles:** {roles}"
            ),
            user=member,
        )
        await channel.send(embed=embed)

    @commands.Cog.listener()
    async def on_member_update(self, before: discord.Member, after: discord.Member) -> None:
        if before.roles == after.roles:
            return

        channel = await self._get_audit_channel(after.guild)
        if not channel:
            return

        added = set(after.roles) - set(before.roles)
        removed = set(before.roles) - set(after.roles)
        parts = []
        if added:
            parts.append("**Added:** " + ", ".join(r.mention for r in added))
        if removed:
            parts.append("**Removed:** " + ", ".join(r.mention for r in removed))

        embed = audit_embed(
            title="Member Roles Updated",
            description=f"**User:** {after.mention}\n" + "\n".join(parts),
            user=after,
        )
        await channel.send(embed=embed)


async def setup(bot: ModBot) -> None:
    await bot.add_cog(AuditLog(bot))

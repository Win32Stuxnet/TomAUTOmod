from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord

from bot.constants import Colors

if TYPE_CHECKING:
    from bot.models.cases import Case


def success_embed(description: str, *, title: str | None = None) -> discord.Embed:
    return discord.Embed(title=title, description=description, color=Colors.SUCCESS)


def error_embed(description: str, *, title: str | None = None) -> discord.Embed:
    return discord.Embed(title=title, description=description, color=Colors.ERROR)


def warning_embed(description: str, *, title: str | None = None) -> discord.Embed:
    return discord.Embed(title=title, description=description, color=Colors.WARNING)


def mod_action_embed(
    *,
    action: str,
    user: discord.User | discord.Member,
    moderator: discord.User | discord.Member,
    reason: str | None,
    case_id: int,
) -> discord.Embed:
    embed = discord.Embed(
        title=f"Case #{case_id} | {action}",
        color=Colors.MOD_ACTION,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="User", value=f"{user} ({user.id})", inline=True)
    embed.add_field(name="Moderator", value=f"{moderator} ({moderator.id})", inline=True)
    embed.add_field(name="Reason", value=reason or "No reason provided", inline=False)
    embed.set_thumbnail(url=user.display_avatar.url)
    return embed


def case_embed(case: Case) -> discord.Embed:
    embed = discord.Embed(
        title=f"Case #{case.case_id} | {case.action.title()}",
        color=Colors.MOD_ACTION,
        timestamp=case.created_at,
    )
    embed.add_field(name="User", value=f"<@{case.user_id}> ({case.user_id})", inline=True)
    embed.add_field(name="Moderator", value=f"<@{case.moderator_id}> ({case.moderator_id})", inline=True)
    embed.add_field(name="Reason", value=case.reason or "No reason provided", inline=False)
    if case.duration:
        embed.add_field(name="Duration", value=case.duration, inline=True)
    if case.pardoned:
        embed.add_field(name="Status", value="Pardoned", inline=True)
    return embed


def audit_embed(
    *,
    title: str,
    description: str,
    user: discord.User | discord.Member | None = None,
) -> discord.Embed:
    embed = discord.Embed(
        title=title,
        description=description,
        color=Colors.AUDIT,
        timestamp=datetime.now(timezone.utc),
    )
    if user:
        embed.set_author(name=str(user), icon_url=user.display_avatar.url)
    return embed

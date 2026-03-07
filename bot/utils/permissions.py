from __future__ import annotations

import discord


def can_moderate(
    moderator: discord.Member,
    target: discord.Member,
) -> tuple[bool, str | None]:
    if target.id == moderator.id:
        return False, "You cannot moderate yourself."

    if target.bot:
        return False, "You cannot moderate a bot."

    if target.top_role >= moderator.top_role:
        return False, "Target has an equal or higher role than you."

    guild = moderator.guild
    if guild.me and target.top_role >= guild.me.top_role:
        return False, "Target has an equal or higher role than me."

    if target.id == guild.owner_id:
        return False, "You cannot moderate the server owner."

    return True, None


def is_mod(member: discord.Member) -> bool:
    return member.guild_permissions.moderate_members or member.guild_permissions.administrator

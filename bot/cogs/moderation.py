from __future__ import annotations

import logging
from datetime import timezone

import discord
from discord import app_commands
from discord.ext import commands

from bot.bot import ModBot
from bot.services.case_service import CaseService
from bot.utils.embeds import success_embed, error_embed, case_embed
from bot.utils.permissions import can_moderate
from bot.utils.time_parser import parse_duration, format_duration
from bot.utils.pagination import PaginatorView

log = logging.getLogger(__name__)


class Moderation(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.cases = CaseService(bot)

    async def _dm_user(
        self, user: discord.User | discord.Member, guild: discord.Guild, action: str, reason: str | None
    ) -> None:
        try:
            msg = f"You have been **{action}** in **{guild.name}**."
            if reason:
                msg += f"\n**Reason:** {reason}"
            await user.send(msg)
        except discord.Forbidden:
            pass

    @app_commands.command(name="warn", description="Warn a member")
    @app_commands.describe(member="Member to warn", reason="Reason for the warning")
    @app_commands.default_permissions(moderate_members=True)
    async def warn(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        reason: str | None = None,
    ) -> None:
        ok, err = can_moderate(interaction.user, member)
        if not ok:
            return await interaction.response.send_message(embed=error_embed(err), ephemeral=True)

        case = await self.cases.create_case(
            guild=interaction.guild, action="warn", user=member,
            moderator=interaction.user, reason=reason,
        )
        await self._dm_user(member, interaction.guild, "warned", reason)
        await interaction.response.send_message(
            embed=success_embed(f"**{member}** warned. (Case #{case.case_id})")
        )

    @app_commands.command(name="kick", description="Kick a member from the server")
    @app_commands.describe(member="Member to kick", reason="Reason for the kick")
    @app_commands.default_permissions(kick_members=True)
    async def kick(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        reason: str | None = None,
    ) -> None:
        ok, err = can_moderate(interaction.user, member)
        if not ok:
            return await interaction.response.send_message(embed=error_embed(err), ephemeral=True)

        await self._dm_user(member, interaction.guild, "kicked", reason)
        await member.kick(reason=reason)
        case = await self.cases.create_case(
            guild=interaction.guild, action="kick", user=member,
            moderator=interaction.user, reason=reason,
        )
        await interaction.response.send_message(
            embed=success_embed(f"**{member}** kicked. (Case #{case.case_id})")
        )

    @app_commands.command(name="ban", description="Ban a member from the server")
    @app_commands.describe(member="Member to ban", reason="Reason for the ban", delete_days="Days of messages to delete (0-7)")
    @app_commands.default_permissions(ban_members=True)
    async def ban(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        reason: str | None = None,
        delete_days: app_commands.Range[int, 0, 7] = 0,
    ) -> None:
        ok, err = can_moderate(interaction.user, member)
        if not ok:
            return await interaction.response.send_message(embed=error_embed(err), ephemeral=True)

        await self._dm_user(member, interaction.guild, "banned", reason)
        await member.ban(reason=reason, delete_message_days=delete_days)
        case = await self.cases.create_case(
            guild=interaction.guild, action="ban", user=member,
            moderator=interaction.user, reason=reason,
        )
        await interaction.response.send_message(
            embed=success_embed(f"**{member}** banned. (Case #{case.case_id})")
        )

    @app_commands.command(name="unban", description="Unban a user by ID")
    @app_commands.describe(user_id="ID of the user to unban", reason="Reason for the unban")
    @app_commands.default_permissions(ban_members=True)
    async def unban(
        self,
        interaction: discord.Interaction,
        user_id: str,
        reason: str | None = None,
    ) -> None:
        try:
            uid = int(user_id)
            user = await self.bot.fetch_user(uid)
        except (ValueError, discord.NotFound):
            return await interaction.response.send_message(
                embed=error_embed("Invalid user ID or user not found."), ephemeral=True
            )

        try:
            await interaction.guild.unban(user, reason=reason)
        except discord.NotFound:
            return await interaction.response.send_message(
                embed=error_embed("That user is not banned."), ephemeral=True
            )

        case = await self.cases.create_case(
            guild=interaction.guild, action="unban", user=user,
            moderator=interaction.user, reason=reason,
        )
        await interaction.response.send_message(
            embed=success_embed(f"**{user}** unbanned. (Case #{case.case_id})")
        )

    @app_commands.command(name="timeout", description="Timeout a member")
    @app_commands.describe(member="Member to timeout", duration="Duration (e.g. 2h30m)", reason="Reason")
    @app_commands.default_permissions(moderate_members=True)
    async def timeout(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        duration: str,
        reason: str | None = None,
    ) -> None:
        ok, err = can_moderate(interaction.user, member)
        if not ok:
            return await interaction.response.send_message(embed=error_embed(err), ephemeral=True)

        delta = parse_duration(duration)
        if not delta:
            return await interaction.response.send_message(
                embed=error_embed("Invalid duration. Use formats like `2h30m`, `1d`, `30m`."), ephemeral=True
            )

        await member.timeout(delta, reason=reason)
        formatted = format_duration(delta)
        case = await self.cases.create_case(
            guild=interaction.guild, action="timeout", user=member,
            moderator=interaction.user, reason=reason, duration=formatted,
        )
        await self._dm_user(member, interaction.guild, f"timed out for {formatted}", reason)
        await interaction.response.send_message(
            embed=success_embed(f"**{member}** timed out for {formatted}. (Case #{case.case_id})")
        )

    @app_commands.command(name="untimeout", description="Remove a member's timeout")
    @app_commands.describe(member="Member to untimeout", reason="Reason")
    @app_commands.default_permissions(moderate_members=True)
    async def untimeout(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        reason: str | None = None,
    ) -> None:
        if not member.is_timed_out():
            return await interaction.response.send_message(
                embed=error_embed("That member is not timed out."), ephemeral=True
            )

        await member.timeout(None, reason=reason)
        case = await self.cases.create_case(
            guild=interaction.guild, action="untimeout", user=member,
            moderator=interaction.user, reason=reason,
        )
        await interaction.response.send_message(
            embed=success_embed(f"**{member}** timeout removed. (Case #{case.case_id})")
        )

    @app_commands.command(name="purge", description="Bulk delete messages")
    @app_commands.describe(count="Number of messages to delete (1-500)", member="Only delete messages from this member")
    @app_commands.default_permissions(manage_messages=True)
    async def purge(
        self,
        interaction: discord.Interaction,
        count: app_commands.Range[int, 1, 500],
        member: discord.Member | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)

        def check(msg: discord.Message) -> bool:
            if member:
                return msg.author.id == member.id
            return True

        deleted = await interaction.channel.purge(limit=count, check=check)
        await interaction.followup.send(
            embed=success_embed(f"Deleted {len(deleted)} message(s)."), ephemeral=True
        )

    @app_commands.command(name="case", description="Look up a specific case")
    @app_commands.describe(case_id="Case number to look up")
    @app_commands.default_permissions(moderate_members=True)
    async def case_lookup(
        self,
        interaction: discord.Interaction,
        case_id: int,
    ) -> None:
        case = await self.cases.get_case(interaction.guild.id, case_id)
        if not case:
            return await interaction.response.send_message(
                embed=error_embed(f"Case #{case_id} not found."), ephemeral=True
            )
        await interaction.response.send_message(embed=case_embed(case))

    @app_commands.command(name="cases", description="List cases for a user or the whole server")
    @app_commands.describe(member="Filter by member")
    @app_commands.default_permissions(moderate_members=True)
    async def cases_list(
        self,
        interaction: discord.Interaction,
        member: discord.User | None = None,
    ) -> None:
        if member:
            cases = await self.cases.get_user_cases(interaction.guild.id, member.id)
        else:
            cases = await self.cases.get_recent_cases(interaction.guild.id)

        if not cases:
            return await interaction.response.send_message(
                embed=error_embed("No cases found."), ephemeral=True
            )

        pages: list[discord.Embed] = []
        for i in range(0, len(cases), 5):
            chunk = cases[i : i + 5]
            desc = "\n".join(
                f"**#{c.case_id}** | {c.action} | <@{c.user_id}> | {c.reason or 'No reason'}"
                + (" | *pardoned*" if c.pardoned else "")
                for c in chunk
            )
            title = f"Cases for {member}" if member else "Recent Cases"
            embed = discord.Embed(
                title=title,
                description=desc,
                color=discord.Color.blurple(),
            )
            embed.set_footer(text=f"Page {i // 5 + 1}/{(len(cases) - 1) // 5 + 1}")
            pages.append(embed)

        if len(pages) == 1:
            return await interaction.response.send_message(embed=pages[0])

        view = PaginatorView(pages, author_id=interaction.user.id)
        await interaction.response.send_message(embed=pages[0], view=view)

    @app_commands.command(name="reason", description="Update the reason for a case")
    @app_commands.describe(case_id="Case number", reason="New reason")
    @app_commands.default_permissions(moderate_members=True)
    async def reason(
        self,
        interaction: discord.Interaction,
        case_id: int,
        reason: str,
    ) -> None:
        case = await self.cases.update_reason(interaction.guild.id, case_id, reason)
        if not case:
            return await interaction.response.send_message(
                embed=error_embed(f"Case #{case_id} not found."), ephemeral=True
            )
        await interaction.response.send_message(
            embed=success_embed(f"Case #{case_id} reason updated.")
        )

    @app_commands.command(name="pardon", description="Pardon (mark as resolved) a case")
    @app_commands.describe(case_id="Case number to pardon")
    @app_commands.default_permissions(moderate_members=True)
    async def pardon(
        self,
        interaction: discord.Interaction,
        case_id: int,
    ) -> None:
        case = await self.cases.pardon_case(interaction.guild.id, case_id)
        if not case:
            return await interaction.response.send_message(
                embed=error_embed(f"Case #{case_id} not found."), ephemeral=True
            )
        await interaction.response.send_message(
            embed=success_embed(f"Case #{case_id} has been pardoned.")
        )


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Moderation(bot))

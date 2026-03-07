from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import timedelta

import discord
from discord import app_commands
from discord.ext import commands

from bot.bot import ModBot
from bot.models.filter_rules import FilterRule
from bot.services.config_service import ConfigService
from bot.services.filter_service import FilterService
from bot.services.case_service import CaseService
from bot.utils.embeds import success_embed, error_embed

log = logging.getLogger(__name__)


class Filters(commands.Cog):

    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.filter_svc = FilterService(bot.db)
        self.config_svc = ConfigService(bot.db)
        self.case_svc = CaseService(bot)

        self._spam_windows: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._join_windows: dict[int, list[float]] = defaultdict(list)

    filter_group = app_commands.Group(
        name="filter", description="Manage content filters",
        default_permissions=discord.Permissions(manage_guild=True),
    )

    @filter_group.command(name="add", description="Add a filter rule")
    @app_commands.describe(
        rule_type="Type of filter",
        pattern="Pattern to match",
        action="Action to take when matched",
    )
    @app_commands.choices(
        rule_type=[
            app_commands.Choice(name="Word", value="word"),
            app_commands.Choice(name="Regex", value="regex"),
            app_commands.Choice(name="Link", value="link"),
        ],
        action=[
            app_commands.Choice(name="Delete", value="delete"),
            app_commands.Choice(name="Warn", value="warn"),
            app_commands.Choice(name="Timeout (5 min)", value="timeout"),
        ],
    )
    async def filter_add(
        self,
        interaction: discord.Interaction,
        rule_type: str,
        pattern: str,
        action: str = "delete",
    ) -> None:
        rule = FilterRule(
            guild_id=interaction.guild.id,
            rule_type=rule_type,
            pattern=pattern,
            action=action,
            created_by=interaction.user.id,
        )
        await self.filter_svc.add_rule(rule)
        await interaction.response.send_message(
            embed=success_embed(f"Filter added: `{pattern}` ({rule_type}) -> {action}")
        )

    @filter_group.command(name="remove", description="Remove a filter rule")
    @app_commands.describe(pattern="Exact pattern to remove")
    async def filter_remove(self, interaction: discord.Interaction, pattern: str) -> None:
        removed = await self.filter_svc.remove_rule(interaction.guild.id, pattern)
        if removed:
            await interaction.response.send_message(embed=success_embed(f"Filter `{pattern}` removed."))
        else:
            await interaction.response.send_message(
                embed=error_embed(f"No filter found with pattern `{pattern}`."), ephemeral=True
            )

    @filter_group.command(name="list", description="List all filter rules")
    async def filter_list(self, interaction: discord.Interaction) -> None:
        rules = await self.filter_svc.get_rules(interaction.guild.id)
        if not rules:
            return await interaction.response.send_message(
                embed=error_embed("No filters configured."), ephemeral=True
            )

        desc = "\n".join(
            f"**{r.rule_type}** | `{r.pattern}` -> {r.action}" for r in rules
        )
        embed = discord.Embed(title="Filter Rules", description=desc, color=discord.Color.blurple())
        await interaction.response.send_message(embed=embed)

    antispam_group = app_commands.Group(
        name="antispam", description="Configure anti-spam",
        default_permissions=discord.Permissions(administrator=True),
    )

    @antispam_group.command(name="config", description="Configure anti-spam settings")
    @app_commands.describe(
        enabled="Enable or disable anti-spam",
        max_messages="Max messages in the interval",
        interval="Interval in seconds",
        action="Action to take",
        duration="Timeout duration in seconds (for timeout action)",
    )
    async def antispam_config(
        self,
        interaction: discord.Interaction,
        enabled: bool,
        max_messages: int = 5,
        interval: int = 5,
        action: str = "timeout",
        duration: int = 300,
    ) -> None:
        await self.config_svc.update(
            interaction.guild.id,
            antispam_enabled=enabled,
            antispam_max_messages=max_messages,
            antispam_interval_seconds=interval,
            antispam_action=action,
            antispam_duration_seconds=duration,
        )
        status = "enabled" if enabled else "disabled"
        await interaction.response.send_message(
            embed=success_embed(
                f"Anti-spam **{status}**: {max_messages} msgs / {interval}s -> {action}"
            )
        )

    raidprotection_group = app_commands.Group(
        name="raidprotection", description="Configure raid protection",
        default_permissions=discord.Permissions(administrator=True),
    )

    @raidprotection_group.command(name="config", description="Configure raid protection")
    @app_commands.describe(
        enabled="Enable or disable raid protection",
        threshold="Number of joins to trigger raid mode",
        interval="Time window in seconds",
    )
    async def raidprotection_config(
        self,
        interaction: discord.Interaction,
        enabled: bool,
        threshold: int = 10,
        interval: int = 10,
    ) -> None:
        await self.config_svc.update(
            interaction.guild.id,
            raid_protection_enabled=enabled,
            raid_join_threshold=threshold,
            raid_join_interval_seconds=interval,
        )
        status = "enabled" if enabled else "disabled"
        await interaction.response.send_message(
            embed=success_embed(
                f"Raid protection **{status}**: {threshold} joins / {interval}s"
            )
        )

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot:
            return
        if isinstance(message.author, discord.Member) and message.author.guild_permissions.administrator:
            return

        rules = await self.filter_svc.get_rules(message.guild.id)
        matched = self.filter_svc.check_message(message.content, rules)
        if matched:
            await self._handle_filter_match(message, matched)
            return

        config = await self.config_svc.get(message.guild.id)
        if config.antispam_enabled:
            await self._check_antispam(message, config)

    async def _handle_filter_match(self, message: discord.Message, rule: FilterRule) -> None:
        try:
            await message.delete()
        except discord.Forbidden:
            pass

        if rule.action == "warn":
            await self.case_svc.create_case(
                guild=message.guild,
                action="warn",
                user=message.author,
                moderator=message.guild.me,
                reason=f"Auto-filter: matched `{rule.pattern}`",
            )
        elif rule.action == "timeout" and isinstance(message.author, discord.Member):
            try:
                await message.author.timeout(timedelta(minutes=5), reason=f"Auto-filter: {rule.pattern}")
            except discord.Forbidden:
                pass
            await self.case_svc.create_case(
                guild=message.guild,
                action="timeout",
                user=message.author,
                moderator=message.guild.me,
                reason=f"Auto-filter: matched `{rule.pattern}`",
                duration="5m",
            )

    async def _check_antispam(self, message: discord.Message, config) -> None:
        now = time.monotonic()
        window = self._spam_windows[message.guild.id][message.author.id]

        cutoff = now - config.antispam_interval_seconds
        window[:] = [t for t in window if t > cutoff]
        window.append(now)

        if len(window) < config.antispam_max_messages:
            return

        window.clear()

        member = message.author
        if not isinstance(member, discord.Member):
            return

        reason = f"Anti-spam: {config.antispam_max_messages} messages in {config.antispam_interval_seconds}s"

        if config.antispam_action == "timeout":
            try:
                await member.timeout(
                    timedelta(seconds=config.antispam_duration_seconds), reason=reason
                )
            except discord.Forbidden:
                return
            action = "timeout"
            duration = f"{config.antispam_duration_seconds}s"
        elif config.antispam_action == "kick":
            try:
                await member.kick(reason=reason)
            except discord.Forbidden:
                return
            action = "kick"
            duration = None
        elif config.antispam_action == "ban":
            try:
                await member.ban(reason=reason)
            except discord.Forbidden:
                return
            action = "ban"
            duration = None
        else:
            return

        await self.case_svc.create_case(
            guild=message.guild,
            action=action,
            user=member,
            moderator=message.guild.me,
            reason=reason,
            duration=duration,
        )

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member) -> None:
        config = await self.config_svc.get(member.guild.id)
        if not config.raid_protection_enabled:
            return

        now = time.monotonic()
        window = self._join_windows[member.guild.id]

        cutoff = now - config.raid_join_interval_seconds
        window[:] = [t for t in window if t > cutoff]
        window.append(now)

        if len(window) >= config.raid_join_threshold:
            window.clear()
            await self._trigger_raid_mode(member.guild)

    async def _trigger_raid_mode(self, guild: discord.Guild) -> None:
        log.warning("Raid detected in guild %s (%d)! Raising verification level.", guild.name, guild.id)
        try:
            await guild.edit(verification_level=discord.VerificationLevel.highest)
        except discord.Forbidden:
            log.error("Cannot change verification level in guild %d", guild.id)


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Filters(bot))

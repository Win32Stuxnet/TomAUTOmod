from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from bot.bot import ModBot
from bot.services.config_service import ConfigService
from bot.utils.embeds import success_embed, error_embed


class Config(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.config = ConfigService(bot.db)

    config_group = app_commands.Group(
        name="config", description="Server configuration", default_permissions=discord.Permissions(administrator=True)
    )

    @config_group.command(name="modlog", description="Set the mod log channel")
    @app_commands.describe(channel="Channel to use for mod log")
    async def set_modlog(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await self.config.update(interaction.guild.id, mod_log_channel_id=channel.id)
        await interaction.response.send_message(
            embed=success_embed(f"Mod log set to {channel.mention}.")
        )

    @config_group.command(name="auditlog", description="Set the audit log channel")
    @app_commands.describe(channel="Channel to use for audit log")
    async def set_auditlog(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await self.config.update(interaction.guild.id, audit_log_channel_id=channel.id)
        await interaction.response.send_message(
            embed=success_embed(f"Audit log set to {channel.mention}.")
        )

    @config_group.command(name="welcome", description="Set the welcome channel")
    @app_commands.describe(channel="Channel for welcome messages")
    async def set_welcome_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await self.config.update(interaction.guild.id, welcome_channel_id=channel.id)
        await interaction.response.send_message(
            embed=success_embed(f"Welcome channel set to {channel.mention}.")
        )

    @config_group.command(name="mlconsent", description="Enable or disable ML data collection")
    @app_commands.describe(enabled="Whether to collect message data for ML training")
    async def set_ml_consent(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self.config.update(interaction.guild.id, ml_consent=enabled)
        status = "enabled" if enabled else "disabled"
        await interaction.response.send_message(
            embed=success_embed(f"ML data collection **{status}** for this server.")
        )

    @config_group.command(name="retention", description="Set how many days message content is kept for review")
    @app_commands.describe(days="Number of days to retain message content (1-90)")
    async def set_retention(self, interaction: discord.Interaction, days: app_commands.Range[int, 1, 90]) -> None:
        await self.config.update(interaction.guild.id, log_retention_days=days)
        await interaction.response.send_message(
            embed=success_embed(f"Log retention set to **{days} days**.")
        )

    @config_group.command(name="reviewchannel", description="Set the channel for review notifications")
    @app_commands.describe(channel="Channel for ML review notifications")
    async def set_review_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await self.config.update(interaction.guild.id, review_channel_id=channel.id)
        await interaction.response.send_message(
            embed=success_embed(f"Review channel set to {channel.mention}.")
        )

    @config_group.command(name="show", description="Show current server configuration")
    async def show_config(self, interaction: discord.Interaction) -> None:
        cfg = await self.config.get(interaction.guild.id)

        def ch(cid: int | None) -> str:
            return f"<#{cid}>" if cid else "Not set"

        embed = discord.Embed(title="Server Configuration", color=discord.Color.blurple())
        embed.add_field(name="Mod Log", value=ch(cfg.mod_log_channel_id), inline=True)
        embed.add_field(name="Audit Log", value=ch(cfg.audit_log_channel_id), inline=True)
        embed.add_field(name="Welcome Channel", value=ch(cfg.welcome_channel_id), inline=True)
        embed.add_field(
            name="Auto Roles",
            value=", ".join(f"<@&{r}>" for r in cfg.auto_role_ids) or "None",
            inline=False,
        )
        embed.add_field(
            name="Anti-Spam",
            value=f"{'Enabled' if cfg.antispam_enabled else 'Disabled'} "
                  f"({cfg.antispam_max_messages} msgs / {cfg.antispam_interval_seconds}s → {cfg.antispam_action})",
            inline=False,
        )
        embed.add_field(
            name="Raid Protection",
            value=f"{'Enabled' if cfg.raid_protection_enabled else 'Disabled'} "
                  f"({cfg.raid_join_threshold} joins / {cfg.raid_join_interval_seconds}s)",
            inline=False,
        )
        embed.add_field(name="ML Consent", value="Enabled" if cfg.ml_consent else "Disabled", inline=True)
        embed.add_field(name="Log Retention", value=f"{cfg.log_retention_days} days", inline=True)
        embed.add_field(name="Review Channel", value=ch(cfg.review_channel_id), inline=True)
        await interaction.response.send_message(embed=embed)

    @config_group.command(name="reset", description="Reset all server configuration to defaults")
    async def reset_config(self, interaction: discord.Interaction) -> None:
        await self.bot.db.guild_configs.delete_one({"guild_id": interaction.guild.id})
        self.config.invalidate(interaction.guild.id)
        await interaction.response.send_message(
            embed=success_embed("Configuration reset to defaults.")
        )


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Config(bot))

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from bot.bot import ModBot
from bot.services.config_service import ConfigService
from bot.utils.embeds import success_embed, error_embed

log = logging.getLogger(__name__)


class Roles(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.config = ConfigService(bot.db)

    autorole_group = app_commands.Group(
        name="autorole", description="Manage auto-assigned roles",
        default_permissions=discord.Permissions(manage_roles=True),
    )

    @autorole_group.command(name="add", description="Add a role to auto-assign on join")
    @app_commands.describe(role="Role to auto-assign")
    async def autorole_add(self, interaction: discord.Interaction, role: discord.Role) -> None:
        cfg = await self.config.get(interaction.guild.id)
        if role.id in cfg.auto_role_ids:
            return await interaction.response.send_message(
                embed=error_embed("That role is already an auto-role."), ephemeral=True
            )

        new_roles = cfg.auto_role_ids + [role.id]
        await self.config.update(interaction.guild.id, auto_role_ids=new_roles)
        await interaction.response.send_message(
            embed=success_embed(f"{role.mention} will now be auto-assigned on join.")
        )

    @autorole_group.command(name="remove", description="Remove an auto-assigned role")
    @app_commands.describe(role="Role to remove from auto-assign")
    async def autorole_remove(self, interaction: discord.Interaction, role: discord.Role) -> None:
        cfg = await self.config.get(interaction.guild.id)
        if role.id not in cfg.auto_role_ids:
            return await interaction.response.send_message(
                embed=error_embed("That role is not an auto-role."), ephemeral=True
            )

        new_roles = [r for r in cfg.auto_role_ids if r != role.id]
        await self.config.update(interaction.guild.id, auto_role_ids=new_roles)
        await interaction.response.send_message(
            embed=success_embed(f"{role.mention} removed from auto-roles.")
        )

    @autorole_group.command(name="list", description="List all auto-assigned roles")
    async def autorole_list(self, interaction: discord.Interaction) -> None:
        cfg = await self.config.get(interaction.guild.id)
        if not cfg.auto_role_ids:
            return await interaction.response.send_message(
                embed=error_embed("No auto-roles configured."), ephemeral=True
            )

        roles_text = "\n".join(f"<@&{rid}>" for rid in cfg.auto_role_ids)
        embed = discord.Embed(title="Auto-Roles", description=roles_text, color=discord.Color.blurple())
        await interaction.response.send_message(embed=embed)

    welcome_group = app_commands.Group(
        name="welcome", description="Manage welcome messages",
        default_permissions=discord.Permissions(manage_guild=True),
    )

    @welcome_group.command(name="set", description="Set the welcome message")
    @app_commands.describe(message="Welcome message ({user} = mention, {server} = server name)")
    async def welcome_set(self, interaction: discord.Interaction, message: str) -> None:
        await self.config.update(interaction.guild.id, welcome_message=message)
        await interaction.response.send_message(
            embed=success_embed(f"Welcome message set to:\n{message}")
        )

    @welcome_group.command(name="channel", description="Set the welcome channel")
    @app_commands.describe(channel="Channel for welcome messages")
    async def welcome_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await self.config.update(interaction.guild.id, welcome_channel_id=channel.id)
        await interaction.response.send_message(
            embed=success_embed(f"Welcome channel set to {channel.mention}.")
        )

    @welcome_group.command(name="test", description="Test the welcome message")
    async def welcome_test(self, interaction: discord.Interaction) -> None:
        cfg = await self.config.get(interaction.guild.id)
        if not cfg.welcome_message or not cfg.welcome_channel_id:
            return await interaction.response.send_message(
                embed=error_embed("Welcome message or channel not configured."), ephemeral=True
            )

        msg = cfg.welcome_message.replace("{user}", interaction.user.mention)
        msg = msg.replace("{server}", interaction.guild.name)
        channel = interaction.guild.get_channel(cfg.welcome_channel_id)
        if channel and isinstance(channel, discord.TextChannel):
            await channel.send(msg)
            await interaction.response.send_message(
                embed=success_embed("Test welcome message sent!"), ephemeral=True
            )
        else:
            await interaction.response.send_message(
                embed=error_embed("Welcome channel not found."), ephemeral=True
            )

    @welcome_group.command(name="disable", description="Disable welcome messages")
    async def welcome_disable(self, interaction: discord.Interaction) -> None:
        await self.config.update(interaction.guild.id, welcome_message=None, welcome_channel_id=None)
        await interaction.response.send_message(embed=success_embed("Welcome messages disabled."))

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member) -> None:
        cfg = await self.config.get(member.guild.id)

        roles_to_add = []
        for role_id in cfg.auto_role_ids:
            role = member.guild.get_role(role_id)
            if role and role < member.guild.me.top_role:
                roles_to_add.append(role)

        if roles_to_add:
            try:
                await member.add_roles(*roles_to_add, reason="Auto-role on join")
            except discord.Forbidden:
                log.warning("Cannot assign auto-roles in guild %d", member.guild.id)

        if cfg.welcome_message and cfg.welcome_channel_id:
            channel = member.guild.get_channel(cfg.welcome_channel_id)
            if channel and isinstance(channel, discord.TextChannel):
                msg = cfg.welcome_message.replace("{user}", member.mention)
                msg = msg.replace("{server}", member.guild.name)
                try:
                    await channel.send(msg)
                except discord.Forbidden:
                    pass


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Roles(bot))

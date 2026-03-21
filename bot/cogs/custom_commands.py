from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from bot.bot import ModBot
from bot.models.custom_command import CustomCommand
from bot.services.custom_command_service import CustomCommandService
from bot.utils.embeds import success_embed, error_embed


class CustomCommands(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.service = CustomCommandService(bot.db)

    cc_group = app_commands.Group(
        name="customcmd",
        description="Manage custom commands",
        default_permissions=discord.Permissions(manage_guild=True),
    )

    @cc_group.command(name="add", description="Add a custom command")
    @app_commands.describe(
        name="Command trigger (no spaces)",
        response="The bot's response text",
        description="Short description of what this command does",
    )
    async def add_command(
        self,
        interaction: discord.Interaction,
        name: str,
        response: str,
        description: str = "",
    ) -> None:
        name = name.lower().strip()
        if " " in name:
            await interaction.response.send_message(
                embed=error_embed("Command name cannot contain spaces."),
                ephemeral=True,
            )
            return

        cmd = CustomCommand(
            guild_id=interaction.guild.id,
            name=name,
            response=response,
            description=description,
            created_by=interaction.user.id,
        )
        ok = await self.service.add(cmd)
        if not ok:
            await interaction.response.send_message(
                embed=error_embed(f"Command `{name}` already exists."),
                ephemeral=True,
            )
            return

        await interaction.response.send_message(
            embed=success_embed(f"Custom command `{name}` created.")
        )

    @cc_group.command(name="remove", description="Remove a custom command")
    @app_commands.describe(name="The command to remove")
    async def remove_command(
        self, interaction: discord.Interaction, name: str
    ) -> None:
        ok = await self.service.remove(interaction.guild.id, name.lower().strip())
        if not ok:
            await interaction.response.send_message(
                embed=error_embed(f"Command `{name}` not found."),
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            embed=success_embed(f"Command `{name}` removed.")
        )

    @cc_group.command(name="list", description="List all custom commands")
    async def list_commands(self, interaction: discord.Interaction) -> None:
        cmds = await self.service.get_all(interaction.guild.id)
        if not cmds:
            await interaction.response.send_message(
                embed=error_embed("No custom commands configured."),
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title="Custom Commands", color=discord.Color.blurple()
        )
        for cmd in sorted(cmds, key=lambda c: c.name):
            value = cmd.description or cmd.response[:80]
            embed.add_field(name=f"`{cmd.name}`", value=value, inline=False)
        await interaction.response.send_message(embed=embed)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return

        prefix = "!"
        if not message.content.startswith(prefix):
            return
        remainder = message.content[len(prefix)]
        if not remainder:
            return
        name = remainder.split(maxsplit=1)[0].lower()
        cmd = await self.service.get(message.guild.id, name)
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if cmd:
            await message.channel.send(cmd.response)


async def setup(bot: ModBot) -> None:
    await bot.add_cog(CustomCommands(bot))

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from bot.bot import ModBot
from bot.services.ticket_service import TicketService
from bot.utils.embeds import success_embed, error_embed

log = logging.getLogger(__name__)

TICKET_LIMIT_PER_USER = 3


class TicketModal(discord.ui.Modal, title="Create a Ticket"):
    topic = discord.ui.TextInput(
        label="What do you need help with?",
        style=discord.TextStyle.paragraph,
        placeholder="Describe your issue...",
        max_length=256,
    )

    def __init__(self, bot: ModBot) -> None:
        super().__init__()
        self.bot = bot
        self.ticket_svc = TicketService(bot.db)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        if not guild:
            return

        open_count = await self.ticket_svc.count_open(guild.id, interaction.user.id)
        if open_count >= TICKET_LIMIT_PER_USER:
            return await interaction.response.send_message(
                embed=error_embed(f"You already have {open_count} open ticket(s). Please close one first."),
                ephemeral=True,
            )

        overwrites = {
            guild.default_role: discord.PermissionOverwrite(read_messages=False),
            interaction.user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
            guild.me: discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_channels=True),
        }
        for role in guild.roles:
            if role.permissions.manage_messages:
                overwrites[role] = discord.PermissionOverwrite(read_messages=True, send_messages=True)

        channel = await guild.create_text_channel(
            name=f"ticket-{interaction.user.name}",
            overwrites=overwrites,
            reason=f"Ticket by {interaction.user}",
        )

        ticket = await self.ticket_svc.create_ticket(
            guild.id, channel.id, interaction.user.id, self.topic.value,
        )

        embed = discord.Embed(
            title="Ticket Created",
            description=(
                f"**Author:** {interaction.user.mention}\n"
                f"**Topic:** {self.topic.value}\n\n"
                "A moderator will be with you shortly."
            ),
            color=discord.Color.green(),
        )
        await channel.send(embed=embed)
        await interaction.response.send_message(
            embed=success_embed(f"Ticket created: {channel.mention}"), ephemeral=True
        )


class TicketButton(discord.ui.View):
    def __init__(self, bot: ModBot) -> None:
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="Create Ticket", style=discord.ButtonStyle.primary, custom_id="ticket:create", emoji="\U0001f3ab")
    async def create_ticket(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(TicketModal(self.bot))


class Tickets(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.ticket_svc = TicketService(bot.db)

    async def cog_load(self) -> None:
        self.bot.add_view(TicketButton(self.bot))

    @app_commands.command(name="ticket-setup", description="Send the ticket creation panel")
    @app_commands.default_permissions(administrator=True)
    async def ticket_setup(self, interaction: discord.Interaction) -> None:
        embed = discord.Embed(
            title="Support Tickets",
            description="Click the button below to create a support ticket.",
            color=discord.Color.blurple(),
        )
        await interaction.channel.send(embed=embed, view=TicketButton(self.bot))
        await interaction.response.send_message(
            embed=success_embed("Ticket panel sent!"), ephemeral=True
        )

    @app_commands.command(name="ticket-close", description="Close the current ticket")
    @app_commands.default_permissions(manage_messages=True)
    async def ticket_close(self, interaction: discord.Interaction) -> None:
        closed = await self.ticket_svc.close_ticket(interaction.channel.id)
        if not closed:
            return await interaction.response.send_message(
                embed=error_embed("This is not an open ticket channel."), ephemeral=True
            )
        await interaction.response.send_message(embed=success_embed("Ticket closed. Channel will be deleted in 10 seconds."))
        await asyncio.sleep(10)
        try:
            await interaction.channel.delete(reason="Ticket closed")
        except discord.Forbidden:
            pass

    @app_commands.command(name="ticket-claim", description="Claim this ticket")
    @app_commands.default_permissions(manage_messages=True)
    async def ticket_claim(self, interaction: discord.Interaction) -> None:
        claimed = await self.ticket_svc.claim_ticket(interaction.channel.id, interaction.user.id)
        if not claimed:
            return await interaction.response.send_message(
                embed=error_embed("This ticket cannot be claimed (not open or not a ticket)."), ephemeral=True
            )
        await interaction.response.send_message(
            embed=success_embed(f"Ticket claimed by {interaction.user.mention}.")
        )

    @app_commands.command(name="ticket-add", description="Add a user to this ticket")
    @app_commands.describe(member="Member to add")
    @app_commands.default_permissions(manage_messages=True)
    async def ticket_add(self, interaction: discord.Interaction, member: discord.Member) -> None:
        await self.ticket_svc.add_user(interaction.channel.id, member.id)
        await interaction.channel.set_permissions(member, read_messages=True, send_messages=True)
        await interaction.response.send_message(
            embed=success_embed(f"{member.mention} added to the ticket.")
        )

    @app_commands.command(name="ticket-remove", description="Remove a user from this ticket")
    @app_commands.describe(member="Member to remove")
    @app_commands.default_permissions(manage_messages=True)
    async def ticket_remove(self, interaction: discord.Interaction, member: discord.Member) -> None:
        await self.ticket_svc.remove_user(interaction.channel.id, member.id)
        await interaction.channel.set_permissions(member, overwrite=None)
        await interaction.response.send_message(
            embed=success_embed(f"{member.mention} removed from the ticket.")
        )


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Tickets(bot))

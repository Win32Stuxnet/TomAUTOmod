from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord

from bot.utils.permissions import is_mod

if TYPE_CHECKING:
    from bot.bot import ModBot


class QuickLabelView(discord.ui.View):
    """Reusable label buttons that any mod can click."""

    def __init__(self, bot: ModBot, guild_id: int, message_id: int, *, timeout: float = 3600) -> None:
        super().__init__(timeout=timeout)
        self.bot = bot
        self.query = {"guild_id": guild_id, "message_id": message_id}

    async def _label(self, interaction: discord.Interaction, label: str) -> None:
        if not is_mod(interaction.user):
            return await interaction.response.send_message("You need Moderate Members permission.", ephemeral=True)

        await self.bot.db.ml_training_data.update_one(
            self.query,
            {"$set": {
                "label": label,
                "reviewed_by": interaction.user.id,
                "reviewed_at": datetime.now(timezone.utc),
            }},
        )
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        await interaction.response.edit_message(
            content=f"Labeled as **{label}** by {interaction.user.mention}",
            view=self,
        )

    @discord.ui.button(label="Safe", style=discord.ButtonStyle.success)
    async def safe_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "safe")

    @discord.ui.button(label="Flag", style=discord.ButtonStyle.secondary)
    async def flag_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "flagged")

    @discord.ui.button(label="Toxic", style=discord.ButtonStyle.danger)
    async def toxic_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "toxic")

    async def on_timeout(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

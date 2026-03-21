from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord

from bot.utils.permissions import is_mod

if TYPE_CHECKING:
    from bot.bot import ModBot


class AggressionLabelView(discord.ui.View):
    """Reusable aggression label buttons for moderator review."""

    def __init__(self, bot: ModBot, guild_id: int, message_id: int, *, timeout: float = 3600) -> None:
        super().__init__(timeout=timeout)
        self.bot = bot
        self.query = {"guild_id": guild_id, "message_id": message_id}

    async def _label(self, interaction: discord.Interaction, label: str) -> None:
        if not is_mod(interaction.user):
            return await interaction.response.send_message("You need Moderate Members permission.", ephemeral=True)

        await self.bot.db.aggression_training_data.update_one(
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

    @discord.ui.button(label="Aggressive", style=discord.ButtonStyle.danger)
    async def aggressive_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "aggressive")

    @discord.ui.button(label="Not Aggressive", style=discord.ButtonStyle.success)
    async def not_aggressive_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "not_aggressive")

    async def on_timeout(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

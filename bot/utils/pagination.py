from __future__ import annotations

from typing import Sequence

import discord


class PaginatorView(discord.ui.View):
    def __init__(self, embeds: Sequence[discord.Embed], *, author_id: int, timeout: float = 120) -> None:
        super().__init__(timeout=timeout)
        self.embeds = embeds
        self.author_id = author_id
        self.page = 0
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.prev_btn.disabled = self.page == 0
        self.next_btn.disabled = self.page >= len(self.embeds) - 1

    @discord.ui.button(label="Previous", style=discord.ButtonStyle.secondary)
    async def prev_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if interaction.user.id != self.author_id:
            return await interaction.response.send_message("Not your paginator.", ephemeral=True)
        self.page = max(0, self.page - 1)
        self._update_buttons()
        await interaction.response.edit_message(embed=self.embeds[self.page], view=self)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.secondary)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if interaction.user.id != self.author_id:
            return await interaction.response.send_message("Not your paginator.", ephemeral=True)
        self.page = min(len(self.embeds) - 1, self.page + 1)
        self._update_buttons()
        await interaction.response.edit_message(embed=self.embeds[self.page], view=self)

    async def on_timeout(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

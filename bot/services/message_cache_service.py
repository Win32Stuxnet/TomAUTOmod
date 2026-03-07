from __future__ import annotations

from typing import TYPE_CHECKING

import discord

from bot.models.message_cache import CachedMessage

if TYPE_CHECKING:
    from bot.database import Database


class MessageCacheService:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def cache_message(self, message: discord.Message) -> None:
        doc = CachedMessage(
            message_id=message.id,
            channel_id=message.channel.id,
            guild_id=message.guild.id,
            author_id=message.author.id,
            content=message.content,
            attachments=[a.url for a in message.attachments],
        )
        await self.db.message_cache.update_one(
            {"message_id": message.id},
            {"$set": doc.to_doc()},
            upsert=True,
        )

    async def get_cached(self, message_id: int) -> CachedMessage | None:
        doc = await self.db.message_cache.find_one({"message_id": message_id})
        return CachedMessage.from_doc(doc) if doc else None

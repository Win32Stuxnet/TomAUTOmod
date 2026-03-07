from __future__ import annotations

from typing import TYPE_CHECKING

from bot.models.guild_config import GuildConfig

if TYPE_CHECKING:
    from bot.database import Database


class ConfigService:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._cache: dict[int, GuildConfig] = {}

    async def get(self, guild_id: int) -> GuildConfig:
        if guild_id in self._cache:
            return self._cache[guild_id]

        doc = await self.db.guild_configs.find_one({"guild_id": guild_id})
        config = GuildConfig.from_doc(doc) if doc else GuildConfig(guild_id=guild_id)
        self._cache[guild_id] = config
        return config

    async def update(self, guild_id: int, **fields) -> GuildConfig:
        await self.db.guild_configs.update_one(
            {"guild_id": guild_id},
            {"$set": fields},
            upsert=True,
        )
        self._cache.pop(guild_id, None)
        return await self.get(guild_id)

    def invalidate(self, guild_id: int) -> None:
        self._cache.pop(guild_id, None)

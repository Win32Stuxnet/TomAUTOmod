from __future__ import annotations

from typing import TYPE_CHECKING

from bot.models.custom_command import CustomCommand

if TYPE_CHECKING:
    from bot.database import Database


class CustomCommandService:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._cache: dict[int, dict[str, CustomCommand]] = {}

    async def get_all(self, guild_id: int) -> list[CustomCommand]:
        if guild_id not in self._cache:
            cursor = self.db.custom_commands.find({"guild_id": guild_id})
            cmds = [CustomCommand.from_doc(doc) async for doc in cursor]
            self._cache[guild_id] = {c.name: c for c in cmds}
        return list(self._cache[guild_id].values())

    async def get(self, guild_id: int, name: str) -> CustomCommand | None:
        await self.get_all(guild_id)
        return self._cache.get(guild_id, {}).get(name)

    async def add(self, cmd: CustomCommand) -> bool:
        existing = await self.get(cmd.guild_id, cmd.name)
        if existing:
            return False
        await self.db.custom_commands.insert_one(cmd.to_doc())
        self._cache.pop(cmd.guild_id, None)
        return True

    async def update(self, guild_id: int, name: str, **fields) -> bool:
        result = await self.db.custom_commands.update_one(
            {"guild_id": guild_id, "name": name},
            {"$set": fields},
        )
        self._cache.pop(guild_id, None)
        return result.modified_count > 0

    async def remove(self, guild_id: int, name: str) -> bool:
        result = await self.db.custom_commands.delete_one(
            {"guild_id": guild_id, "name": name}
        )
        self._cache.pop(guild_id, None)
        return result.deleted_count > 0

    def invalidate(self, guild_id: int) -> None:
        self._cache.pop(guild_id, None)

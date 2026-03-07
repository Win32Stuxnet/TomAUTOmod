from __future__ import annotations

import logging

from pymongo import AsyncMongoClient, ASCENDING, DESCENDING, IndexModel

log = logging.getLogger(__name__)


class Database:
    def __init__(self, uri: str, db_name: str) -> None:
        self._client: AsyncMongoClient = AsyncMongoClient(uri)
        self._db = self._client[db_name]

    @property
    def guild_configs(self):
        return self._db["guild_configs"]

    @property
    def cases(self):
        return self._db["cases"]

    @property
    def filter_rules(self):
        return self._db["filter_rules"]

    @property
    def tickets(self):
        return self._db["tickets"]

    @property
    def message_cache(self):
        return self._db["message_cache"]

    @property
    def ml_training_data(self):
        return self._db["ml_training_data"]

    async def create_indexes(self) -> None:
        await self.guild_configs.create_indexes([
            IndexModel([("guild_id", ASCENDING)], unique=True),
        ])

        await self.cases.create_indexes([
            IndexModel([("guild_id", ASCENDING), ("case_id", DESCENDING)]),
            IndexModel([("guild_id", ASCENDING), ("user_id", ASCENDING)]),
        ])

        await self.filter_rules.create_indexes([
            IndexModel([("guild_id", ASCENDING)]),
        ])

        await self.tickets.create_indexes([
            IndexModel([("guild_id", ASCENDING), ("channel_id", ASCENDING)]),
            IndexModel([("guild_id", ASCENDING), ("status", ASCENDING)]),
        ])

        await self.message_cache.create_indexes([
            IndexModel([("message_id", ASCENDING)], unique=True),
            IndexModel([("created_at", ASCENDING)], expireAfterSeconds=7 * 24 * 3600),
        ])

        await self.ml_training_data.create_indexes([
            IndexModel([("guild_id", ASCENDING), ("message_id", ASCENDING)]),
            IndexModel([("label", ASCENDING)]),
        ])

        log.info("All database indexes ensured.")

    async def next_case_id(self, guild_id: int) -> int:
        doc = await self.cases.find_one(
            {"guild_id": guild_id},
            sort=[("case_id", DESCENDING)],
            projection={"case_id": 1},
        )
        return (doc["case_id"] + 1) if doc else 1

    def close(self) -> None:
        self._client.close()

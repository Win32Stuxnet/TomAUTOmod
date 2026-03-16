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
        """
        Access the MongoDB collection used to cache messages; cached documents are configured to expire after 7 days.
        
        Returns:
            Collection: The `message_cache` collection for storing cached message documents.
        """
        return self._db["message_cache"]

    @property
    def custom_commands(self):
        """
        Access the database collection that stores custom command documents.
        
        Returns:
            The collection used to store custom command documents (documents keyed by guild and command name).
        """
        return self._db["custom_commands"]

    @property
    def ml_training_data(self):
        """
        Access the MongoDB collection used to store machine learning training data.
        
        Returns:
            Collection: The `ml_training_data` collection containing ML training examples and related metadata.
        """
        return self._db["ml_training_data"]

    async def create_indexes(self) -> None:
        """
        Ensure required indexes exist on the database collections used by the application.
        
        Creates the following indexes:
        - guild_configs: unique index on `guild_id`.
        - cases: compound indexes on (`guild_id`, `case_id` descending) and (`guild_id`, `user_id`).
        - filter_rules: index on `guild_id`.
        - tickets: compound indexes on (`guild_id`, `channel_id`) and (`guild_id`, `status`).
        - message_cache: unique index on `message_id` and TTL index on `created_at` that expires documents after 7 days.
        - custom_commands: unique compound index on (`guild_id`, `name`).
        - web_sessions: TTL index on `updated_at` that expires documents after 86400 seconds (24 hours).
        - ml_training_data: indexes on (`guild_id`, `message_id`), `label`, (`guild_id`, `label`, `confidence` descending), and (`guild_id`, `label`, `created_at` descending).
        
        Logs an informational message when all indexes have been ensured.
        """
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

        await self.custom_commands.create_indexes([
            IndexModel([("guild_id", ASCENDING), ("name", ASCENDING)], unique=True),
        ])

        await self._db["web_sessions"].create_indexes([
            IndexModel([("updated_at", ASCENDING)], expireAfterSeconds=86400),
        ])

        await self.ml_training_data.create_indexes([
            IndexModel([("guild_id", ASCENDING), ("message_id", ASCENDING)]),
            IndexModel([("label", ASCENDING)]),
            IndexModel([("guild_id", ASCENDING), ("label", ASCENDING), ("confidence", DESCENDING)]),
            IndexModel([("guild_id", ASCENDING), ("label", ASCENDING), ("created_at", DESCENDING)]),
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

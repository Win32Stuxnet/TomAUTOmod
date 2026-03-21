from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from bot.models.aggression_data import AggressionTrainingSample

if TYPE_CHECKING:
    from bot.database import Database


class StrikeTracker:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def record_strike(
        self,
        *,
        guild_id: int,
        user_id: int,
        message_id: int,
        channel_id: int,
        score: float,
        content: str,
        embedding: list[float],
        created_at: datetime | None = None,
    ) -> None:
        created_at = created_at or datetime.now(timezone.utc)
        strike_doc = {
            "guild_id": guild_id,
            "user_id": user_id,
            "message_id": message_id,
            "channel_id": channel_id,
            "score": score,
            "content": content[:300],
            "alert_sent": False,
            "alert_sent_at": None,
            "created_at": created_at,
        }
        await self.db.aggression_strikes.update_one(
            {"message_id": message_id},
            {"$setOnInsert": strike_doc},
            upsert=True,
        )

        sample = AggressionTrainingSample(
            guild_id=guild_id,
            message_id=message_id,
            channel_id=channel_id,
            user_id=user_id,
            content=content[:300],
            embedding=embedding,
            score=score,
            created_at=created_at,
        )
        await self.db.aggression_training_data.update_one(
            {"message_id": message_id},
            {"$setOnInsert": sample.to_doc()},
            upsert=True,
        )

    async def count_recent_strikes(self, guild_id: int, user_id: int, window_hours: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        return await self.db.aggression_strikes.count_documents({
            "guild_id": guild_id,
            "user_id": user_id,
            "created_at": {"$gte": cutoff},
        })

    async def check_threshold(self, guild_id: int, user_id: int, strike_count: int, window_hours: int) -> bool:
        count = await self.count_recent_strikes(guild_id, user_id, window_hours)
        return count >= strike_count

    async def has_alert_in_window(self, guild_id: int, user_id: int, window_hours: int) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        doc = await self.db.aggression_strikes.find_one(
            {
                "guild_id": guild_id,
                "user_id": user_id,
                "created_at": {"$gte": cutoff},
                "alert_sent": True,
            },
            projection={"_id": 1},
        )
        return doc is not None

    async def mark_alerted(self, guild_id: int, user_id: int, window_hours: int) -> None:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=window_hours)
        await self.db.aggression_strikes.update_many(
            {
                "guild_id": guild_id,
                "user_id": user_id,
                "created_at": {"$gte": cutoff},
            },
            {"$set": {"alert_sent": True, "alert_sent_at": now}},
        )

    async def get_recent_strikes(self, guild_id: int, user_id: int, window_hours: int, *, limit: int = 5) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        cursor = self.db.aggression_strikes.find(
            {
                "guild_id": guild_id,
                "user_id": user_id,
                "created_at": {"$gte": cutoff},
            },
        ).sort("created_at", -1).limit(limit)
        return [doc async for doc in cursor]

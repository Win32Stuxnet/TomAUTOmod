from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord

from bot.models.tickets import Ticket

if TYPE_CHECKING:
    from bot.database import Database


class TicketService:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def create_ticket(
        self, guild_id: int, channel_id: int, author_id: int, topic: str
    ) -> Ticket:
        ticket = Ticket(
            guild_id=guild_id,
            channel_id=channel_id,
            author_id=author_id,
            topic=topic,
        )
        await self.db.tickets.insert_one(ticket.to_doc())
        return ticket

    async def get_by_channel(self, channel_id: int) -> Ticket | None:
        doc = await self.db.tickets.find_one({"channel_id": channel_id})
        return Ticket.from_doc(doc) if doc else None

    async def close_ticket(self, channel_id: int) -> bool:
        result = await self.db.tickets.update_one(
            {"channel_id": channel_id, "status": {"$ne": "closed"}},
            {"$set": {"status": "closed", "closed_at": datetime.now(timezone.utc)}},
        )
        return result.modified_count > 0

    async def claim_ticket(self, channel_id: int, claimer_id: int) -> bool:
        result = await self.db.tickets.update_one(
            {"channel_id": channel_id, "status": "open"},
            {"$set": {"status": "claimed", "claimed_by": claimer_id}},
        )
        return result.modified_count > 0

    async def add_user(self, channel_id: int, user_id: int) -> bool:
        result = await self.db.tickets.update_one(
            {"channel_id": channel_id},
            {"$addToSet": {"added_users": user_id}},
        )
        return result.modified_count > 0

    async def remove_user(self, channel_id: int, user_id: int) -> bool:
        result = await self.db.tickets.update_one(
            {"channel_id": channel_id},
            {"$pull": {"added_users": user_id}},
        )
        return result.modified_count > 0

    async def count_open(self, guild_id: int, author_id: int) -> int:
        return await self.db.tickets.count_documents(
            {"guild_id": guild_id, "author_id": author_id, "status": {"$ne": "closed"}}
        )

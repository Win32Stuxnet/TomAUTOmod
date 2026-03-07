from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord

from bot.models.cases import Case
from bot.utils.embeds import mod_action_embed

if TYPE_CHECKING:
    from bot.bot import ModBot

log = logging.getLogger(__name__)


class CaseService:
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.db = bot.db

    async def create_case(
        self,
        *,
        guild: discord.Guild,
        action: str,
        user: discord.User | discord.Member,
        moderator: discord.User | discord.Member,
        reason: str | None = None,
        duration: str | None = None,
    ) -> Case:
        case_id = await self.db.next_case_id(guild.id)
        case = Case(
            guild_id=guild.id,
            case_id=case_id,
            action=action,
            user_id=user.id,
            moderator_id=moderator.id,
            reason=reason,
            duration=duration,
        )
        await self.db.cases.insert_one(case.to_doc())

        from bot.services.config_service import ConfigService
        config_svc = ConfigService(self.db)
        config = await config_svc.get(guild.id)
        if config.mod_log_channel_id:
            channel = guild.get_channel(config.mod_log_channel_id)
            if channel and isinstance(channel, discord.TextChannel):
                embed = mod_action_embed(
                    action=action,
                    user=user,
                    moderator=moderator,
                    reason=reason,
                    case_id=case_id,
                )
                await channel.send(embed=embed)

        await self._label_recent_messages(guild.id, user.id, action)

        log.info("Case #%d created in guild %d: %s on %s", case_id, guild.id, action, user)
        return case

    async def get_case(self, guild_id: int, case_id: int) -> Case | None:
        doc = await self.db.cases.find_one({"guild_id": guild_id, "case_id": case_id})
        return Case.from_doc(doc) if doc else None

    async def get_user_cases(self, guild_id: int, user_id: int) -> list[Case]:
        cursor = self.db.cases.find(
            {"guild_id": guild_id, "user_id": user_id}
        ).sort("case_id", -1)
        return [Case.from_doc(doc) async for doc in cursor]

    async def get_recent_cases(self, guild_id: int, limit: int = 20) -> list[Case]:
        cursor = self.db.cases.find(
            {"guild_id": guild_id}
        ).sort("case_id", -1).limit(limit)
        return [Case.from_doc(doc) async for doc in cursor]

    async def update_reason(self, guild_id: int, case_id: int, reason: str) -> Case | None:
        result = await self.db.cases.find_one_and_update(
            {"guild_id": guild_id, "case_id": case_id},
            {"$set": {"reason": reason}},
            return_document=True,
        )
        return Case.from_doc(result) if result else None

    async def pardon_case(self, guild_id: int, case_id: int) -> Case | None:
        result = await self.db.cases.find_one_and_update(
            {"guild_id": guild_id, "case_id": case_id},
            {"$set": {"pardoned": True}},
            return_document=True,
        )
        return Case.from_doc(result) if result else None

    async def _label_recent_messages(self, guild_id: int, user_id: int, action: str) -> None:
        label_map = {
            "warn": "flagged",
            "kick": "toxic",
            "ban": "toxic",
            "timeout": "flagged",
        }
        label = label_map.get(action)
        if not label:
            return

        result = await self.db.ml_training_data.update_many(
            {"guild_id": guild_id, "user_id": user_id, "label": None},
            {"$set": {"label": label}},
        )
        if result.modified_count:
            log.info(
                "Auto-labeled %d messages as '%s' for user %d in guild %d",
                result.modified_count, label, user_id, guild_id,
            )

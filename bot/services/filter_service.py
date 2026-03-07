from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bot.models.filter_rules import FilterRule

if TYPE_CHECKING:
    from bot.database import Database


class FilterService:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._cache: dict[int, list[FilterRule]] = {}

    async def get_rules(self, guild_id: int) -> list[FilterRule]:
        if guild_id in self._cache:
            return self._cache[guild_id]

        cursor = self.db.filter_rules.find({"guild_id": guild_id})
        rules = [FilterRule.from_doc(doc) async for doc in cursor]
        self._cache[guild_id] = rules
        return rules

    async def add_rule(self, rule: FilterRule) -> None:
        await self.db.filter_rules.insert_one(rule.to_doc())
        self._cache.pop(rule.guild_id, None)

    async def remove_rule(self, guild_id: int, pattern: str) -> bool:
        result = await self.db.filter_rules.delete_one(
            {"guild_id": guild_id, "pattern": pattern}
        )
        self._cache.pop(guild_id, None)
        return result.deleted_count > 0

    def invalidate(self, guild_id: int) -> None:
        self._cache.pop(guild_id, None)

    def check_message(self, content: str, rules: list[FilterRule]) -> FilterRule | None:
        lower = content.lower()
        for rule in rules:
            if rule.rule_type == "word":
                if rule.pattern.lower() in lower:
                    return rule
            elif rule.rule_type == "regex":
                try:
                    if re.search(rule.pattern, content, re.IGNORECASE):
                        return rule
                except re.error:
                    continue
            elif rule.rule_type == "link":
                if rule.pattern.lower() in lower:
                    return rule
        return None

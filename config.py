from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    discord_token: str = field(repr=False, default_factory=lambda: os.environ["DISCORD_TOKEN"])
    mongodb_uri: str = field(default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    mongodb_db_name: str = field(default_factory=lambda: os.getenv("MONGODB_DB_NAME", "discord_mod_bot"))
    owner_ids: frozenset[int] = field(
        default_factory=lambda: frozenset(
            int(i) for i in os.getenv("OWNER_IDS", "").split(",") if i.strip()
        )
    )


settings = Settings()

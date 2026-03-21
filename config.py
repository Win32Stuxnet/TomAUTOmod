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
    # Web dashboard
    discord_client_id: str = field(default_factory=lambda: os.getenv("DISCORD_CLIENT_ID", ""))
    discord_client_secret: str = field(repr=False, default_factory=lambda: os.getenv("DISCORD_CLIENT_SECRET", ""))
    web_secret_key: str = field(repr=False, default_factory=lambda: os.getenv("WEB_SECRET_KEY", "change-me-in-production"))
    web_port: int = field(default_factory=lambda: int(os.getenv("WEB_PORT", "3000")))
    web_base_url: str = field(default_factory=lambda: os.getenv("WEB_BASE_URL", "http://localhost:3000"))


settings = Settings()

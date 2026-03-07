from __future__ import annotations

import discord


class Colors:
    SUCCESS = discord.Color.green()
    WARNING = discord.Color.gold()
    ERROR = discord.Color.red()
    INFO = discord.Color.blurple()
    MOD_ACTION = discord.Color.orange()
    AUDIT = discord.Color.dark_grey()


class Limits:
    MAX_WARNINGS_BEFORE_BAN = 5
    PURGE_MAX = 500
    EMBED_DESCRIPTION_MAX = 4096
    CASE_REASON_MAX = 512
    TICKET_TOPIC_MAX = 256


class Emotes:
    CHECK = "\u2705"
    CROSS = "\u274c"
    WARN = "\u26a0\ufe0f"
    BAN = "\U0001f528"
    KICK = "\U0001f462"
    TIMEOUT = "\u23f1\ufe0f"
    TICKET = "\U0001f3ab"

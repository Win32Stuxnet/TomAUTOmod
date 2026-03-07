from __future__ import annotations

import re
from datetime import timedelta

_PATTERN = re.compile(
    r"(?:(\d+)\s*w(?:eeks?)?)?\s*"
    r"(?:(\d+)\s*d(?:ays?)?)?\s*"
    r"(?:(\d+)\s*h(?:ours?|rs?)?)?\s*"
    r"(?:(\d+)\s*m(?:in(?:ute)?s?)?)?\s*"
    r"(?:(\d+)\s*s(?:ec(?:ond)?s?)?)?",
    re.IGNORECASE,
)


def parse_duration(text: str) -> timedelta | None:
    match = _PATTERN.fullmatch(text.strip())
    if not match or not any(match.groups()):
        return None

    weeks, days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    delta = timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)

    if delta.total_seconds() == 0:
        return None
    return delta


def format_duration(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    parts: list[str] = []

    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")

    return " ".join(parts) or "0s"

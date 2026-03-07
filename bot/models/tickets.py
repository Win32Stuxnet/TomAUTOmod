from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Ticket:
    guild_id: int
    channel_id: int
    author_id: int
    topic: str
    status: str = "open"
    claimed_by: int | None = None
    added_users: list[int] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None

    def to_doc(self) -> dict:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict) -> Ticket:
        doc.pop("_id", None)
        return cls(**{k: v for k, v in doc.items() if k in cls.__dataclass_fields__})

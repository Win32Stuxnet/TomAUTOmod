from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class CachedMessage:
    message_id: int
    channel_id: int
    guild_id: int
    author_id: int
    content: str
    attachments: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_doc(self) -> dict:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict) -> CachedMessage:
        doc.pop("_id", None)
        return cls(**{k: v for k, v in doc.items() if k in cls.__dataclass_fields__})

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Case:
    guild_id: int
    case_id: int
    action: str
    user_id: int
    moderator_id: int
    reason: str | None = None
    duration: str | None = None
    pardoned: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_doc(self) -> dict:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict) -> Case:
        doc.pop("_id", None)
        return cls(**doc)

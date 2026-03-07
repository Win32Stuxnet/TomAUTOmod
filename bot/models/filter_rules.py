from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class FilterRule:
    guild_id: int
    rule_type: str
    pattern: str
    action: str = "delete"
    created_by: int = 0
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_doc(self) -> dict:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict) -> FilterRule:
        doc.pop("_id", None)
        return cls(**{k: v for k, v in doc.items() if k in cls.__dataclass_fields__})

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Sample:
    text: str
    label: str | None  # "toxic", "flagged", "safe", or None
    source: str  # "reddit", "mongodb", "manual"
    metadata: dict = field(default_factory=dict)
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Sample:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class MLTrainingSample:
    guild_id: int
    message_id: int
    user_id: int
    channel_id: int = 0
    content: str = ""
    features: dict = field(default_factory=dict)
    label: str | None = None
    prediction: str | None = None
    confidence: float = 0.0
    reviewed_by: int | None = None
    reviewed_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_doc(self) -> dict:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict) -> MLTrainingSample:
        doc.pop("_id", None)
        return cls(**{k: v for k, v in doc.items() if k in cls.__dataclass_fields__})

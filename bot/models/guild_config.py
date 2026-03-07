from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class GuildConfig:
    guild_id: int
    mod_log_channel_id: int | None = None
    audit_log_channel_id: int | None = None
    welcome_channel_id: int | None = None
    welcome_message: str | None = None
    auto_role_ids: list[int] = field(default_factory=list)
    antispam_enabled: bool = False
    antispam_max_messages: int = 5
    antispam_interval_seconds: int = 5
    antispam_action: str = "timeout"
    antispam_duration_seconds: int = 300
    raid_protection_enabled: bool = False
    raid_join_threshold: int = 10
    raid_join_interval_seconds: int = 10
    ml_consent: bool = False

    def to_doc(self) -> dict:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict) -> GuildConfig:
        doc.pop("_id", None)
        return cls(**{k: v for k, v in doc.items() if k in cls.__dataclass_fields__})

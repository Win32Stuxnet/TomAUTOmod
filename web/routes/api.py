from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, field_validator, Field

from web.auth import require_guild_permission
from bot.models.guild_config import GuildConfig
from bot.models.filter_rules import FilterRule
from bot.models.custom_command import CustomCommand
from bot.services.config_service import ConfigService
from bot.services.filter_service import FilterService
from bot.services.custom_command_service import CustomCommandService

router = APIRouter(tags=["api"])


# --- Pydantic schemas ---

VALID_ANTISPAM_ACTIONS = {"mute", "kick", "ban", "warn"}
VALID_FILTER_TYPES = {"word", "regex", "wildcard"}
VALID_FILTER_ACTIONS = {"delete", "warn", "mute", "kick", "ban"}
MAX_PATTERN_LEN = 500
MAX_COMMAND_NAME_LEN = 32
MAX_RESPONSE_LEN = 2000
MAX_WELCOME_LEN = 2000


class ConfigUpdate(BaseModel):
    mod_log_channel_id: int | None = None
    audit_log_channel_id: int | None = None
    welcome_channel_id: int | None = None
    welcome_message: str | None = Field(None, max_length=MAX_WELCOME_LEN)
    antispam_enabled: bool | None = None
    antispam_max_messages: int | None = Field(None, ge=1, le=100)
    antispam_interval_seconds: int | None = Field(None, ge=1, le=300)
    antispam_action: str | None = None
    antispam_duration_seconds: int | None = Field(None, ge=0, le=2_592_000)
    raid_protection_enabled: bool | None = None
    raid_join_threshold: int | None = Field(None, ge=1, le=100)
    raid_join_interval_seconds: int | None = Field(None, ge=1, le=300)
    ml_consent: bool | None = None
    log_retention_days: int | None = Field(None, ge=1, le=365)
    review_channel_id: int | None = None
    aggression_channel_id: int | None = None
    aggression_strike_count: int | None = Field(None, ge=1, le=100)
    aggression_window_hours: int | None = Field(None, ge=1, le=720)

    @field_validator("antispam_action")
    @classmethod
    def validate_antispam_action(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_ANTISPAM_ACTIONS:
            raise ValueError(f"Must be one of {VALID_ANTISPAM_ACTIONS}")
        return v


class FilterRuleCreate(BaseModel):
    rule_type: str
    pattern: str = Field(..., min_length=1, max_length=MAX_PATTERN_LEN)
    action: str = "delete"

    @field_validator("rule_type")
    @classmethod
    def validate_rule_type(cls, v: str) -> str:
        if v not in VALID_FILTER_TYPES:
            raise ValueError(f"Must be one of {VALID_FILTER_TYPES}")
        return v

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in VALID_FILTER_ACTIONS:
            raise ValueError(f"Must be one of {VALID_FILTER_ACTIONS}")
        return v

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str, info) -> str:
        if info.data.get("rule_type") == "regex":
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex: {e}")
        return v


class CustomCommandCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=MAX_COMMAND_NAME_LEN)
    response: str = Field(..., min_length=1, max_length=MAX_RESPONSE_LEN)
    description: str = Field("", max_length=MAX_RESPONSE_LEN)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.lower().strip()
        if not re.fullmatch(r"[a-z0-9_-]+", v):
            raise ValueError("Only lowercase letters, numbers, hyphens, and underscores")
        return v


class CustomCommandUpdate(BaseModel):
    response: str | None = Field(None, min_length=1, max_length=MAX_RESPONSE_LEN)
    description: str | None = Field(None, max_length=MAX_RESPONSE_LEN)


# --- Helpers ---

def _get_services(request: Request):
    db = request.app.state.db
    return (
        ConfigService(db),
        FilterService(db),
        CustomCommandService(db),
    )


# --- Config ---

@router.get("/guilds/{guild_id}/config")
async def get_config(request: Request, guild_id: int):
    require_guild_permission(request, guild_id)
    config_svc, _, _ = _get_services(request)
    cfg = await config_svc.get(guild_id)
    return cfg.to_doc()


@router.patch("/guilds/{guild_id}/config")
async def update_config(request: Request, guild_id: int, body: ConfigUpdate):
    require_guild_permission(request, guild_id)
    config_svc, _, _ = _get_services(request)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "No fields to update")
    cfg = await config_svc.update(guild_id, **fields)
    return cfg.to_doc()


# --- Filter Rules ---

@router.get("/guilds/{guild_id}/filters")
async def get_filters(request: Request, guild_id: int):
    require_guild_permission(request, guild_id)
    _, filter_svc, _ = _get_services(request)
    rules = await filter_svc.get_rules(guild_id)
    return [r.to_doc() for r in rules]


@router.post("/guilds/{guild_id}/filters")
async def add_filter(request: Request, guild_id: int, body: FilterRuleCreate):
    user = require_guild_permission(request, guild_id)
    _, filter_svc, _ = _get_services(request)
    rule = FilterRule(
        guild_id=guild_id,
        rule_type=body.rule_type,
        pattern=body.pattern,
        action=body.action,
        created_by=int(user["id"]),
    )
    await filter_svc.add_rule(rule)
    return {"ok": True}


@router.delete("/guilds/{guild_id}/filters/{pattern:path}")
async def delete_filter(request: Request, guild_id: int, pattern: str):
    require_guild_permission(request, guild_id)
    _, filter_svc, _ = _get_services(request)
    removed = await filter_svc.remove_rule(guild_id, pattern)
    if not removed:
        raise HTTPException(404, "Filter not found")
    return {"ok": True}


# --- Custom Commands ---

@router.get("/guilds/{guild_id}/commands")
async def get_commands(request: Request, guild_id: int):
    require_guild_permission(request, guild_id)
    _, _, cmd_svc = _get_services(request)
    cmds = await cmd_svc.get_all(guild_id)
    return [c.to_doc() for c in cmds]


@router.post("/guilds/{guild_id}/commands")
async def create_command(request: Request, guild_id: int, body: CustomCommandCreate):
    user = require_guild_permission(request, guild_id)
    _, _, cmd_svc = _get_services(request)
    cmd = CustomCommand(
        guild_id=guild_id,
        name=body.name,
        response=body.response,
        description=body.description,
        created_by=int(user["id"]),
    )
    ok = await cmd_svc.add(cmd)
    if not ok:
        raise HTTPException(409, f"Command '{body.name}' already exists")
    return {"ok": True}


@router.patch("/guilds/{guild_id}/commands/{name}")
async def update_command(
    request: Request, guild_id: int, name: str, body: CustomCommandUpdate
):
    require_guild_permission(request, guild_id)
    _, _, cmd_svc = _get_services(request)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "No fields to update")
    ok = await cmd_svc.update(guild_id, name, **fields)
    if not ok:
        raise HTTPException(404, "Command not found")
    return {"ok": True}


@router.delete("/guilds/{guild_id}/commands/{name}")
async def delete_command(request: Request, guild_id: int, name: str):
    require_guild_permission(request, guild_id)
    _, _, cmd_svc = _get_services(request)
    ok = await cmd_svc.remove(guild_id, name)
    if not ok:
        raise HTTPException(404, "Command not found")
    return {"ok": True}
